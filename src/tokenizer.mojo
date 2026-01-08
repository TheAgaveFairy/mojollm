from layout import Layout, LayoutTensor
from math import sqrt, exp, log, ceildiv
from random import random_float64, random_si64, randint, randn, rand, seed
from sys.info import (
    simd_bit_width,
    simd_byte_width,
    simd_width_of,
    num_logical_cores,
)  # sizeof moved
from sys import stderr, is_big_endian, argv
from utils.index import IndexList
import os
from memory import memcpy, memset, memset_zero
from time import perf_counter_ns
from algorithm.functional import vectorize, parallelize
from reflection import get_linkage_name
from compile import compile_info
import benchmark  # run, Unit.ms
from hashlib.hasher import Hasher, default_hasher
from utils.lock import BlockingSpinLock
from os.atomic import Atomic
from collections import Set

from helpers import (
    showProgress,
    cleanFunctionName,
    compareBuffers,
    fillTensorRand,
)

from attention import ftype, sftype, nelts

struct Pair(Hashable, Copyable, ImplicitlyDestructible, Equatable, ImplicitlyCopyable, Representable):
    """
    Stores a hashable pair of Ints. There's probably another way to do this,
    but the motivation is that our vocab size (indexes) will probably be small
    enough to always fit in half of an Int64, allowing us to smash a pair of
    numbers side by side to get unique hashes consistently.

    Example with 4-bit indexes as 8-bit:
    a: 1101
    b: 0101
    => bit-shift "a" to the left and add yields the unique =>
    hash_me: 11010101

    GPT-2 has a vocab size of about 50k, later models perhaps double that.
    This will easily fit in an int32, meaning an Int64 can always fit everything we could ever need. Note, we can ignore the "sign" bit, it doesn't affect anything for simple
    hashing.
    """
    comptime __copyinit__is_trivial = True
    comptime DUMMY = Self(-1, -1)
    comptime delimeter = '|'

    # data must be BIGGER than a Byte to allow for expansion
    var a: Int
    var b: Int

    fn __init__(out self, a: Int, b: Int):
        self.a = a
        self.b = b

    @staticmethod
    @always_inline("nodebug")
    fn fromRepr(repr: StringSlice) raises -> Self:
        # '(123|456)' - watch out for leading spaces
        var stripped_repr = repr[1:-1] # remove ()
        var pair_strs = stripped_repr.split(Self.delimeter)
        try:
            var a = Int(pair_strs[0])
            var b = Int(pair_strs[1])
            return Self(a, b)
        except e:
            raise e^

    fn __hash__[H: Hasher](self, mut hasher: H):
        var hash_me: Int64 = self.a << 16 & self.b
        hasher.update(hash_me)

    fn __eq__(self, other: Self) -> Bool:
        return self.a == other.a and self.b == other.b

    fn __repr__(self) -> String:
        # be kind to yourself and use a sane delimeter
        return "(" + String(self.a) + Self.delimeter + String(self.b) + ")"

struct ASCIITokenizer(Copyable, Movable):
    comptime PairCounter = Dict[Pair, Int]
    var vocab_encode: List[Pair] # "id" is "idx + 256"
    var vocab_size: Int
    var special_tokens: List[String]

    fn __init__(out self, desired_vocab_size: Int):
        var safe_desired_vocab_size = max(desired_vocab_size, 256)
        self.vocab_size = safe_desired_vocab_size # we'll BPE up to this number
        self.vocab_encode = type_of(self.vocab_encode)()
        self.special_tokens = type_of(self.special_tokens)()
    
    fn exportVocab(self) -> String:
        """
        For use with Claude 4.5's tsx visualizer.
        """
        var result = ""
        for i in range(len(self.vocab_encode)):
            var pair = self.vocab_encode[i]
            result += "{}: ({}, {})\n".format(256 + i, pair.a, pair.b)
        return result

    fn save(self, filename: String) raises:
        try:
            with open(filename, "w") as f:
                f.write(self.vocab_encode)
                f.write("\n")
                f.write(self.special_tokens)
        except e:
            raise e^

    @staticmethod
    fn load(filename: String) raises -> Self:
        var self = Self(0)
        try:
            with open(filename, "r") as f:
                var lines = f.read().split('\n')
                if len(lines) != 2:
                    raise Error(filename, " encoded wrong. Should have two lines.")
                # ex. encode line [(123, 456), (13, 37)]
                var encode_pairs = lines[0][1:-1].split(',')
                self.vocab_size = 256 + len(encode_pairs)

                try:
                    for pair_str in encode_pairs:
                        var pair = Pair.fromRepr(pair_str.strip())
                        self.vocab_encode.append(pair)
                except e:
                    print(e, file = stderr)
                var spec_toks = lines[1][1:-1] # TODO: implement + dont forget vocab size
                
        except e:
            raise e^
        return self^

    fn encode(self, text: StringSlice) -> List[Int]:
        var ids = Self.stringToTokenList(text)
        for i, pair in enumerate(self.vocab_encode):
            var token_id = i + 256
            ids = self._merge(ids, pair, token_id) 

        return ids^
            
    fn decode(self, tokens: List[Int]) raises -> String:
        var unmerged = tokens.copy()
        for i in range(len(self.vocab_encode) - 1, -1, -1): # LIFO
            var pair = self.vocab_encode[i]
            var token_id = 256 + i
            unmerged = self._unmerge(unmerged, pair, token_id)

        var bytes = List[Byte](capacity = len(unmerged))
        for n in unmerged:
            if n > 255:
                raise Error("Tokenizer error!", n, "> 255")
            bytes.append(Byte(n))

        return String(bytes = bytes)

    fn _unmerge(self, text_tokens: List[Int], pair: Pair, token_id: Int) -> List[Int]:
        var n = len(text_tokens)
        var unmerged = List[Int](capacity = n) # will be BIGGER than this at end
        
        for _, token in enumerate(text_tokens):
            if token != token_id:
                unmerged.append(token)
            else:
                unmerged.append(pair.a)
                unmerged.append(pair.b)

        return unmerged^

    @staticmethod
    @always_inline("nodebug")
    fn stringToTokenList(text: StringSlice) -> List[Int]:
        var bytes = text.as_bytes() # originally, we assume ASCII
        var n = len(bytes)
        var ids = List[Int](capacity = n)
        for b in bytes:
            ids.append(Int(b))
        return ids^

    fn trainParallelized(mut self, text: StringSlice, threads: Int):
        """
        Currently not actually thread safe. I can't figure out how to count
        all pairs into a global data structure without either blowing up memory
        usage by duplicating that into local copies that then get merged,
        or without better mutex support. For some reason, I can't make a
        List[Atomic[DType.int32]] or something, and a spinlock would probably
        burn too much.

        However, I postulate the following:
        Given a sufficiently large enough dataset, the number of actual
        collisions that would happen are small or rare. Additionally, even if
        collisions were to happen, it's highly unlikely that they would prevent
        us from our ultimate goal: find the most common byte pair. Worst case,
        we're probably selecting the "second place" candidate some rare number
        of times, which seems totally fine. Encode / decode will work fine.

        Therefore, we will proceed without thread safety!
        """
        #var threads = num_logical_cores()
        #var lock = BlockingSpinLock()

        print("Training with BPE to vocab size", self.vocab_size, " using", threads, "threads...")
        var ids = Self.stringToTokenList(text)

        var n = len(ids)
        var toks_per_chunk = n // threads # or ceildiv(n, threads)

        var num_merges = self.vocab_size - 256
        for i in range(num_merges):
            showProgress(i, num_merges)
            var s = 256 + i
            comptime itype = DType.int32
            comptime sitype = Scalar[itype]
            var pair_counts_global = List[sitype](length = s * s, fill = 0)
            @parameter
            fn parallelClosure(tid: Int):
                var chunk_start = toks_per_chunk * tid
                var chunk_end = min(n, toks_per_chunk * (tid + 1) + 1) # halo size of 1
                var local_ids_slice = ids[chunk_start : chunk_end]
                for j in range(len(local_ids_slice) - 1):
                    var a = ids[chunk_start + j]
                    var b = ids[chunk_start + j + 1]
                    #pair_counts_global[a * s + b] += 1 # "should" be Atomic Add
                    var temp_ptr = pair_counts_global.unsafe_ptr() + (a * s + b)
                    _ = Atomic[itype].fetch_add(temp_ptr, 1)
            parallelize[parallelClosure](threads)

            var global_max_count: sitype = 0
            var global_max_pair = Pair.DUMMY
            for j, count in enumerate(pair_counts_global):
                var a = j // s
                var b = j % s
                if count > global_max_count:
                    global_max_count = count
                    global_max_pair = Pair(a, b)

            self.vocab_encode.append(global_max_pair)
            ids = self._merge(ids, global_max_pair, s)
        print()

    fn train(mut self, text: StringSlice):
        """
        Byte Pair Encoding.
        """
        var num_merges = self.vocab_size - 256
        print("Training with BPE to vocab size", self.vocab_size, "...")
        var ids = Self.stringToTokenList(text)

        for i in range(num_merges):
            showProgress(i, num_merges)
            var most_common_pair_count = 0
            var most_common_pair = Pair.DUMMY # dummy values

            var s = 256 + i
            var pair_counts = self._countPairsIterative(ids, s) # returns an 's' x 's' "matrix"
            for j, count in enumerate(pair_counts):
                var a = j // s
                var b = j % s

                if count > most_common_pair_count:
                    most_common_pair_count = count
                    var pair = Pair(a, b)
                    most_common_pair = pair

            #print("MOST COMMON PAIR", String(256 + i), ":", most_common_pair.__repr__(), most_common_pair_count)
            self.vocab_encode.append(most_common_pair)
            ids = self._merge(ids, most_common_pair, s)
        print()

    fn _merge(self, text_tokens: List[Int], pair: Pair, token_id: Int) -> List[Int]:
        """
        We take in a read-only list of token ids and allocate new fresh memory.
        We know that the list will be shorter after merging, so we can know
        that pre-allocating memory at the same size as the incoming "text" will
        always work.
        This is "good enough" for our needs. This isn't a bottleneck.
        """
        var n = len(text_tokens)
        var merged = List[Int](capacity = n)

        var c = pair.a
        var d = pair.b

        var i = 0
        while i < n:
            var a = text_tokens[i]
            var b = text_tokens[i + 1]
            if i < n - 1 and (a == c and b == d):
                merged.append(token_id)
                i += 2
            else:
                merged.append(text_tokens[i])
                i += 1
        
        return merged^

    fn _countPairsIterative(self, text_tokens: List[Int], s: Int) -> List[Int]:
        var counts = List[Int](length = s * s, fill = 0)
        var a = text_tokens[0]
        var b = text_tokens[1]
        for i in range(len(text_tokens) - 1):
            #var a = text_tokens[i]
            #var b = text_tokens[i + 1]
            counts[a * s + b] += 1
            a = b
            b = text_tokens[i + 1]
        return counts^

    fn _countPairsIterativeNew(self, text_tokens: List[Int], s: Int) -> Pair:
        var counts = List[Int](length = s * s, fill = 0)
        var a = text_tokens[0]
        var b = text_tokens[1]
        var most = 0
        var pair = Pair.DUMMY
        for i in range(len(text_tokens) - 1):
            counts[a * s + b] += 1
            if counts[a * s + b] > most:
                most = counts[a * s + b]
                pair = Pair(a, b)
            a = b
            b = text_tokens[i + 1]
        return pair

    fn __copyinit__(out self, other: Self):
        self.vocab_size = other.vocab_size # we'll BPE up to this number
        self.vocab_encode = other.vocab_encode.copy()
        self.special_tokens = other.special_tokens.copy()

    fn __eq__(self, other: Self) -> Bool:
        var vs = self.vocab_size == other.vocab_size
        var pc = self.vocab_encode == other.vocab_encode
        var st = self.special_tokens == other.special_tokens
        return vs and pc and st

fn main():
    # tests
    var args = argv()
    var vocab_size = 500
    if len(args) > 1:
        try:
            vocab_size = Int(args[1])
        except e:
            print(e, file = stderr)
            return
    
    try:
        with open("./datasets/input.txt", "r") as f:
            var text = f.read()

            #var tokenizer = ASCIITokenizer(vocab_size) # 280 is enough to display recursion
            #tokenizer.train(text)
            
            _ = """
            var tok_multi = ASCIITokenizer(vocab_size)
            
            comptime runs = 1
            for r in range(runs):
                var start = perf_counter_ns()
                tokenizer.train(text)
                var mid = perf_counter_ns()
                tok_multi.trainParallelized(text, num_logical_cores())
                var end = perf_counter_ns()

                var verify = Set[Pair](tokenizer.vocab_encode)
                var n = len(tokenizer.vocab_encode) # vocab_size - 256
                var seen = 0
                for i in range(n):
                    if tok_multi.vocab_encode[i] in verify:
                        seen += 1
                
                var single_ms = (mid - start) // 1_000_000
                var multi_ms = (end - mid) // 1_000_000
                print(seen == n, ", single is", multi_ms / single_ms, "times faster.")

            _ = """
            #var encoded = tokenizer.encode(text)
            #var decoded = tokenizer.decode(encoded)

            #print("Encode -> Decode reverts to original?:", decoded == text)
            var filename = "bpe_{}.tok".format(vocab_size)
            tokenizer.save(filename)

            var tok2 = ASCIITokenizer.load(filename)
            print("Save / Load worked?:", tokenizer == tok2)
            benchmark.compiler.keep(tokenizer)
            var vocab_filename = "vocab_{}.txt".format(vocab_size)
            with open(vocab_filename, "w") as v:
                v.write(tokenizer.exportVocab())
    except e:
        print(e)
