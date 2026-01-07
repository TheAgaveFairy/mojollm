from layout import Layout, LayoutTensor
from math import sqrt, exp, log, ceildiv
from random import random_float64, random_si64, randint, randn, rand, seed
from sys.info import (
    simd_bit_width,
    simd_byte_width,
    simd_width_of,
    num_logical_cores,
)  # sizeof moved
from sys import stderr, is_big_endian
from utils.index import IndexList
import os
from memory import memcpy, memset, memset_zero
from time import perf_counter_ns
from algorithm.functional import vectorize, parallelize
from reflection import get_linkage_name
from compile import compile_info
import benchmark  # run, Unit.ms
from hashlib.hasher import Hasher, default_hasher

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

    # TODO: should i make this explicitly [U]Int64 based?
    
    fn __init__(out self, desired_vocab_size: Int):
        var safe_desired_vocab_size = max(desired_vocab_size, 256)
        self.vocab_size = safe_desired_vocab_size # we'll BPE up to this number
        self.vocab_encode = type_of(self.vocab_encode)()
        self.special_tokens = type_of(self.special_tokens)()

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
            var most_common_pair = Pair(-1, -1) # dummy values

            var s = 256 + i
            var pair_counts = self._countPairsIterativeNew(ids, s) # returns an 's' x 's' "matrix"
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

        var i = 0
        while i < n:
            if i < n - 1 and Pair(text_tokens[i], text_tokens[i + 1]) == pair:
                merged.append(token_id)
                i += 2
            else:
                merged.append(text_tokens[i])
                i += 1
        
        return merged^

    fn _countPairsIterativeNew(self, text_tokens: List[Int], s: Int) -> List[Int]:
        var counts = List[Int](length = s * s, fill = 0)
        for i in range(len(text_tokens) - 1):
            var a = text_tokens[i]
            var b = text_tokens[i + 1]
            counts[a * s + b] += 1
        return counts^

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
    try:
        with open("./datasets/input.txt", "r") as f:
            var text = f.read()

            var tokenizer = ASCIITokenizer(500) # 280 is enough to display recursion
            tokenizer.train(text)

            var encoded = tokenizer.encode(text)
            var decoded = tokenizer.decode(encoded)

            print("Encode -> Decode reverts to original?:", decoded == text)
            comptime filename = "bpe.tok"
            tokenizer.save(filename)

            var tok2 = ASCIITokenizer.load(filename)
            print("Save / Load worked?:", tokenizer == tok2)
            
    except e:
        print(e)
