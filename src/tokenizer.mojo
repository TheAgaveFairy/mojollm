from layout import Layout, LayoutTensor
from math import sqrt, exp, log, ceildiv
from random import random_float64, random_si64, randint, randn, rand, seed
from sys.info import (
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

# from python import Python # for RegExp engines

from helpers import (
    showProgress,
    cleanFunctionName,
    compareBuffers,
    fillTensorRand,
)
from attention import ftype, sftype, nelts
from cliparser import TokenizerParser
from tokenchunks import TokenChunks


@fieldwise_init
struct Pair(
    Copyable,
    Equatable,
    Hashable,
    ImplicitlyCopyable,
    ImplicitlyDestructible,
    Representable,
    TrivialRegisterPassable,
):
    """
    Stores a hashable pair of Ints. There's probably another way to do this,
    but the motivation is that our vocab size (indexes) will probably be small
    enough to always fit in half of an Int64, allowing us to smash a pair of
    numbers side by side to get unique hashes consistently.

    Example with 4-bit indexes as 8-bit:
    a: 1101
    b: 0101
    => bit-shift "a" to the left 4 times and add "b" yields the unique =>
    hash_me: 11010101

    GPT-2 has a vocab size of about 50k, later models perhaps double that.
    This will easily fit in an int32, meaning an Int64 can always fit everything we could ever need. Note, we can ignore the "sign" bit, it doesn't affect anything for simple
    hashing.
    """

    # comptime __copyinit__is_trivial = True  # possibly redundant w/ @register_passable
    comptime DUMMY = Self(-1, -1)
    comptime delimeter = "|"

    # data must be BIGGER than a Byte to allow for expansion
    var a: Int
    var b: Int

    @staticmethod
    @always_inline("nodebug")
    fn fromRepr(repr: StringSlice) raises -> Self:
        """Not the safest but should be fine."""
        # var had_space: Int = 1 if repr[0] == ' ' else 0
        var had_space = 0
        var stripped_repr = repr[1 + had_space : -1]  # remove ()
        var pair_strs = stripped_repr.split(Self.delimeter)
        try:
            var a = Int(pair_strs[0])
            var b = Int(pair_strs[1])
            return Self(a, b)
        except e:
            raise e^

    fn __hash__[H: Hasher](self, mut hasher: H):
        var hash_me: Int64 = Int64(self.a) << 16 & Int64(self.b)
        hasher.update(hash_me)

    fn __eq__(self, other: Self) -> Bool:
        return self.a == other.a and self.b == other.b

    fn __repr__(self) -> String:
        return "(" + String(self.a) + Self.delimeter + String(self.b) + ")"


struct Tokenizer(Copyable, Movable):
    var vocab_encode: List[Pair]  # "id" is "idx + 256"
    var vocab_size: Int
    var special_tokens: List[String]

    fn __init__(out self, desired_vocab_size: Int):
        var safe_desired_vocab_size = max(desired_vocab_size, 256)
        self.vocab_size = safe_desired_vocab_size  # we'll BPE up to this number
        self.vocab_encode = type_of(self.vocab_encode)()
        self.special_tokens = type_of(self.special_tokens)()

    @staticmethod
    fn boundaryProtect(raw_ids: List[Int]) -> TokenChunks:
        """
        Splits into chunks based on: contractions | words | punctuation | whitespace
        See below comment for Pattern approximation (raw strings not supported yet).
        Claude 4.5 Sonnet refined my skeleton attempt. Verbose, but looks good.
        """
        # Pattern approximation: r"'(?i:[sdmt]|ll|ve|re)|\w+|[^\w\s]+|\s+".
        # TODO: add word beginning token ('220') and attach spaces to beginnings

        var n = len(raw_ids)
        var chunks = TokenChunks(n)  # conceptually a List[List[Int]]()
        var i = 0

        while i < n:
            var id = raw_ids[i]

            # Check for apostrophe contractions first
            if id == ord("'"):
                # Check for 'll, 've, 're (2 chars after ')
                if i + 2 < n:
                    var next_two = (raw_ids[i + 1], raw_ids[i + 2])
                    if (
                        next_two == (ord("l"), ord("l"))
                        or next_two == (ord("L"), ord("L"))
                        or next_two == (ord("v"), ord("e"))
                        or next_two == (ord("V"), ord("E"))
                        or next_two == (ord("r"), ord("e"))
                        or next_two == (ord("R"), ord("E"))
                    ):
                        var chunk = List[Int]()
                        chunk.append(raw_ids[i])
                        chunk.append(raw_ids[i + 1])
                        chunk.append(raw_ids[i + 2])
                        chunks.addChunk(chunk^)
                        i += 3
                        continue

                # Check for 's, 't, 'd, 'm (1 char after ')
                if i + 1 < n:
                    var next_one = raw_ids[i + 1]
                    if (
                        next_one == ord("s")
                        or next_one == ord("S")
                        or next_one == ord("t")
                        or next_one == ord("T")
                        or next_one == ord("d")
                        or next_one == ord("D")
                        or next_one == ord("m")
                        or next_one == ord("M")
                    ):
                        var chunk = List[Int]()
                        chunk.append(raw_ids[i])
                        chunk.append(raw_ids[i + 1])
                        chunks.addChunk(chunk^)
                        i += 2
                        continue

                # Just a lone apostrophe - treat as punctuation
                var chunk = List[Int]()
                chunk.append(id)
                chunks.addChunk(chunk^)
                i += 1

            # Whitespace - separate chunk
            elif (
                id == ord(" ")
                or id == ord("\n")
                or id == ord("\t")
                or id == ord("\r")
            ):
                var chunk = List[Int]()
                chunk.append(id)
                chunks.addChunk(chunk^)
                i += 1

            # Alphanumeric - collect into word
            elif (
                (id >= ord("a") and id <= ord("z"))
                or (id >= ord("A") and id <= ord("Z"))
                or (id >= ord("0") and id <= ord("9"))
            ):
                var chunk = List[Int]()
                while i < n:
                    var c = raw_ids[i]
                    if (
                        (c >= ord("a") and c <= ord("z"))
                        or (c >= ord("A") and c <= ord("Z"))
                        or (c >= ord("0") and c <= ord("9"))
                    ):
                        chunk.append(c)
                        i += 1
                    else:
                        break
                # chunks.addChunk(chunk^) # removed so we can check for "n't"

                # Check if word ends with 'n' and is followed by "'t"
                if len(chunk) > 0 and i + 2 <= n:
                    var last_char = chunk[len(chunk) - 1]
                    if (
                        (last_char == ord("n") or last_char == ord("N"))
                        and i + 1 < n
                        and raw_ids[i] == ord("'")
                        and i + 1 < n
                        and (
                            raw_ids[i + 1] == ord("t")
                            or raw_ids[i + 1] == ord("T")
                        )
                    ):
                        # Remove 'n' from word chunk
                        _ = chunk.pop()
                        chunks.addChunk(chunk^)

                        # Create "n't" chunk
                        var contraction = List[Int]()
                        contraction.append(last_char)  # n
                        contraction.append(raw_ids[i])  # '
                        contraction.append(raw_ids[i + 1])  # t
                        chunks.addChunk(contraction^)
                        i += 2
                        continue

                chunks.addChunk(chunk^)

            # Punctuation - separate chunk
            else:
                var chunk = List[Int]()
                chunk.append(id)
                chunks.addChunk(chunk^)
                i += 1

        # for chunk in chunks[:100]:
        #    print(chunk)
        return chunks^

    fn trainWithProtections(
        mut self, text: StringSlice, debug_display: Bool = False
    ):
        """
        Byte Pair Encoding. Text is assumed to be ASCII. Implements "regexp".
        """
        if debug_display:
            print("Training with BPE to vocab size", self.vocab_size, "...")
        var raw_ids: List[Int] = Self.stringToTokenList(text)
        var ids: TokenChunks = Self.boundaryProtect(raw_ids)

        var num_merges = self.vocab_size - 256
        for i in range(num_merges):
            if debug_display:
                showProgress(i, num_merges)

            var most_common_pair_count = 0
            var most_common_pair = Pair.DUMMY  # dummy values

            var s = 256 + i
            var pair_counts = self._countPairsChunks(
                ids, s
            )  # returns an 's' x 's' "matrix"
            for j, count in enumerate(pair_counts):
                var a = j // s
                var b = j % s

                if count > most_common_pair_count:
                    most_common_pair_count = count
                    var pair = Pair(a, b)
                    most_common_pair = pair

            self.vocab_encode.append(most_common_pair)
            ids = self._mergeChunks(ids, most_common_pair, s)

        if debug_display:
            print()

    fn encode(self, text: StringSlice) -> List[Int]:
        """Encodes some text into a List of our new token ids."""
        var ids = Self.stringToTokenList(text)
        for i, pair in enumerate(self.vocab_encode):
            var token_id = i + 256
            ids = self._merge(ids, pair, token_id)

        return ids^

    fn decode(self, tokens: List[Int]) raises -> String:
        """Decodes a list of token ids into the original text."""
        var unmerged = tokens.copy()
        for i in range(len(self.vocab_encode) - 1, -1, -1):  # LIFO
            var pair = self.vocab_encode[i]
            var token_id = 256 + i
            unmerged = self._unmerge(unmerged, pair, token_id)

        var bytes = List[Byte](capacity=len(unmerged))
        for n in unmerged:
            if n > 255:
                raise Error("Tokenizer error!", n, "> 255")
            bytes.append(Byte(n))

        return String(unsafe_from_utf8=bytes)

    fn exportVocab(self, filename: String):
        """For use with Claude 4.5's tsx visualizer.html (which has bugs but that's ok).
        """
        print("Exporting vocab for visualizer:", filename)
        try:
            with open(filename, "w") as f:
                var result = ""
                for i in range(len(self.vocab_encode)):
                    var pair = self.vocab_encode[i]
                    result += "{}: ({}, {})\n".format(256 + i, pair.a, pair.b)
                f.write(result)
        except e:
            print(e, file=stderr)

    fn save(self, filename: String) raises:
        """Saves as a plain text file, for now."""
        try:
            with open(filename, "w") as f:
                f.write(self.vocab_encode)
                f.write("\n")
                f.write(self.special_tokens)
                print("Saved tokenizer to:", filename)
        except e:
            raise e^

    @staticmethod
    fn load(filename: String) raises -> Self:
        """Undoes the save and loads with some minor protections."""
        var self = Self(0)
        try:
            with open(filename, "r") as f:
                var lines = f.read().split("\n")
                for i, line in enumerate(lines):
                    if not len(line):
                        _ = lines.pop(i)
                if len(lines) != 2:
                    # for line in lines:
                    #    print(line[:5], file = stderr)
                    raise Error(
                        filename,
                        " encoded wrong. Should have two lines. Had:",
                        len(lines),
                    )
                # ex. encode line [(123, 456), (13, 37)]
                var encode_pairs = lines[0][1:-1].split(",")
                self.vocab_size = 256 + len(encode_pairs)

                try:
                    for pair_str in encode_pairs:
                        var pair = Pair.fromRepr(pair_str.strip())
                        self.vocab_encode.append(pair)
                except e:
                    print(e, file=stderr)
                var spec_toks = lines[1][
                    1:-1
                ]  # TODO: implement + dont forget vocab size
                _ = spec_toks

        except e:
            raise e^
        return self^

    fn decodeToken(self, token_id: Int) -> String:  # could return List[Byte]
        """
        Might be faster than "decode" for a single token.
        """
        var internal: List[Int] = [token_id]

        var replaced = True
        while replaced:
            replaced = False
            var i = 0
            while i < len(internal):
                var tok = internal[i]
                if tok > 255:
                    replaced = True
                    var its_pair = self.vocab_encode[tok - 256]
                    internal[i] = its_pair.a
                    internal.insert(i + 1, its_pair.b)
                else:
                    i += 1

        var result = String(unsafe_from_utf8=[Byte(x) for x in internal])
        return result

    fn _merge(
        self, text_tokens: List[Int], pair: Pair, token_id: Int
    ) -> List[Int]:
        """
        We take in a read-only list of token ids and allocate new fresh memory.
        """
        var n = len(text_tokens)
        var merged = List[Int](capacity=n)

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

    fn _unmerge(
        self, text_tokens: List[Int], pair: Pair, token_id: Int
    ) -> List[Int]:
        """Used for decode. Looks for compound tokens and replaces them with parts.
        """
        var n = len(text_tokens)
        comptime extra_capacity_factor = 0.125  # could tune this to avoid reallocing
        var bonus = Int(Float64(n) * extra_capacity_factor)
        var unmerged = List[Int](
            capacity=n + bonus
        )  # might be BIGGER than this at end

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
        """Splits into bytes as Ints. UTF-8 or ASCII doesn't really matter."""
        var bytes = text.as_bytes()
        var n = len(bytes)
        var ids = List[Int](capacity=n)
        for b in bytes:
            ids.append(Int(b))
        return ids^

    fn _mergeChunks(
        self, read chunks: TokenChunks, pair: Pair, new_token_id: Int
    ) -> TokenChunks:
        """We take in a read-only list of token ids and allocate new fresh memory.
        """
        var tokens_capacity = len(chunks.tokens)
        var num_chunks = len(chunks.boundaries)
        var merged = TokenChunks(
            tokens_capacity, num_chunks
        )  # capacity won't get filled, that's ok!

        var c = pair.a
        var d = pair.b

        for chunk in chunks:
            var m = len(chunk)
            var temp = List[Int]()
            var i = 0
            while i + 1 < m:
                var a = chunk[i]
                var b = chunk[i + 1]
                if a == c and b == d:
                    temp.append(new_token_id)
                    i += 2
                else:
                    temp.append(chunk[i])
                    i += 1
            if i < m:
                temp.append(chunk[i])
            merged.addChunk(temp^)

        return merged^

    fn _countPairsChunks(self, chunks: TokenChunks, s: Int) -> List[Int]:
        """
        For the Shakespeare dataset, the max counted pair is seen 27643 times.
        We could probably save a lot of memory by using [U]Int16, need be on a
        giant training dataset.
        """
        var counts = List[Int](length=s * s, fill=0)  # UInt16
        for chunk in chunks:
            for i in range(len(chunk) - 1):
                var a = chunk[i]
                var b = chunk[i + 1]
                counts[a * s + b] += 1
        return counts^

    fn __copyinit__(out self, copy: Self):
        self.vocab_size = copy.vocab_size  # we'll BPE up to this number
        self.vocab_encode = copy.vocab_encode.copy()
        self.special_tokens = copy.special_tokens.copy()

    fn __eq__(self, other: Self) -> Bool:
        var vs = self.vocab_size == other.vocab_size
        var pc = self.vocab_encode == other.vocab_encode
        var st = self.special_tokens == other.special_tokens
        return vs and pc and st


fn main():
    # tests
    var config = TokenizerParser()

    try:
        with open(config.input_filename, "r") as f:
            var text = f.read()

            var tokenizer = Tokenizer(
                config.vocab_size
            )  # ~280 is enough to display recursion
            tokenizer.trainWithProtections(text, True)
            # print("Decode encode test result:", decodeEncodeTest(tokenizer, text))
            # print(showExample(tokenizer, tokenizer.encode(text[:500])))
            tokenizer.save(config.save_name)

            var vocab_filename = "models/vocab_{}.txt".format(config.vocab_size)
            tokenizer.exportVocab(vocab_filename)

            # var ratio = compareTrainingTimesTest(text, vocab_size = 1000, runs = 5)
            _ = """
            var tok2 = Tokenizer.load("bpe_25000.tok")
            var test_ids = tok2.encode(text[:200])
            for id in test_ids:
                print(String(id), "\t", tok2.decodeToken(id))

            with open(vocab_filename, "w") as v:
                v.write(tokenizer.exportVocab())
            """
    except e:
        print(e)


fn saveLoadTest(tokenizer: Tokenizer) -> Bool:
    var filename = "bpe_{}.tok".format(tokenizer.vocab_size)
    try:
        tokenizer.save(filename)
        var tok2 = Tokenizer.load(filename)
        return tokenizer == tok2
    except e:
        print(e, file=stderr)
        return False


fn decodeEncodeTest(tokenizer: Tokenizer, text: StringSlice) -> Bool:
    try:
        return text == tokenizer.decode(tokenizer.encode(text))
    except e:
        print(e, file=stderr)
        return False


fn compareVocabsTest(a: Tokenizer, b: Tokenizer) -> Bool:
    """
    Takes in two tokenizers to compare their vocabs.
    'a' is the reference implementation, and 'b' is checked against it.
    Could also 'return round(seen / n, 4) * 100'.
    """
    var verify = Set[Pair](a.vocab_encode)
    var n = len(a.vocab_encode)  # vocab_size - 256
    var seen = 0
    for i in range(n):
        if b.vocab_encode[i] in verify:
            seen += 1

    return seen == n


@deprecated("No other training methods implemented any longer.")
fn compareTrainingTimesTest(
    text: StringSlice, vocab_size: Int = 500, runs: Int = 5
) -> Float64:
    var tok_a = Tokenizer(vocab_size)
    var tok_b = Tokenizer(vocab_size)

    var accum = 0.0
    for r in range(runs):
        var start = perf_counter_ns()
        # tok_a.train(text)
        var mid = perf_counter_ns()
        # tok_b.trainParallelized(text, num_logical_cores())
        tok_b.trainWithProtections(text)
        var end = perf_counter_ns()

        var single_ms = (mid - start) // 1_000_000
        var multi_ms = (end - mid) // 1_000_000
        var ratio = multi_ms / single_ms
        accum += ratio
        print(
            compareVocabsTest(tok_a, tok_b),
            ", single is",
            ratio,
            "times faster.",
        )
    return accum / runs


fn showExample(tokenizer: Tokenizer, encoded: List[Int]) -> String:
    var result = ""
    comptime max_len = 250
    for tok in encoded[:max_len]:
        var temp = tokenizer.decodeToken(tok)
        # if temp[-1] == "\n":
        #    temp = String(temp[:-1]) + "\\n"
        result += String(tok) + "\t||" + temp + "\n"
    return result^
