from std.iter import Iterator, Iterable
from std.testing import TestSuite, assert_equal, assert_true

# from llm import token_itype as itype
# comptime sitype = Scalar[itype]


# @fieldwise_init
struct TokenChunks(Copyable):  # , Iterable):
    """
    Instead of having a sane List[List[Int]], we're going to allocate flattened
    memory and mark the boundaries between each trainable chunk by using another
    array to hold the indexes of where each boundary starts.

    Ex:
    chunks: [[1,2,3], [4], [5, 6]] ==>

    tokens: [1,2,3,4,5,6]
    bounds: [0,3,4] # last chunk goes until the end
    """

    var tokens: List[Int]
    var boundaries: List[Int]

    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    def __init__(out self):
        self.tokens = type_of(self.tokens)()
        self.boundaries = type_of(self.boundaries)()

    def __init__(out self, tokens: List[Int], boundaries: List[Int]):
        self.tokens = tokens.copy()
        self.boundaries = boundaries.copy()

    def __init__(out self, tokens_capacity: Int):
        self.tokens = type_of(self.tokens)(capacity=tokens_capacity)
        self.boundaries = type_of(self.boundaries)()

    def __init__(out self, tokens_capacity: Int, num_chunks: Int):
        self.tokens = type_of(self.tokens)(capacity=tokens_capacity)
        self.boundaries = type_of(self.boundaries)(capacity=num_chunks)

    def addChunk(mut self, chunk: List[Int]):
        """Automatically calculates boundary and extends the tokens list."""
        var idx = len(self.tokens)
        for tok in chunk:
            self.tokens.append(tok)
        self.boundaries.append(idx)

    def get(self, chunk_idx: Int) -> Optional[Span[Int, origin_of(self.tokens)]]:
        """Returns an optional Span of the requested token_ids if the chunk_idx is valid.
        """
        var num_chunks = len(self.boundaries)
        if chunk_idx >= num_chunks:
            return None
        var start = self.boundaries[chunk_idx]
        if chunk_idx == (num_chunks - 1):
            return self.tokens[start:]
        var end = self.boundaries[chunk_idx + 1]
        return self.tokens[start:end]


def main() raises:
    var suite = TestSuite()
    suite.test[testAddChunk]()
    # suite.test[testGetChunk]()
    suite.test[testDumbIteration]()
    suite^.run()


def testDumbIteration() raises:
    var tokens = [1, 2, 3, 4, 5, 6]
    var boundaries = [0, 3, 4]
    var tc = TokenChunks(tokens^, boundaries^)

    var result = ""
    var i = 0
    while tc.get(i):
        result += String(tc.get(i).value()) + "|"
        # print(i, tc.get(i).value())
        i += 1
    #print(result)
    assert_equal(result, "[1, 2, 3]|[4]|[5, 6]|")


def testAddChunk() raises:
    var tc = TokenChunks()
    tc.addChunk([1, 2, 3])
    tc.addChunk([4])
    tc.addChunk([5, 6])

    assert_equal(String(tc.get(0).value()), "[1, 2, 3]")
    assert_equal(String(tc.get(1).value()), "[4]")
    assert_equal(String(tc.get(2).value()), "[5, 6]")
    assert_true(tc.get(3) is None)
