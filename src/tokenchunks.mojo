from iter import Iterator, Iterable

@fieldwise_init
struct TokenChunks(Iterable, Copyable):
    """
    Instead of having a sane List[List[Int]], we're going to allocate flattened
    memory and mark the boundaries between each trainable chunk by using another
    array to hold the indexes of where each boundary starts.

    Ex:
    chunks: [[1,2,3], [4], [5, 6]] ==>

    tokens:[1,2,3,4,5,6]
    bounds: [0,3,4] # last chunk goes until the end
    """
    comptime IteratorType[iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]]: Iterator = ChunkIterator
    comptime Element = Int
    var tokens: List[Int]
    var boundaries: List[Int]

    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    fn __init__(out self):
        self.tokens = type_of(self.tokens)()
        self.boundaries = type_of(self.boundaries)()

    fn __init__(out self, tokens_capacity: Int):
        self.tokens = type_of(self.tokens)(capacity = tokens_capacity) # oversized
        self.boundaries = type_of(self.boundaries)()

    fn __init__(out self, tokens_capacity: Int, num_chunks: Int):
        self.tokens = type_of(self.tokens)(capacity = tokens_capacity) # oversized
        self.boundaries = type_of(self.boundaries)(capacity = num_chunks) # static

    #fn __init__(out self, tokens: List[Int], boundaries: List[Int]):
    #    self.tokens = tokens
    #    self.boundaries = boundaries

    fn addChunk(mut self, chunk: List[Int]):
        idx = len(self.tokens)
        for tok in chunk:
            self.tokens.append(tok)
        self.boundaries.append(idx)

    fn get(self, chunk_idx: Int) -> Optional[Span[Int, origin_of(self.tokens)]]:
        var num_chunks = len(self.boundaries)
        if chunk_idx >= num_chunks:
            return None
        var start = self.boundaries[chunk_idx]
        if chunk_idx == (num_chunks - 1):
            return self.tokens[start : ]
        var end = self.boundaries[chunk_idx + 1]
        return self.tokens[start : end]

    fn __iter__(ref self: Self) -> Self.IteratorType[origin_of(self)]:
        return ChunkIterator(self.tokens, self.boundaries)

@fieldwise_init
struct ChunkIterator(Iterator):
    comptime Element = Span[TokenChunks.Element, ImmutAnyOrigin] # tie to Iterable
    var tokens: Self.Element#Span[Int, ImmutAnyOrigin]
    var boundaries: Self.Element#Span[Int, ImmutAnyOrigin]
    var current: Int
    var num_tokens: Int # remember, last chunk doesn't have a clear "end" marker

    fn __init__(out self, tokens: Self.Element, boundaries: Self.Element):
        self.tokens = tokens
        self.boundaries = boundaries
        self.current = 0
        self.num_tokens = len(self.boundaries)

    fn __next__(mut self) raises StopIteration -> Self.Element:
        if self.current >= self.num_tokens:
            raise StopIteration()
        
        var start = self.boundaries[self.current]
        # REMEMBER! There is no token telling us when the end is (idx == len(boundaries))
        if self.current == (self.num_tokens - 1):
            self.current += 1
            return self.tokens[start : ]

        # else
        var end = self.boundaries[self.current + 1]
        self.current += 1
        return self.tokens[start : end]

fn main():
    print("Tests passed?", tests())

fn tests() -> Bool:
    var tokens = [1,2,3,4,5,6]
    var boundaries = [0,3,4]
    var my_iterable = TokenChunks(tokens^, boundaries^)
    my_iterable.addChunk([7,8,9])
    my_iterable.addChunk([10])
    var add_chunk_result = ""
    for chunk in my_iterable:
        for c in chunk:
            add_chunk_result += "{},".format(c)
        add_chunk_result += '|'

    #print(add_chunk_result)
    var add_chunk_passed = add_chunk_result == "1,2,3,|4,|5,6,|7,8,9,|10,|"
    
    var get_result = ""
    var chunk_zero = my_iterable.get(0)
    if chunk_zero:
        var iterable = chunk_zero.value()
        for c in iterable:
            get_result += "{},".format(c)
        get_result += '|'
    
    #print(result)
    var get_passed = get_result == "1,2,3,|"

    return add_chunk_passed and get_passed

