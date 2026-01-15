from layout import Layout, LayoutTensor
from memory import memset_zero
from reflection import struct_field_types, struct_field_names, struct_field_count, get_type_name
from sys import stderr
from sys.info import size_of, align_of

from attention import ftype, sftype, Weights

comptime itype = DType.uint16
comptime sitype  = Scalar[itype]

struct TestWeights(Weights):
    comptime layout = Layout.row_major(7)
    var a: LayoutTensor[ftype, Self.layout, MutAnyOrigin]

    fn __init__(out self):
        self.a = type_of(self.a)(alloc[sftype](self.a.layout.size())).fill(0.0)

    @staticmethod
    fn initRandom(out self: Self, std: Float64 = 0.02):
        self = Self()
        _ = self.a.fill(1.0)

    fn freeMemory(mut self):
        self.a.ptr.free()

struct TestContainer():
    comptime layout = Layout.row_major(5)
    var a: LayoutTensor[ftype, Self.layout, MutAnyOrigin]
    var sub_weights: TestWeights

    fn __init__(out self):
        self.a = type_of(self.a)(alloc[sftype](self.a.layout.size())).fill(0.0)
        self.sub_weights = TestWeights.initRandom()

    fn __del__(deinit self):
        pass
        #self.a.ptr.free()
        #self.sub_weights.a.ptr.free()

struct ClaudeArena():
    # TODO: return Spans?
    """
    Simple bump allocator. Allocates sequentially from a buffer.
    """
    var buffer: UnsafePointer[UInt8, MutAnyOrigin]
    var capacity: Int
    var offset: Int  # Current allocation position
    
    fn __init__(out self, capacity_bytes: Int):
        self.buffer = alloc[UInt8](capacity_bytes)
        self.capacity = capacity_bytes
        self.offset = 0
    
    fn __del__(deinit self):
        self.buffer.free()
    
    fn alloc[T: AnyType](mut self, count: Int = 1) raises -> UnsafePointer[T, MutAnyOrigin]:
        """Allocate space for `count` items of type T."""
        var size = size_of[T]() * count
        var alignment = align_of[T]()
        
        # Align offset
        var aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1)
        
        # Check capacity
        if aligned_offset + size > self.capacity:
            # Could auto-grow here, or just panic
            #print("Arena out of memory!", file=stderr)
            raise Error("Arena out of memory!")
            #return UnsafePointer[T]()
        
        # Bump the pointer
        var ptr = (self.buffer + aligned_offset).bitcast[T]()
        self.offset = aligned_offset + size
        
        #print("allocating", String(count), get_type_name[T](), "begin", Int(ptr), "->", Int(ptr + count))
        return ptr
    
    fn reset(mut self):
        """Free all allocations at once by resetting the offset."""
        self.offset = 0
        # Memory is still there, just reusable
    
    fn clear(mut self):
        """Reset and zero out memory."""
        memset_zero(self.buffer, self.capacity)
        self.offset = 0

struct ArenaBumpAllocator():
    # TODO: return spans?
    """
    Since every bit of data in here will be of 'sftype', we will use that as
    the base unit instead of a more traditional "bytes" approach.

    However, that does mean if we want to have mixed precision, have the uint16
    token layer in this arena, etc, then that will have to change.

    This Arena offers two main benefits: loading and saving the model becomes a
    very simple operation, and there might be some performance gains.
    """
    var buffer: UnsafePointer[sftype, MutAnyOrigin]
    var capacity: Int
    var offset: Int

    fn __init__(out self, capacity: Int):
        self.buffer = alloc[sftype](capacity)
        self.capacity = capacity
        self.offset = 0

    fn alloc(mut self, count: Int) raises -> UnsafePointer[sftype, MutAnyOrigin]:
        if self.offset + count > self.capacity:
            raise Error("Out of memory in arena!")
    
        var pointer = self.buffer + self.offset
        self.offset += count
        print("allocating", String(count), get_type_name[sftype](), "begin", Int(pointer), "->", Int(pointer + count))
        return pointer

    fn reset(mut self):
        self.offset = 0

    fn clear(mut self):
        memset_zero(self.buffer, self.capacity)
        self.offset = 0

fn calcSize() -> Int:
    #comptime result = TestContainer.layout.size() + TestWeights.layout.size()
    var result = 0
    result += 5
    result += 7
    result += 3
    return result

fn printFields[T: AnyType]():
    print(get_type_name[T](), "has fields:")
    comptime f_types = struct_field_types[T]()
    comptime f_names = struct_field_names[T]()
    @parameter
    for i in range(struct_field_count[T]()):
        print("\t", f_names[i], ":", get_type_name[f_types[i]]())

fn printTypeInfo[T: DType]():
    comptime thing = "{}:\n\tsize: {}, align {}".format(T, size_of[Scalar[T]](), align_of[Scalar[T]]())
    print(thing)

fn main():
    printTypeInfo[ftype]()
    printTypeInfo[itype]()
    print("Some reflection tests...")
    comptime test_weights = TestWeights()
    comptime T = type_of(test_weights)
    printFields[T]()
    print("Arena time...")

    # running this function is possible at compile time, confirmed
    comptime size = calcSize()
    
    print("Simple, safe arena:")
    var dumb_arena = ArenaBumpAllocator(size)
    try:
        var p0 = dumb_arena.alloc(5)
        var p1 = dumb_arena.alloc(7)
        var _test = dumb_arena.alloc(3)
    except e:
        print(e)

    print("Claude's 'smarter' arena:")
    var size_in_bytes = 12 * size_of[sftype]() + size_of[sitype]() * 3
    var c_arena = ClaudeArena(size_in_bytes)
    try:
        var p0 = c_arena.alloc[sftype](5)
        var p1 = c_arena.alloc[sftype](7)
        # DIFFERENT TYPE BEING ALLOCATED
        var p2 = c_arena.alloc[sitype](3)
    except e:
        print(e)

