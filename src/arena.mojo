# for reflection
from layout import Layout, LayoutTensor
from reflection import (
    struct_field_types,
    struct_field_names,
    struct_field_count,
    get_type_name,
)
from attention import Weights, ModelParams

# for allocation arena
from memory import memset_zero
from sys import stderr
from sys.info import size_of, align_of
from testing import assert_equal, TestSuite
from os import abort

comptime ftype = DType.float32
comptime sftype = Scalar[ftype]
comptime itype = DType.uint16
comptime sitype = Scalar[itype]


struct BumpArenaAllocator(Copyable, ImplicitlyCopyable):
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True
    # TODO: return Spans?
    """
    Simple bump allocator. Minor design help from Claude 4.5.
    """
    var buffer: UnsafePointer[UInt8, MutAnyOrigin]
    var capacity: Int
    var offset: Int

    fn __init__(
        out self, capacity_bytes: Int, extra_space_factor: Float64 = 0.0
    ):
        var expanded_size = capacity_bytes + Int(
            capacity_bytes * extra_space_factor
        )
        self.buffer = alloc[UInt8](expanded_size)
        self.capacity = capacity_bytes
        self.offset = 0

    fn __del__(deinit self):
        # TODO: I think we can reinstate the self-clearing
        # print("BumpArenaAllocator __del__()")
        pass
        # self.buffer.free()

    fn alloc[
        T: AnyType
    ](mut self, count: Int = 1) -> UnsafePointer[T, MutAnyOrigin]:
        """Allocate space for `count` items of type T."""
        var size = size_of[T]() * count
        var alignment = align_of[T]()

        var aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1)

        if aligned_offset + size > self.capacity:
            # Could auto-grow here, or just panic
            abort("Arena out of memory! Aborting!")
            # raise Error("Arena out of memory!")

        var ptr = (self.buffer + aligned_offset).bitcast[T]()
        self.offset = aligned_offset + size

        # print("allocating", String(count), get_type_name[T](), "begin", Int(ptr), "->", Int(ptr + count))
        return ptr

    fn reset(mut self):
        """Free all allocations at once by resetting the offset."""
        self.offset = 0
        # Memory is still there, just reusable

    fn clear(mut self):
        """Reset and zero out memory."""
        memset_zero(self.buffer, self.capacity)
        self.offset = 0


def main():
    """Tests here. Some reflection examples to start if you `uncomment`."""
    _ = """
    printTypeInfo[ftype]()
    printTypeInfo[itype]()
    print("Some reflection tests...")
    comptime test_weights = TestWeights()
    comptime T = type_of(test_weights)
    printFields[T]()
    print("Arena time...")
    """
    printFields[ModelParams]()
    test_nested_arena()

    var suite = TestSuite()
    suite.test[test_allocator_offsets]()
    # suite.test[test_allocation_failure]() # now panics/aborts instead of raises
    suite.test[test_allocator_clear]()
    suite.test[test_allocator_reset]()
    suite^.run()


fn printFields[T: AnyType]():
    """Testing new reflection features."""
    print(get_type_name[T](), "has fields:")
    comptime f_types = struct_field_types[T]()
    comptime f_names = struct_field_names[T]()

    @parameter
    for i in range(struct_field_count[T]()):
        print("\t", f_names[i], ":", get_type_name[f_types[i]]())


fn printTypeInfo[T: DType]():
    """Prints type name, size, and alignment."""
    comptime thing = "{}:\n\tsize: {}, align {}".format(
        T, size_of[Scalar[T]](), align_of[Scalar[T]]()
    )
    print(thing)


def test_allocator_offsets():
    var size_in_bytes = 12 * size_of[sftype]() + size_of[sitype]() * 3
    var c_arena = BumpArenaAllocator(size_in_bytes)
    try:
        var p0 = c_arena.alloc[sftype](5)
        var p1 = c_arena.alloc[sftype](7)
        # DIFFERENT TYPE BEING ALLOCATED
        var p2 = c_arena.alloc[sitype](3)
        var end = p2 + 3 * size_of[sitype]()

        var size0 = Int(p1) - Int(p0)
        var size1 = Int(p2) - Int(p1)
        var size2 = Int(end) - Int(p2)

        assert_equal(size0, 20)  # 5 float32
        assert_equal(size1, 28)
        assert_equal(size2, 12)
    except e:
        print(e)
        assert_equal(0, -1)


@deprecated("alloc now aborts instead of raises")
def test_allocation_failure():
    var arena = BumpArenaAllocator(5)
    try:
        var ptr = arena.alloc[sftype](10)
    except e:
        _ = e
        assert_equal(0, 0)


def test_allocator_clear():
    var arena = BumpArenaAllocator(128)
    var ptr = arena.alloc[sitype](10)
    for i in range(10):
        ptr[i] = 69
    arena.clear()
    for i in range(10):
        assert_equal(ptr[i], 0)


def test_allocator_reset():
    var arena = BumpArenaAllocator(128)
    var ptr0 = arena.alloc[UInt8](128)
    arena.reset()
    var ptr1 = arena.alloc[UInt8](128)
    assert_equal(ptr0, ptr1)


def test_nested_arena():
    var tc = TestContainer()  # allocates Arena itself
    var tw_arena = BumpArenaAllocator(7 * size_of[sftype]())
    var tw = TestWeights(tw_arena)

    print(tw.a.ptr, tc.a.ptr, tc.sub_weights.a.ptr)


struct TestWeights(Weights):
    var arena: BumpArenaAllocator

    comptime layout = Layout.row_major(7)
    var a: LayoutTensor[ftype, Self.layout, MutAnyOrigin]

    fn __init__(out self, arena: BumpArenaAllocator):
        self.arena = arena
        self.a = type_of(self.a)(
            self.arena.alloc[sftype](self.a.layout.size())
        ).fill(3.0)

    @staticmethod
    fn sizeInBytes() -> Int:
        return Self.layout.size() * size_of[ftype]()

    @staticmethod
    fn initRandom(
        out self: Self, arena: BumpArenaAllocator, std: Float64 = 0.02
    ):
        self = Self(arena)
        _ = self.a.fill(1.0)

    fn freeMemory(mut self):
        self.a.ptr.free()


struct TestContainer:
    var arena: BumpArenaAllocator

    comptime layout = Layout.row_major(5)
    var a: LayoutTensor[ftype, Self.layout, MutAnyOrigin]
    var sub_weights: TestWeights

    fn __init__(out self):
        self.arena = type_of(self.arena)(
            self.sizeInBytes() + TestWeights.sizeInBytes()
        )
        self.a = type_of(self.a)(
            self.arena.alloc[sftype](self.a.layout.size())
        ).fill(1.0)
        self.sub_weights = TestWeights.initRandom(self.arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        return Self.layout.size() * size_of[sftype]()

    fn __del__(deinit self):
        pass
        # self.a.ptr.free()
        # self.sub_weights.a.ptr.free()
