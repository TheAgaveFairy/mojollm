from std.sys.info import (
    simd_byte_width,
    num_logical_cores,
    simd_width_of,
    size_of,
)
from std.reflection import get_linkage_name
from layout import Layout, LayoutTensor
from std.random import randn  # , seed
from std.memory import memset, memset_zero, memcpy
from std.sys import stderr
from std.os import abort as os_abort

from attention import ftype, sftype, nelts
from arena import BumpArenaAllocator

comptime std_std_deviation = 0.5


def _arenaTensorHelper[
    layout: Layout, d_type: DType = ftype
](
    mut arena: BumpArenaAllocator, *, random: Bool = False, std: Float64 = 0.02
) -> LayoutTensor[d_type, layout, MutAnyOrigin]:
    var offset_before = arena.offset
    var ptr = arena.alloc[Scalar[d_type]](comptime (layout.size()))
    var offset_after = arena.offset
    var expected = comptime (layout.size()) * size_of[Scalar[d_type]]()
    var actual = offset_after - offset_before
    if (
        expected != actual
    ):  # TODO: not sure this should be needed, Arena should handle?
        os_abort(
            "Allocation failure! Expected: {} != Actual: {}".format(
                expected, actual
            )
        )
    var tensor = LayoutTensor[d_type, layout, MutAnyOrigin](ptr)
    if random:
        randn(ptr, comptime (layout.size()), 0, std)
    return tensor


@always_inline("nodebug")
def _trans[layout: Layout]() -> Layout:
    # debug_assert[assert_mode = "safe"](layout.shape[0].is_value(), "this was causing corruption")
    comptime m = layout.shape[0].value()
    comptime n = layout.shape[1].value()
    return Layout.row_major(n, m)


def _myTensorCopyFrom[
    layout_a: Layout, layout_b: Layout, d_type: DType
](
    *,
    src: LayoutTensor[d_type, layout_a, _],
    dest: LayoutTensor[d_type, layout_b, MutAnyOrigin],
    transposed: Bool = False,
):
    """Automagically handles transposition."""
    comptime assert (
        layout_a.rank() == layout_b.rank()
    ), "Invalid tensor ranks at _myTensorCopyFrom"
    comptime m = src.shape[0]()
    comptime n = src.shape[1]()
    comptime mm = dest.shape[0]()
    comptime nn = dest.shape[1]()
    comptime equal = m == mm and n == nn
    comptime transposed_valid = m == nn and n == mm
    comptime valid = equal or transposed_valid and layout_a.size() == layout_b.size()  # just to be safe
    comptime assert valid, "Invalid tensor shapes at _myTensorCopyFrom"

    if not transposed:
        memcpy(dest=dest.ptr, src=src.ptr, count=comptime (layout_a.size()))
    else:
        # DO NOT PARAMETERIZE THE FOR LOOPS!
        for i in range(m):
            for j in range(n):
                dest[j, i] = src[i, j]
                # dest.ptr[j * m + i] = src.ptr[i * n + j]


@always_inline("nodebug")
def fillTensorRand[
    layout: Layout
](
    x: LayoutTensor[ftype, layout, MutAnyOrigin],
    std: Float64 = std_std_deviation,
):
    """Fills an existing LayoutTensor with a random, normal distribution."""
    randn(x.ptr, comptime (layout.size()), 0, std)  # mean of zero


@always_inline("nodebug")
def randTensorHeap[
    layout: Layout
](std: Float64 = std_std_deviation) -> LayoutTensor[
    ftype, layout, MutAnyOrigin
]:
    """Heap allocates and returns a LayoutTensor filled with a random normal distribution.
    """
    var storage = alloc[sftype](comptime (layout.size()))
    randn(storage, comptime (layout.size()), 0, 1)
    return LayoutTensor[ftype, layout, MutAnyOrigin](storage)


@always_inline("nodebug")
def zeroTensorHeap[
    layout: Layout
]() -> LayoutTensor[ftype, layout, MutAnyOrigin]:
    """Heap allocates and returns a LayoutTensor filled with zeros."""
    var storage = alloc[sftype](comptime (layout.size()))
    memset_zero(storage, comptime (layout.size()))
    return LayoutTensor[ftype, layout, MutAnyOrigin](storage)


@always_inline("nodebug")
def cleanFunctionName[func: def() -> None]() -> String:
    """Simple reflection."""
    var func_name = get_linkage_name[func]().split("(")[0]
    func_name = func_name.split("[")[0].split("::")[1]
    return func_name


def showProgress(progress: Int, total: Int):
    """Simple progress bar. You'll want to print a newline when complete."""
    comptime bar_width = 50

    if progress == (total - 1):
        print("\r[", end="")
        for _ in range(bar_width):
            print("=", end="")
        print("] TASK COMPLETE!")
    else:
        var ratio = Float64(progress) / Float64(total)
        var filled = Int(bar_width * ratio)
        print("\r[", end="")
        for _ in range(filled):
            print("=", end="")
        for _ in range(filled, bar_width):
            print(" ", end="")
        print("]", round(ratio * 100, 1), "%", end="")


def systemInfo[ftype: DType]():
    """Helps display SIMD capabilities on your machine."""
    comptime nelts = simd_width_of[ftype]()
    var output = """
Your machine has multi-core and SIMD support as:
{} logical cores,
{} SIMD byte width,
{} width for the model using {}.""".format(
        num_logical_cores(), simd_byte_width(), nelts, ftype
    )
    print(output)


def compareBuffers(
    a: UnsafePointer[sftype, _],  # infer origins
    b: UnsafePointer[sftype, _],
    length: Int,
    epsilon: sftype = 1e-6,
) -> Bool:
    """Ensures two buffers are equal within some error bounds (floating point isn't perfect).
    """
    # comptime epsilon = 1e-6
    for i in range(length):
        if (a[i] < (b[i] - epsilon)) or (a[i] > (b[i] + epsilon)):
            print("element i", i, ":", a[i], "!=", b[i], file=stderr)
            return False
    return True


struct ColorsEnum:
    comptime COLOR_RESET = "\x1b[0m"
    comptime COLOR_WHITE = "\x1b[37"
    comptime COLOR_GREEN = "\x1b[32m"
    comptime COLOR_RED = "\x1b[31m"
    comptime COLOR_BLUE = "\x1b[34m"
    comptime COLOR_YELLOW = "\x1b[33m"
    comptime COLOR_PURPLE = "\x1b[35m"


def coloredString(
    str: String, color: String = ColorsEnum.COLOR_YELLOW
) -> String:
    return String(color + str + ColorsEnum.COLOR_RESET)
