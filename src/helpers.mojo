from sys.info import simd_byte_width, num_logical_cores, simd_width_of
from reflection import get_linkage_name
from layout import Layout, LayoutTensor
from random import randn  # , seed
from memory import memset, memset_zero
from sys import stderr

from attention import ftype, sftype, nelts

comptime std_std_deviation = 0.5


@always_inline("nodebug")
fn fillTensorRand[
    layout: Layout
](
    x: LayoutTensor[ftype, layout, MutAnyOrigin],
    std: Float64 = std_std_deviation,
):
    """Fills an existing LayoutTensor with a random, normal distribution."""
    randn(x.ptr, comptime(layout.size()), 0, std)  # mean of zero


@always_inline("nodebug")
fn randTensorHeap[
    layout: Layout
](std: Float64 = std_std_deviation) -> LayoutTensor[
    ftype, layout, MutAnyOrigin
]:
    """Heap allocates and returns a LayoutTensor filled with a random normal distribution.
    """
    var storage = alloc[sftype](comptime(layout.size()))
    randn(storage, comptime(layout.size()), 0, 1)
    return LayoutTensor[ftype, layout, MutAnyOrigin](storage)


@always_inline("nodebug")
fn zeroTensorHeap[
    layout: Layout
]() -> LayoutTensor[ftype, layout, MutAnyOrigin]:
    """Heap allocates and returns a LayoutTensor filled with zeros."""
    var storage = alloc[sftype](comptime(layout.size()))
    memset_zero(storage, comptime(layout.size()))
    return LayoutTensor[ftype, layout, MutAnyOrigin](storage)


@always_inline("nodebug")
fn cleanFunctionName[func: fn () -> None]() -> String:
    """Simple reflection."""
    var func_name = get_linkage_name[func]().split("(")[0]
    func_name = func_name.split("[")[0].split("::")[1]
    return func_name


fn showProgress(progress: Int, total: Int):
    """Simple progress bar. You'll want to print a newline when complete."""
    comptime bar_width = 50

    if progress == (total - 1):
        print("\r[", end="")
        for _ in range(bar_width):
            print("=", end="")
        print("] TASK COMPLETE!")
    else:
        var ratio = progress / total
        var filled = Int(bar_width * ratio)
        print("\r[", end="")
        for _ in range(filled):
            print("=", end="")
        for _ in range(filled, bar_width):
            print(" ", end="")
        print("]", round(ratio * 100, 1), "%", end="")


fn systemInfo[ftype: DType]():
    """Helps display SIMD capabilities on your machine."""
    comptime nelts = simd_width_of[ftype]()
    print(
        "All tests are with dtype "
        + String(ftype)
        + " and comptime known lengths."
        + "\nYour machine gives:"
        + "\n\tNum logical cores:\t"
        + String(num_logical_cores())
        + "\n\tSIMD byte width:\t"
        + String(simd_byte_width())
        + "\n\tSIMD"
        + String(ftype)
        + "width:\t"
        + String(nelts)
        + "\n"
    )


fn compareBuffers(
        a: UnsafePointer[Scalar[ftype]], b: UnsafePointer[sftype], length: Int, epsilon: sftype = 1e-6
) -> Bool:
    """Ensures two buffers are equal within some error bounds (floating point isn't perfect).
    """
    #comptime epsilon = 1e-6
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


fn coloredString(
    str: String, color: String = ColorsEnum.COLOR_YELLOW
) -> String:
    return String(color + str + ColorsEnum.COLOR_RESET)
