from sys.info import simd_byte_width, num_logical_cores, simd_width_of
from reflection import get_linkage_name
from layout import Layout, LayoutTensor
from random import randn#, seed
from memory import memset, memset_zero

from attention import ftype, sftype, nelts

comptime std_std_deviation = 0.5
fn fillTensorRand[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin], std: Float64 = std_std_deviation):
    randn(x.ptr, layout.size(), 0, std) # mean of zero

fn randTensorHeap[layout: Layout](std: Float64 = std_std_deviation) -> LayoutTensor[ftype, layout, MutAnyOrigin]:
    var storage = alloc[sftype](layout.size())
    randn(storage, layout.size(), 0, 1)
    return LayoutTensor[ftype, layout, MutAnyOrigin](storage)

fn zeroTensorHeap[layout: Layout]() -> LayoutTensor[ftype, layout, MutAnyOrigin]:
    var storage = alloc[sftype](layout.size())
    memset_zero(storage, layout.size())
    return LayoutTensor[ftype, layout, MutAnyOrigin](storage)

fn cleanFunctionName[func: fn() -> None]() -> String:
    var func_name = get_linkage_name[func]().split('(')[0]
    func_name = func_name.split('[')[0].split('::')[1]
    return func_name

fn showProgress(progress: Int, total: Int):
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
    comptime nelts = simd_width_of[ftype]()
    print("All tests are with dtype " + String(ftype) + " and comptime known lengths." +
          "\nYour machine gives:" +
          "\n\tNum logical cores:\t" + String(num_logical_cores()) + 
          "\n\tSIMD byte width:\t" + String(simd_byte_width()) +
          "\n\tSIMD" + String(ftype) + "width:\t" + String(nelts) + '\n')
    
fn compareBuffers(a: UnsafePointer[Scalar[ftype]], b: UnsafePointer[sftype], length: Int) -> Bool:
    comptime epsilon = 1e-6
    for i in range(length):
        if (a[i] < (b[i] - epsilon)) or (a[i] > (b[i] + epsilon)):
            print("element i", i, ":", a[i], "!=", b[i])
            return False
    return True

