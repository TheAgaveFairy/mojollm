from layout import LayoutTensor, Layout
from linalg.matmul import matmul
from std.benchmark.compiler import keep
from std.random import randn, random_float64

comptime ftype = DType.float64
comptime sftype = Scalar[ftype]
comptime N = 1 << 8
comptime Tensor = LayoutTensor[ftype, Layout.row_major(N, N), MutAnyOrigin]

def getTensor(x: sftype = 1.0) -> Tensor:
    return Tensor(alloc[sftype](N * N)).fill(x)

def main() raises:
    print("a")
    var a = getTensor(sftype(random_float64(-3, 8)))
    var b = getTensor()
    var c = getTensor(0.0)


    print("b")
    comptime times = 1 << 32
    for _ in range(times):
        matmul(c, a, b, None)
        keep(c)

