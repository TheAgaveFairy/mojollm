from random import randn
from math import sqrt, exp#, log, ceildiv

fn main():
    var ptr = alloc[Scalar[DType.float32]](18)
    randn(ptr, 18, 0, 1)
    ptr[0] = exp(ptr[1])
    ptr[1] = sqrt(ptr[2])
