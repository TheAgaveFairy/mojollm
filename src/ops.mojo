from layout import Layout, LayoutTensor
from math import sqrt, exp, log, ceildiv
from random import random_float64, random_si64, randint, randn, rand, seed
from sys.info import (
    simd_bit_width,
    simd_byte_width,
    simd_width_of,
    num_logical_cores,
)  # sizeof moved
from sys import stderr, is_big_endian
from utils.index import IndexList
import os
from memory import memcpy, memset, memset_zero
from time import perf_counter_ns
from algorithm.functional import vectorize, parallelize
#from kernels.nn.softmax import softmax
from testing import TestSuite, assert_equal, assert_true

from attention import ftype, sftype, token_itype, nelts, _myTensorCopyFrom
from activation_fn import ActivationFunction, ReLU
from helpers import compareBuffers, fillTensorRand


fn weightAndBias[
    layout_input: Layout,
    layout_weights: Layout,
    layout_bias: Layout,
    layout_output: Layout,
](
    input: LayoutTensor[ftype, layout_input],
    weights: LayoutTensor[ftype, layout_weights],
    biases: LayoutTensor[ftype, layout_bias],
    output: LayoutTensor[ftype, layout_output, MutAnyOrigin],
) -> None:
    """
    Does a dot product and adds a bias to output which is modified in-place.
    """

    dotProductTiledVectorizedParallelized(input, weights, output) # can swap for any dotProduct
    #naiveDotProduct(input, weights, output)

    # @parameter # explodes compile time
    for i in range(output.shape[0]()):
        # @parameter
        for j in range(output.shape[1]()):
            output[i, j] += biases[j]


fn dotProductTiledVectorizedParallelized[
    layout_a: Layout, layout_b: Layout, layout_c: Layout, tile_size: Int = 32
](
    a: LayoutTensor[ftype, layout_a],
    b: LayoutTensor[ftype, layout_b],
    c: LayoutTensor[ftype, layout_c, MutAnyOrigin],
) -> None:
    """
    Only handles shapes with shape sizes as powers of 2.
    Modifies 'c' in place.
    """
    comptime M = a.shape[0]()
    comptime L = a.shape[1]()
    comptime N = b.shape[1]()

    comptime tile_small_enough = tile_size <= M and tile_size <= L and tile_size <= N
    comptime are_powers_of_two = isPowerOfTwo(M) and isPowerOfTwo(L) and isPowerOfTwo(N) and isPowerOfTwo(tile_size)
    constrained[tile_small_enough, "Tile size too big for matrices of sizes {} {} {}.".format(M, L, N)]()
    constrained[are_powers_of_two, "Shapes of tiled matmul not powers of two."]()

    comptime num_tiles_m = M // tile_size
    comptime num_tiles_n = N // tile_size
    comptime num_output_tiles = num_tiles_m * num_tiles_n

    # TODO : shape asserts

    @parameter
    fn parallel_closure(tid: Int):  # thread id, range(0, num_output_tiles)
        var ci = tid // num_tiles_n
        var cj = tid % num_tiles_n

        var c_tile = c.tile[tile_size, tile_size](ci, cj).fill(0.0)

        for k in range(L // tile_size):
            var a_tile = a.tile[tile_size, tile_size](ci, k)
            var b_tile = b.tile[tile_size, tile_size](k, cj).transpose()

            var bT_tile = LayoutTensor[ftype, Layout.row_major(tile_size, tile_size), MutAnyOrigin].stack_allocation()
            bT_tile.copy_from(b_tile)
            # TODO:: bT_tile needs to correctly MOVE data

            for ti in range(tile_size):
                for tj in range(tile_size):
                    #var ti = tii
                    #var tj = tjj
                    @parameter
                    fn dot_product[width: Int](tk: Int) unified {read ti, read tj, mut a_tile, mut bT_tile, mut c_tile}:
                        var v1 = a_tile.load[width](ti, tk)
                        var v2 = bT_tile.load[width](tj, tk)
                        var v3 = v1 * v2
                        var temp_sum = v3.reduce_add()
                        c_tile[ti, tj] += temp_sum

                    vectorize[nelts](tile_size, dot_product)

    parallelize[parallel_closure](num_output_tiles)
    # benchmark.compiler.keep(c)


@always_inline("nodebug")
fn naiveDotProduct[
    layout_a: Layout, layout_b: Layout, layout_c: Layout
](
    a: LayoutTensor[ftype, layout_a],
    b: LayoutTensor[ftype, layout_b],
    c: LayoutTensor[ftype, layout_c, MutAnyOrigin],
) -> None:
    """
    Very simple matrix multiplication. No SIMD, parallelization, etc..
    """
    comptime rows = a.shape[0]()
    comptime inner = a.shape[1]()  # == b.shape[0]()
    comptime cols = b.shape[1]()

    for i in range(rows):
        for j in range(cols):
            var temp: sftype = 0.0
            for k in range(inner):
                temp += rebind[sftype](a[i, k] * b[k, j])
            c[i, j] = temp


fn dotProductVectorized[
    layout_a: Layout, layout_b: Layout, layout_c: Layout
](
    a: LayoutTensor[ftype, layout_a],
    b: LayoutTensor[ftype, layout_b],
    c: LayoutTensor[ftype, layout_c, MutAnyOrigin],
) -> None:
    """
    Modifies 'c' in-place as output.
    Vectorized / faster than naive. Use tiled version when available.
    """

    comptime rows = a.shape[0]()  # seq_len
    comptime inner = a.shape[1]()  # d_model == b.shape[0]()
    comptime cols = b.shape[1]()  # d_k or d_v

    # var bT = LayoutTensor[ftype, layout_b.transpose(), MutAnyOrigin].stack_allocation() #b.transpose()
    # bT.copy_from(b)
    #var b_temp = rebind[
    #    LayoutTensor[ftype, Layout.row_major(inner, cols), MutAnyOrigin]
    #](b)
    #var bT = layoutTensorDataTranspose2D[inner, cols](b) # returns (cols, inner)
    var bT = LayoutTensor[ftype, Layout.row_major(cols, inner), MutAnyOrigin].stack_allocation()
    _myTensorCopyFrom(src=b, dest = bT, transposed = True)

    for i in range(rows):
        for j in range(cols):
            #var temp = SIMD[ftype, width](0.0)
            var temp: sftype = 0.0

            @parameter
            fn dot_product[width: Int](k: Int) unified {mut}:
                var v1 = a.load[width](i, k)
                var v2 = bT.load[width](j, k)
                var v3 = v1 * v2
                @parameter
                for k in range(width):
                    temp[k] += v3[k]
                #temp += v3.reduce_add()  # less accurate but whatever

            vectorize[nelts](inner, dot_product)
            c[i, j] = temp.reduce_add()


fn naiveSoftmax[
    layout: Layout
](x: LayoutTensor[ftype, layout, MutAnyOrigin], *, temp: sftype = 1.0) -> None:
    """
    In-place, stable softmax. Modifies in-place.
    Temperature value should generally be close to 1.0.
    """
    comptime rows = x.shape[0]()
    comptime cols = x.shape[1]()

    for i in range(rows):
        var max_val: sftype = rebind[sftype](x[i, 0])
        for j in range(1, cols):
            if x[i, j] > max_val:
                max_val = rebind[sftype](x[i, j])

        var sum_exp: sftype = 0
        for j in range(cols):
            var exp_val = exp(x[i, j] - max_val)
            x[i, j] = exp_val / temp
            sum_exp += rebind[sftype](exp_val)

        for j in range(cols):
            x[i, j] /= sum_exp / temp


fn layerNorm[
    layout_input: Layout,
    layout_gamma: Layout,
    layout_beta: Layout,
    layout_output: Layout,
](
    x: LayoutTensor[ftype, layout_input],
    gamma: LayoutTensor[ftype, layout_gamma],
    beta: LayoutTensor[ftype, layout_beta],
    output: LayoutTensor[ftype, layout_output, MutAnyOrigin],
) -> None:
    """
    For a tensor of shape (seq_len, d_model),
    this normalizes each d_model separately.
    Modifies output in-place.
    """
    comptime seq_len = x.shape[0]()
    comptime d_model = x.shape[1]()  # == gamma and beta's shape of [d_model,]

    # @parameter
    for sl in range(seq_len):
        var sum: sftype = 0.0
        # @parameter
        for i in range(d_model):
            sum += rebind[sftype](x[sl, i])
        var mean = sum / d_model

        var variance: sftype = 0.0
        # @parameter
        for i in range(d_model):
            diff = rebind[sftype](x[sl, i] - mean)
            variance += diff * diff
        variance = variance / d_model

        comptime epsilon = 1e-5  # for stability near 0
        var std_dev = sqrt(variance + epsilon)
        for i in range(d_model):
            var normed = (x[sl, i] - mean) / std_dev
            output[sl, i] = gamma[i] * normed + beta[i]


fn feedForward[
    layout_x: Layout,
    layout_w0: Layout,
    layout_b0: Layout,
    layout_w1: Layout,
    layout_b1: Layout,
    layout_hidden: Layout,
    act_fn: ActivationFunction = ReLU,
](
    x: LayoutTensor[ftype, layout_x],
    w0: LayoutTensor[ftype, layout_w0],
    b0: LayoutTensor[ftype, layout_b0],
    w1: LayoutTensor[ftype, layout_w1],
    b1: LayoutTensor[ftype, layout_b1],
    hidden: LayoutTensor[ftype, layout_hidden, MutAnyOrigin],  # internal buffer
    output: LayoutTensor[ftype, layout_x, MutAnyOrigin],
):
    """
    Linear layers where the middle / hidden
    buffer is of a higher dimension, d_ff.
    """
    # dotProductTiledVectorizedParallelized(x, w0, hidden)
    #naiveDotProduct(x, w0, hidden)
    weightAndBias(x, w0, b0, hidden)
    act_fn.forward(hidden)
    weightAndBias(hidden, w1, b1, output)


@always_inline("nodebug")
fn layoutTensorDataTranspose2D[
    rows: Int, cols: Int
](
    tensor: LayoutTensor[ftype, Layout.row_major(rows, cols), MutAnyOrigin]
) -> LayoutTensor[ftype, Layout.row_major(cols, rows), MutAnyOrigin]:
    # var output_buffer = alloc[sftype](rows * cols)
    # stack allocation okay because inlined (why do i feel uneasy)
    var output = LayoutTensor[
        ftype, Layout.row_major(cols, rows), MutAnyOrigin
    ].stack_allocation()
    for r in range(rows):
        for c in range(cols):
            output[c, r] = tensor[r, c]
    return output


fn naiveAttention[
    layout0: Layout,
    layout1: Layout,
    layout2: Layout,
    layout3: Layout,
](
    Q: LayoutTensor[ftype, layout0],
    K: LayoutTensor[ftype, layout0],
    V: LayoutTensor[ftype, layout1],
    scores: LayoutTensor[ftype, layout2, MutAnyOrigin],
    scores_probs: LayoutTensor[ftype, layout2, MutAnyOrigin],
    output: LayoutTensor[ftype, layout3, MutAnyOrigin],
) -> None:
    """
    Modifies output in-place. Biases not yet supported.
    """
    comptime d_k = Q.shape[1]()

    # var KT = layoutTensorDataTranspose2D[K.shape[0](), K.shape[1]()](K)
    var KT = (
        LayoutTensor[ftype, K.layout.transpose(), MutAnyOrigin]
        .stack_allocation()
        .fill(0.0)
    )
    # KT.copy_from(K) # compiler "bug", kills compilation due to unrolling
    _myTensorCopyFrom(src=K, dest=KT)

    dotProductTiledVectorizedParallelized(Q, KT, scores)
    #naiveDotProduct(Q, KT, scores)

    # scores = scores / sqrt(d_k) # TODO: big sizes break compilation bug
    for i in range(scores.shape[0]()):
        for j in range(scores.shape[1]()):
            scores[i, j] /= sqrt(d_k)

    applyCausalMask(scores)

    # scores_probs.copy_from(scores)
    _myTensorCopyFrom(src=scores, dest=scores_probs)

    # modifies scores_probs in-place
    naiveSoftmax(scores_probs)

    # modifies output in-place
    dotProductTiledVectorizedParallelized(scores_probs, V, output)
    #naiveDotProduct(scores_probs, V, output)

    # AT A HIGH LEVEL THIS PERFORMED:
    # var scores = Q @ K^T / sqrt(d_k)
    # var atten = softmax(scores)
    # return attn @ V


fn applyCausalMask[
    layout: Layout
](scores: LayoutTensor[ftype, layout, MutAnyOrigin]):
    """Modifies data in-place."""
    comptime seq_len = scores.shape[0]()
    # constrained[seq_len == ModelParams.seq_len, "Invalid scores shape for causal mask."]()
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            scores[i, j] = sftype(FloatLiteral.negative_infinity)

fn applyMasks[
        layout_scores: Layout, seq_len: Int](
                scores: LayoutTensor[ftype, layout_scores, MutAnyOrigin],
                padding_mask: InlineArray[Bool, seq_len]
                ) -> None:
    """Uses the padding mask for variable length sequences, and also applies causal masking."""
    #comptime seq_len = scores.shape[0]() # == ModelParams.seq_len
    comptime neg_inf = sftype(FloatLiteral.negative_infinity)

    for i in range(seq_len):
        if not padding_mask[i]:
            for j in range(seq_len):
                scores[i, j] = neg_inf
            continue
        for j in range(seq_len):
            if j > i:
                scores[i, j] = neg_inf
            elif not padding_mask[j]:
                scores[i, j] = neg_inf

fn crossEntropyLoss[
    layout_logits: Layout, layout_targets: Layout
](
    logits: LayoutTensor[ftype, layout_logits],
    targets: LayoutTensor[token_itype, layout_targets],
    # padding_mask: LayoutTensor[DType.bool, layout_flat],
) -> sftype:
    """Computes cross entropy loss."""
    comptime seq_len = logits.shape[0]()
    comptime vocab_size = logits.shape[1]()

    var total_loss: sftype = 0.0
    for i in range(seq_len):
        var max_logit = rebind[sftype](logits[i, 0])
        for j in range(1, vocab_size):
            if logits[i, j] > max_logit:
                max_logit = rebind[sftype](logits[i, j])

        var sum_exp: sftype = 0.0
        for j in range(vocab_size):
            sum_exp += exp(rebind[sftype](logits[i, j]) - max_logit)
        var log_sum_exp = log(sum_exp) + max_logit

        var target_idx = Int(targets[i])
        var target_logit = rebind[sftype](logits[i, target_idx])
        total_loss += -(target_logit - log_sum_exp)

    return total_loss / seq_len

fn crossEntropyLossMasked[
        layout_logits: Layout, layout_targets: Layout, layout_mask: Layout, seq_len: Int
](
    logits: LayoutTensor[ftype, layout_logits],
    targets: LayoutTensor[token_itype, layout_targets],
    padding_mask: InlineArray[Bool, seq_len]
) -> sftype:
    """Computes cross entropy loss."""
    #comptime seq_len = logits.shape[0]()
    comptime vocab_size = logits.shape[1]()

    var num_real_tokens = 0
    var total_loss: sftype = 0.0
    for i in range(seq_len):
        if not padding_mask[i]:
            continue

        var max_logit = rebind[sftype](logits[i, 0])
        for j in range(1, vocab_size):
            if logits[i, j] > max_logit:
                max_logit = rebind[sftype](logits[i, j])

        var sum_exp: sftype = 0.0
        for j in range(vocab_size):
            sum_exp += exp(rebind[sftype](logits[i, j]) - max_logit)
        var log_sum_exp = log(sum_exp) + max_logit

        var target_idx = Int(targets[i])
        var target_logit = rebind[sftype](logits[i, target_idx])
        total_loss += -(target_logit - log_sum_exp)
        num_real_tokens += 1

    return total_loss / num_real_tokens


def main():
    var suite = TestSuite()
    suite.test[matmulCorrectnessTest]()
    suite.test[powersOfTwoTest]()
    suite^.run()

def powersOfTwoTest():
    assert_true(not isPowerOfTwo(0), "0 is not a power of two")
    assert_true(isPowerOfTwo(1), "1 is a power of two")
    assert_true(not isPowerOfTwo(3), "3 is not a power of two")
    assert_true(isPowerOfTwo(4), "4 is a power of two")

    assert_true(not isPowerOfTwo(-25), "-25 is not a power of two AND we reject negative numbers")

fn matmulCorrectnessTest() raises:
    """Test matrix multiplication against known values."""
    comptime M = 1 << 7
    comptime L = 1 << 5
    comptime N = 1 << 6
    comptime layout_a = Layout.row_major(M, L)
    comptime layout_b = Layout.row_major(L, N)
    comptime layout_c = Layout.row_major(M, N)
    
    var a = LayoutTensor[ftype, layout_a, MutAnyOrigin].stack_allocation()
    var b = LayoutTensor[ftype, layout_b, MutAnyOrigin].stack_allocation()
    var c = LayoutTensor[ftype, layout_c, MutAnyOrigin].stack_allocation().fill(0)
    var expected = LayoutTensor[ftype, layout_c, MutAnyOrigin].stack_allocation().fill(0)
    
    fillTensorRand(a)
    fillTensorRand(b)

    # known correct
    naiveDotProduct(a, b, expected)

    # pick the other to try
    dotProductTiledVectorizedParallelized[tile_size = 16](a, b, c) # default tile_size = 32
    #dotProductVectorized(a, b, c) # default tile_size = 32
    
    comptime epsilon = 1e-4 # absolute tolerance, relative would be better
    assert_true(compareBuffers(c.ptr, expected.ptr, comptime(layout_c.size()), epsilon), "matrix multiplication correctness, epsilon: " + String(epsilon))

fn isPowerOfTwo(m: Int) -> Bool:
    """Brian Kernighan's algorithm for set-bit-counting."""
    if m < 0:
        return False
    var count = 0
    var n = m
    while n:
        n &= (n - 1)
        count += 1

    return count == 1
