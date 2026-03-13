from layout import Layout, LayoutTensor
from std.math import sqrt, exp, log, ceildiv
from std.random import random_float64, random_si64, randint, randn, rand, seed
from std.sys.info import (
    simd_bit_width,
    simd_byte_width,
    simd_width_of,
    num_logical_cores,
)  # sizeof moved
from std.sys import stderr, is_big_endian
from std.utils.index import IndexList
import std.os
from std.memory import memcpy, memset, memset_zero
from std.time import perf_counter_ns
from std.algorithm.functional import vectorize, parallelize
from linalg.matmul import matmul

# from kernels.nn.softmax import softmax
from std.testing import TestSuite, assert_equal, assert_true
from std.benchmark import compiler

from attention import ftype, sftype, token_itype, nelts, _myTensorCopyFrom, _trans
from activation_fn import ActivationFunction, ReLU, GELU, GELUTanh, GELUFast
from helpers import compareBuffers, fillTensorRand


fn weightAndBias[
    layout_input: Layout,
    layout_weights: Layout,
    layout_bias: Layout,
    layout_output: Layout,
](
    input: LayoutTensor[ftype, layout_input, _],
    weights: LayoutTensor[ftype, layout_weights, _],
    biases: LayoutTensor[ftype, layout_bias, _],
    output: LayoutTensor[ftype, layout_output, MutAnyOrigin],
) -> None:
    """
    Does a dot product and adds a bias to output which is modified in-place.
    """

    #dotProductTiledVectorizedParallelized(
            #input, weights, output
        #)  # can swap for any dotProduct
    # naiveDotProduct(input, weights, output)
    try:
        # mine is within an order of magnitude, but might as well use a known BLAS
        matmul(output, input, weights, None)
    except e:
        print(e, file=stderr)
        naiveDotProduct(input, weights, output)

    # @parameter # explodes compile time
    for i in range(output.shape[0]()):
        # @parameter
        for j in range(output.shape[1]()):
            output[i, j] += biases[j]


fn dotProductTiledVectorizedParallelized[
    layout_a: Layout, layout_b: Layout, layout_c: Layout, tile_size: Int = 32
](
    a: LayoutTensor[ftype, layout_a, _],
    b: LayoutTensor[ftype, layout_b, _],
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
    comptime are_powers_of_two = isPowerOfTwo(M) and isPowerOfTwo(
        L
    ) and isPowerOfTwo(N) and isPowerOfTwo(tile_size)
    comptime assert tile_small_enough,
        "Tile size too big, dotProductTiledVectorizedParallelized"
    comptime assert are_powers_of_two, "Shapes of tiled matmul not powers of two."

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

            var bT_tile = LayoutTensor[
                ftype, Layout.row_major(tile_size, tile_size), MutAnyOrigin
            ].stack_allocation()
            bT_tile.copy_from(b_tile)
            # TODO:: bT_tile needs to correctly MOVE data

            for ti in range(tile_size):
                for tj in range(tile_size):
                    # var ti = tii
                    # var tj = tjj
                    @parameter
                    fn dot_product[
                        width: Int
                    ](tk: Int) unified {
                        read ti, read tj, mut a_tile, mut bT_tile, mut c_tile
                    }:
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
    a: LayoutTensor[ftype, layout_a, _],
    b: LayoutTensor[ftype, layout_b, _],
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
    a: LayoutTensor[ftype, layout_a, _],
    b: LayoutTensor[ftype, layout_b, _],
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
    # var b_temp = rebind[
    #    LayoutTensor[ftype, Layout.row_major(inner, cols), MutAnyOrigin]
    # ](b)
    # var bT = layoutTensorDataTranspose2D[inner, cols](b) # returns (cols, inner)
    var bT = LayoutTensor[
        ftype, Layout.row_major(cols, inner), MutAnyOrigin
    ].stack_allocation()
    _myTensorCopyFrom(src=b, dest=bT, transposed=True)

    for i in range(rows):
        for j in range(cols):
            # var temp = SIMD[ftype, width](0.0)
            var temp: sftype = 0.0

            @parameter
            fn dot_product[
                width: Int
            ](k: Int) unified {mut a, mut bT, mut temp, read i, read j}:
                var v1 = a.load[width](i, k)
                var v2 = bT.load[width](j, k)
                var v3 = v1 * v2

                comptime for k in range(width):
                    temp[k] += v3[k]
                # temp += v3.reduce_add()  # less accurate but whatever

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
    x: LayoutTensor[ftype, layout_input, _],
    gamma: LayoutTensor[ftype, layout_gamma, _],
    beta: LayoutTensor[ftype, layout_beta, _],
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
        var mean = sum / sftype(d_model)

        var variance: sftype = 0.0
        # @parameter
        for i in range(d_model):
            diff = rebind[sftype](x[sl, i] - mean)
            variance += diff * diff
        variance = variance / sftype(d_model)

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
    act_fn: ActivationFunction = GELUFast,
](
    x: LayoutTensor[ftype, layout_x, _],
    w0: LayoutTensor[ftype, layout_w0, _],
    b0: LayoutTensor[ftype, layout_b0, _],
    w1: LayoutTensor[ftype, layout_w1, _],
    b1: LayoutTensor[ftype, layout_b1, _],
    hidden: LayoutTensor[ftype, layout_hidden, MutAnyOrigin],  # internal buffer
    output: LayoutTensor[ftype, layout_x, MutAnyOrigin],
):
    """
    Linear layers where the middle / hidden
    buffer is of a higher dimension, d_ff.
    """
    weightAndBias(x, w0, b0, hidden)
    act_fn.forward(hidden)
    weightAndBias(hidden, w1, b1, output)

@deprecated("Don't fully trust this, needs testing.")
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
    Q: LayoutTensor[ftype, layout0, _],
    K: LayoutTensor[ftype, layout0, _],
    V: LayoutTensor[ftype, layout1, _],
    scores: LayoutTensor[ftype, layout2, MutAnyOrigin],
    scores_probs: LayoutTensor[ftype, layout2, MutAnyOrigin],
    output: LayoutTensor[ftype, layout3, MutAnyOrigin],
) -> None:
    """
    Modifies output in-place. Biases not yet supported.
    """
    comptime d_k = Q.shape[1]()

    # var KT = layoutTensorDataTranspose2D[K.shape[0](), K.shape[1]()](K)
    var KT = LayoutTensor[ftype, _trans[K.layout](), MutAnyOrigin].stack_allocation()
    # KT.copy_from(K) # compiler "bug", kills compilation due to unrolling
    _myTensorCopyFrom(src=K, dest=KT, transposed=True)

    try:
        matmul[transpose_b = True](scores, Q, K, None)
    except e:
        print(e, file=stderr)
        dotProductTiledVectorizedParallelized(Q, KT, scores)
        # naiveDotProduct(Q, KT, scores)

    # scores = scores / sqrt(d_k) # TODO: big sizes break compilation bug
    for i in range(scores.shape[0]()):
        for j in range(scores.shape[1]()):
            scores[i, j] /= sqrt(sftype(d_k))

    applyCausalMask(scores)

    # scores_probs.copy_from(scores)
    _myTensorCopyFrom(src=scores, dest=scores_probs)

    # modifies scores_probs in-place
    naiveSoftmax(scores_probs)

    # modifies output in-place
    try:
        matmul(output, scores_probs, V, None)
    except e:
        print(e, file=stderr)
        dotProductTiledVectorizedParallelized(scores_probs, V, output)
    # naiveDotProduct(scores_probs, V, output)

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
    layout_scores: Layout, seq_len: Int
](
    scores: LayoutTensor[ftype, layout_scores, MutAnyOrigin],
    padding_mask: InlineArray[Bool, seq_len],
) -> None:
    """Uses the padding mask for variable length sequences, and also applies causal masking.
    """
    # comptime seq_len = scores.shape[0]() # == ModelParams.seq_len
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
    logits: LayoutTensor[ftype, layout_logits, _],
    targets: LayoutTensor[token_itype, layout_targets, _],
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

    return total_loss / sftype(seq_len)


fn crossEntropyLossMasked[
    layout_logits: Layout,
    layout_targets: Layout,
    layout_mask: Layout,
    seq_len: Int,
](
    logits: LayoutTensor[ftype, layout_logits, _],
    targets: LayoutTensor[token_itype, layout_targets, _],
    padding_mask: InlineArray[Bool, seq_len],
) -> sftype:
    """Computes cross entropy loss."""
    # comptime seq_len = logits.shape[0]()
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

    return total_loss / sftype(num_real_tokens)


# BACKWARDS TIME!


fn accumulateGrads[
    layout: Layout
](
    *,
    src: LayoutTensor[ftype, layout, _],
    dest: LayoutTensor[ftype, layout, MutAnyOrigin],
):
    """Element-wise addition with SIMD."""
    comptime size = layout.size()

    fn closure[width: Int](i: Int) unified {mut}:
        var v0 = src.ptr.load[width](i)
        var v1 = dest.ptr.load[width](i)
        var res = v0 + v1
        dest.ptr.store[width](i, res)

    vectorize[nelts](size, closure)


fn crossEntropyLossBackward[
    layout: Layout, seq_len: Int
](
    logits: LayoutTensor[ftype, layout, _],
    target_tokens: InlineArray[Int, seq_len],
    d_logits: LayoutTensor[ftype, layout, MutAnyOrigin],
):
    """Computes gradient of cross-entropy loss with respect to logits.
    dL / d_logits[i, j] = softmax(logits)[i,j] - one_hot[i,j]
    Modifies d_logits in-place."""
    comptime assert seq_len == logits.shape[0](), "Invalid seq_len for cross-entropy backwards."
    comptime vocab_size = logits.shape[1]()

    var logits_temp = type_of(logits).stack_allocation().fill(0.0)
    _myTensorCopyFrom(src=logits, dest=logits_temp)
    naiveSoftmax(logits_temp)
    _myTensorCopyFrom(src=logits_temp, dest=d_logits)

    # TODO: vectorize with SIMD
    for i in range(seq_len):
        var target_idx = target_tokens[i]
        d_logits[i, target_idx] -= sftype(1.0)  # one-hot

    comptime scale = sftype(1.0) / sftype(seq_len)

    # parameter unrolling "bug", do it manually
    fn closure[width: Int](i: Int) unified {mut}:
        var v = d_logits.ptr.load[width](i)
        v *= scale
        d_logits.ptr.store[width](i, v)

    vectorize[nelts](comptime (layout.size()), closure)


fn weightAndBiasBackward[
    layout_input: Layout,
    layout_weights: Layout,
    layout_bias: Layout,
    layout_d_output: Layout,
](
    input: LayoutTensor[ftype, layout_input, _],
    weights: LayoutTensor[ftype, layout_weights, _],
    d_output: LayoutTensor[ftype, layout_d_output, MutAnyOrigin],
    d_input: LayoutTensor[ftype, layout_input, MutAnyOrigin],
    d_weights: LayoutTensor[ftype, layout_weights, MutAnyOrigin],
    d_bias: LayoutTensor[ftype, layout_bias, MutAnyOrigin],
):
    """
    Backwards pass for Y = X @ W + b.

    Input shapes: (thank you Claude 4.5 for this)
        X: (seq_len, d_in)
        W: (d_in, d_out)
        b: (d_out,)
        dY: (seq_len, d_out)

    Computes:
        dX = dY @ W^T  → (seq_len, d_in)
        dW = X^T @ dY  → (d_in, d_out)
        db = sum(dY, 0) → (d_out,)

    Modifies d_* buffers in-place.
    """
    comptime seq_len = input.shape[0]()
    comptime d_in = input.shape[1]()
    comptime d_out = weights.shape[1]()

    # dX = dY @ W^T
    var W_T = LayoutTensor[
        ftype, Layout.row_major(d_out, d_in), MutAnyOrigin
    ].stack_allocation()
    _myTensorCopyFrom(src=weights, dest=W_T, transposed=True)
    dotProductTiledVectorizedParallelized(d_output, W_T, d_input)

    # dW = X^T @ dY
    var X_T = LayoutTensor[
        ftype, Layout.row_major(d_in, seq_len), MutAnyOrigin
    ].stack_allocation()
    _myTensorCopyFrom(src=input, dest=X_T, transposed=True)
    dotProductTiledVectorizedParallelized(X_T, d_output, d_weights)

    # db = sum(dY, axis = 0)
    for i in range(seq_len):
        for j in range(d_out):
            d_bias[j] += d_output[i, j]

    # TODO: could vectorize with a transpose of d_output...
    # ... this is WRONG as is!
    # it won't load the correct column vecs (row-major storage)
    _ = """
    for k in range(seq_len):
        var d_output_slice = d_output.slice_1d[
            Slice(0, seq_len), IndexList[1](0)
        ](IndexList[1](k))

        fn closure[width: Int](i: Int) unified {mut}:
            var v0 = d_bias.ptr.load[width](i)
            var v1 = d_output_slice.ptr.load[width](i)
            var res = v0 + v1
            d_output_slice.ptr.store[width](i, res)

        vectorize[nelts](d_out, closure)
    """

fn layerNormBackward[layout_input: Layout, layout_output: Layout, layout_gamma: Layout](
        x: LayoutTensor[ftype, layout_input, _],
        gamma: LayoutTensor[ftype, layout_gamma, _],
        d_output: LayoutTensor[ftype, layout_output, _],
        d_input: LayoutTensor[ftype, layout_input, MutAnyOrigin],
        d_gamma: LayoutTensor[ftype, layout_gamma, MutAnyOrigin],
        d_beta: LayoutTensor[ftype, layout_gamma, MutAnyOrigin]):
    """
    Backward pass for Layer Normalization.

    Forward was: y[i] = gamma * (x[i] - mean) / sqrt(var + eps) + beta

    Must calculate full Jacobian per sample. Helped by Claude 4.5.
    """
    comptime seq_len = x.shape[0]()
    comptime d_model = x.shape[1]()
    comptime epsilon = sftype(1e-5)

    var N = sftype(d_model)
    for sl in range(seq_len):
        # recalculate mean and std
        var mean: sftype = 0.0
        for i in range(d_model):
            mean += rebind[sftype](x[sl , i])
        mean /= sftype(d_model)

        var variance: sftype = 0.0
        for i in range(d_model):
            var diff = x[sl, i] - mean
            variance += rebind[sftype](diff * diff)
        variance /= sftype(d_model)

        var std = sqrt(variance + epsilon)
        var inv_std = 1.0 / std

        # now we can continue
        for i in range(d_model):
            var x_norm = (x[sl, i] - mean) / std
            d_gamma[i] += d_output[sl, i] * x_norm
            d_beta[i] += d_output[sl, i]

        # Jacobian, straight from Claude
        var sum_dL_dy_times_gamma: sftype = 0.0
        var sum_dL_dy_times_xm_times_gamma: sftype = 0.0
        
        for i in range(d_model):
            var x_centered = rebind[sftype](x[sl, i]) - mean
            var dout_times_gamma = d_output[sl, i] * rebind[sftype](gamma[i])
            sum_dL_dy_times_gamma += rebind[sftype](dout_times_gamma)
            sum_dL_dy_times_xm_times_gamma += rebind[sftype](dout_times_gamma * x_centered)
        
        for i in range(d_model):
            var x_centered = rebind[sftype](x[sl, i]) - mean
            var term1 = d_output[sl, i] * rebind[sftype](gamma[i]) * inv_std
            var term2 = sum_dL_dy_times_gamma / N * inv_std
            var term3 = (x_centered * sum_dL_dy_times_xm_times_gamma) / (N * (variance + epsilon))
            
            d_input[sl, i] = term1 - term2 - term3
        
fn feedForwardBackward[
        layout_x: Layout,
        layout_w0: Layout,
        layout_b0: Layout,
        layout_w1: Layout,
        layout_b1: Layout,
        layout_hidden: Layout,
    act_fn: ActivationFunction = GELUFast,
](
    x: LayoutTensor[ftype, layout_x, _],
    w0: LayoutTensor[ftype, layout_w0, _],
    b0: LayoutTensor[ftype, layout_b0, _],
    w1: LayoutTensor[ftype, layout_w1, _],
    b1: LayoutTensor[ftype, layout_b1, _],
    hidden: LayoutTensor[ftype, layout_hidden, _],
    d_output: LayoutTensor[ftype, layout_x, _],
    d_x: LayoutTensor[ftype, layout_x, MutAnyOrigin],
    d_w0: LayoutTensor[ftype, layout_w0, MutAnyOrigin],
    d_b0: LayoutTensor[ftype, layout_b0, MutAnyOrigin],
    d_w1: LayoutTensor[ftype, layout_w1, MutAnyOrigin],
    d_b1: LayoutTensor[ftype, layout_b1, MutAnyOrigin],
):
    """
    Backward pass for the feed-forward network.
    Reminder, FFN Forward is:
    1) weightAndBias(x, w0, b0, hidden): hidden = x @ w0 + b0
    2) act_fn.forward(hidden)
    3) weightAndBias(hidden, w1, b1, output): output = hidden @ w1 + b1
    Modifies buffers in-place.
    """
    comptime d_ff = hidden.shape[1]()
    var d_hidden = LayoutTensor[ftype, layout_hidden, MutAnyOrigin].stack_allocation()
    # TODO: uhhhh wand signature

    #weightAndBiasBackward(hidden, w1, b1, d_output, d_hidden, d_w1, d_b1)
    var d_hidden_pre_act = type_of(d_hidden).stack_allocation()
    act_fn.backward(hidden, d_hidden, d_hidden_pre_act)
    #weightAndBiasBackward(x, w0, b0, d_hidden_pre_act, d_x, d_w0, d_b0)

fn naiveAttentionBackward[
        layout_q: Layout,
        layout_k: Layout, # same as layout_q
        layout_v: Layout,
        layout_scores: Layout,
        layout_attn: Layout,
        layout_output: Layout,
](
        Q: LayoutTensor[ftype, layout_q, _],
        K: LayoutTensor[ftype, layout_k, _],
        V: LayoutTensor[ftype, layout_v, _],
        attn_scores: LayoutTensor[ftype, layout_scores, _],
        attn_probs: LayoutTensor[ftype, layout_attn, _],
        d_output: LayoutTensor[ftype, layout_output, _],
        d_Q: LayoutTensor[ftype, layout_q, MutAnyOrigin],
        d_K: LayoutTensor[ftype, layout_k, MutAnyOrigin],
        d_V: LayoutTensor[ftype, layout_v, MutAnyOrigin],
):
    """
    Backward pass for naive, scaled dot-product attention.
    Reminder, the forward was:
    scores = Q @ K^T / sqrt(d_k)
    attn_probs = softmax(scores)
    output = attn_probs @ V 
    Modifies d_Q, d_K, d_V in-place.
    """
    comptime seq_len = Q.shape[0]()
    comptime d_k = Q.shape[1]()

    comptime scale = 1.0 / sqrt(sftype(d_k))

    # d(attn_probs) = d_output @ V^T
    var V_T = LayoutTensor[ftype, _trans[V.layout](), MutAnyOrigin].stack_allocation()
    #var V_T = type_of(V.transpose()).stack_allocation()
    _myTensorCopyFrom(src=V, dest=V_T, transposed=True)

    var d_attn_probs = LayoutTensor[ftype, layout_attn, MutAnyOrigin].stack_allocation()
    dotProductTiledVectorizedParallelized(d_output, V_T, d_attn_probs)

    # Softmax backward
    var d_scores = LayoutTensor[ftype, layout_scores, MutAnyOrigin].stack_allocation().fill(0.0)
    for i in range(seq_len):
        pass
# TESTS


def main():
    myBestBenchmarkTest()
    var suite = TestSuite()
    suite.test[matmulCorrectnessTest]()
    suite.test[powersOfTwoTest]()
    suite^.run()

def myBestBenchmarkTest():
    """Test matrix multiplication against known values."""
    comptime M = 1 << 7
    comptime L = 1 << 5
    comptime N = 1 << 6
    comptime layout_a = Layout.row_major(M, L)
    comptime layout_b = Layout.row_major(L, N)
    comptime layout_c = Layout.row_major(M, N)

    var a = LayoutTensor[ftype, layout_a, MutAnyOrigin].stack_allocation()
    var b = LayoutTensor[ftype, layout_b, MutAnyOrigin].stack_allocation()
    var c = (
        LayoutTensor[ftype, layout_c, MutAnyOrigin].stack_allocation().fill(0)
    )
    var expected = (
        LayoutTensor[ftype, layout_c, MutAnyOrigin].stack_allocation().fill(0)
    )

    fillTensorRand(a)
    fillTensorRand(b)

    comptime warmups = 1000
    comptime times = 10000
    for _ in range(warmups):
        matmul(expected, a, b, None)
    var start = perf_counter_ns()
    for _ in range(times):
        # known correct
        matmul(expected, a, b, None)
        #naiveDotProduct(a, b, expected)
    var mid_best = perf_counter_ns()

    for _ in range(warmups):
        dotProductTiledVectorizedParallelized[tile_size=16](
            a, b, c
        )  # default tile_size = 32
    var mid_mine = perf_counter_ns()
    for _ in range(times):
        # pick the other to try
        dotProductTiledVectorizedParallelized[tile_size=16](
            a, b, c
        )  # default tile_size = 32
    var end = perf_counter_ns()

    compiler.keep(expected)
    compiler.keep(c)

    var best_ns = mid_best - start
    var mine_ns = end - mid_mine
    var ratio = Float64(mine_ns) / Float64(best_ns)
    print("best: {}\nmine: {}\n\t=> mine is {}x slower".format(best_ns, mine_ns, ratio))

def attention():
    assert_true(True) # compare to nn.attention

def powersOfTwoTest():
    assert_true(not isPowerOfTwo(0), "0 is not a power of two")
    assert_true(isPowerOfTwo(1), "1 is a power of two")
    assert_true(not isPowerOfTwo(3), "3 is not a power of two")
    assert_true(isPowerOfTwo(4), "4 is a power of two")

    assert_true(
        not isPowerOfTwo(-25),
        "-25 is not a power of two AND we reject negative numbers",
    )


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
    var c = (
        LayoutTensor[ftype, layout_c, MutAnyOrigin].stack_allocation().fill(0)
    )
    var expected = (
        LayoutTensor[ftype, layout_c, MutAnyOrigin].stack_allocation().fill(0)
    )

    fillTensorRand(a)
    fillTensorRand(b)

    # known correct
    naiveDotProduct(a, b, expected)
    #matmul(c, a, b)

    # pick the other to try
    dotProductTiledVectorizedParallelized[tile_size=16](
        a, b, c
    )  # default tile_size = 32
    # dotProductVectorized(a, b, c) # default tile_size = 32

    comptime epsilon = 1e-4  # absolute tolerance, relative would be better
    assert_true(
        compareBuffers(
            c.ptr, expected.ptr, comptime (layout_c.size()), epsilon
        ),
        "matrix multiplication correctness, epsilon: " + String(epsilon),
    )


fn isPowerOfTwo(m: Int) -> Bool:
    """Brian Kernighan's algorithm for set-bit-counting."""
    if m < 0:
        return False
    var count = 0
    var n = m
    while n:
        n &= n - 1
        count += 1

    return count == 1
