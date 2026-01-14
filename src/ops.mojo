from layout import Layout, LayoutTensor
from math import sqrt, exp, log, ceildiv
from random import random_float64, random_si64, randint, randn, rand, seed
from sys.info import simd_bit_width, simd_byte_width, simd_width_of, num_logical_cores # sizeof moved
from sys import stderr, is_big_endian
from utils.index import IndexList
import os
from memory import memcpy, memset, memset_zero
from time import perf_counter_ns
from algorithm.functional import vectorize, parallelize
from kernels.nn.softmax import softmax

from attention import ftype, sftype, nelts
from activation_fn import ActivationFunction, ReLU

fn weightAndBias[layout_input: Layout,
        layout_weights: Layout,
        layout_bias: Layout,
        layout_output: Layout](
                input: LayoutTensor[ftype, layout_input, MutAnyOrigin],
                weights: LayoutTensor[ftype, layout_weights, MutAnyOrigin],
                biases: LayoutTensor[ftype, layout_bias, MutAnyOrigin],
                output: LayoutTensor[ftype, layout_output, MutAnyOrigin]) -> None:

    #dotProductTiledVectorizedParallelized(input, weights, output) # can swap for any dotProduct
    naiveDotProduct(input, weights, output)

    #@parameter
    for i in range(output.shape[0]()):
        #@parameter
        for j in range(output.shape[1]()):
            output[i, j] += biases[j]

fn dotProductTiledVectorizedParallelized[layout_a: Layout,
                                   layout_b: Layout,
                                   layout_c: Layout,
                                   tile_size: Int = 32](
                                           a: LayoutTensor[ftype, layout_a, MutAnyOrigin],
                                           b: LayoutTensor[ftype, layout_b, MutAnyOrigin],
                                           c: LayoutTensor[ftype, layout_c, MutAnyOrigin]) -> None:
    """
    Only handles shapes with shape sizes as powers of 2.
    Modifies 'c' in place.
    """
    comptime M = a.shape[0]()
    comptime L = a.shape[1]()
    comptime N = b.shape[1]()

    comptime num_tiles_m = M // tile_size
    comptime num_tiles_n = N // tile_size
    comptime num_output_tiles = num_tiles_m * num_tiles_n

    # TODO : shape asserts

    @parameter
    fn parallel_closure(tid: Int): # thread id, range(0, num_output_tiles)
        var ci = tid // num_tiles_n
        var cj = tid %  num_tiles_n
        
        var c_tile = c.tile[tile_size, tile_size](ci, cj).fill(0.0)
        
        for k in range(L // tile_size):
            var a_tile = a.tile[tile_size, tile_size](ci, k)
            var bT_tile = b.tile[tile_size, tile_size](k, cj).transpose()

            for ti in range(tile_size):
                for tj in range(tile_size):
                    
                    @parameter
                    fn dot_product[width: Int](tk: Int) unified {mut}:
                        var v1 = a_tile.load[width](ti, tk)
                        var v2 = bT_tile.load[width](tj, tk)
                        var v3 = v1 * v2
                        var temp_sum = v3.reduce_add()
                        c_tile[ti, tj] += temp_sum
                    vectorize[nelts](tile_size, dot_product)
    parallelize[parallel_closure](num_output_tiles)
    #benchmark.compiler.keep(c)

fn naiveDotProduct[layout_a: Layout,
            layout_b: Layout,
            layout_c: Layout](
            a: LayoutTensor[ftype, layout_a],
            b: LayoutTensor[ftype, layout_b],
            c: LayoutTensor[ftype, layout_c, MutAnyOrigin]) -> None:
    comptime rows = a.shape[0]()
    comptime inner = a.shape[1]() # == b.shape[0]()
    comptime cols = b.shape[1]()

    for i in range(rows):
        for j in range(cols):
            var temp: sftype = 0.0
            for k in range(inner):
                temp += rebind[sftype](a[i, k] * b[k, j])
            c[i, j] = temp

fn dotProductSlices[layout_a: Layout,
              layout_b: Layout,
              layout_c: Layout](
                a: LayoutTensor[ftype, layout_a, MutAnyOrigin],
                b: LayoutTensor[ftype, layout_b, MutAnyOrigin],
                c: LayoutTensor[ftype, layout_c, MutAnyOrigin]) -> None:
    """
    Modifies 'c' in-place as output.
    Vectorized / faster than naive.
    Use tiled version when available.
    """

    comptime rows = a.shape[0]()   # seq_len
    comptime inner = a.shape[1]()  # d_model == b.shape[0]()
    comptime cols = b.shape[1]()   # d_k or d_v
    
    #var bT = LayoutTensor[ftype, layout_b.transpose(), MutAnyOrigin].stack_allocation() #b.transpose()
    #bT.copy_from(b)
    var b_temp = rebind[LayoutTensor[ftype, Layout.row_major(inner, cols), MutAnyOrigin]](b)
    var bT = layoutTensorDataTranspose2D[inner, cols](b_temp)

    for i in range(rows):
        for j in range(cols):
            #var temp = SIMD[ftype, nelts](0.0)
            var temp: sftype = 0.0
            @parameter
            fn dot_product[width: Int](k: Int) unified {mut}:
                var v1 = a.load[width](i, k)
                var v2 = bT.load[width](k, j)
                var v3 = v1 * v2
                temp += v3.reduce_add() # less accurate but whatever

            vectorize[nelts](inner, dot_product)
            c[i, j] = temp#.reduce_add()

fn naiveSoftmax[layout: Layout](mut x: LayoutTensor[ftype, layout, MutAnyOrigin]) -> None:
    """
    In-place, stable softmax.
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
            x[i, j] = exp_val
            sum_exp += rebind[sftype](exp_val)

        for j in range(cols):
            x[i, j] /= sum_exp

fn layerNorm[layout_input: Layout,
             layout_gamma: Layout,
             layout_beta: Layout,
             layout_output: Layout](
                x: LayoutTensor[ftype, layout_input, MutAnyOrigin],
                gamma: LayoutTensor[ftype, layout_gamma, MutAnyOrigin],
                beta: LayoutTensor[ftype, layout_beta, MutAnyOrigin],
                output: LayoutTensor[ftype, layout_output, MutAnyOrigin]) -> None:
    """
    For a tensor of shape (seq_len, d_model),
    this normalizes each d_model separately.
    """
    comptime seq_len = x.shape[0]()
    comptime d_model = x.shape[1]() # == gamma and beta's shape of [d_model,]

    #@parameter
    for sl in range(seq_len):
        var sum: sftype = 0.0
        #@parameter
        for i in range(d_model):
            sum += rebind[sftype](x[sl, i])
        var mean = sum / d_model

        var variance: sftype = 0.0
        #@parameter
        for i in range(d_model):
            diff = rebind[sftype](x[sl, i] - mean)
            variance += diff * diff
        variance = variance / d_model

        comptime epsilon = 1e-5 # for stability near 0
        var std_dev = sqrt(variance + epsilon)
        for i in range(d_model):
            var normed = (x[sl, i] - mean) / std_dev
            output[sl, i] = gamma[i] * normed + beta[i]

fn feedForward[layout_x: Layout,
               layout_w0: Layout,
               layout_b0: Layout,
               layout_w1: Layout,
               layout_b1: Layout,
               layout_hidden: Layout,
               act_fn: ActivationFunction = ReLU](
                x: LayoutTensor[ftype, layout_x, MutAnyOrigin],
                w0: LayoutTensor[ftype, layout_w0, MutAnyOrigin],
                b0: LayoutTensor[ftype, layout_b0, MutAnyOrigin],
                w1: LayoutTensor[ftype, layout_w1, MutAnyOrigin],
                b1: LayoutTensor[ftype, layout_b1, MutAnyOrigin],
                hidden: LayoutTensor[ftype, layout_hidden, MutAnyOrigin], # internal buffer
                mut output: LayoutTensor[ftype, layout_x, MutAnyOrigin]):
    """
    Linear layers where the middle / hidden
    buffer is of a higher dimension, d_ff.
    """
    #dotProductTiledVectorizedParallelized(x, w0, hidden)
    naiveDotProduct(x, w0, hidden)
    weightAndBias(x, w0, b0, hidden)
    act_fn.forward(hidden)
    weightAndBias(hidden, w1, b1, output)

@always_inline("nodebug")
fn layoutTensorDataTranspose2D[rows: Int, cols: Int](tensor: LayoutTensor[ftype, Layout.row_major(rows, cols), MutAnyOrigin]) -> LayoutTensor[ftype, Layout.row_major(cols, rows), MutAnyOrigin]:
    #var output_buffer = alloc[sftype](rows * cols)
    # stack allocation okay because inlined (why do i feel uneasy)
    var output = LayoutTensor[ftype, Layout.row_major(cols, rows), MutAnyOrigin].stack_allocation()
    for r in range(rows):
        for c in range(cols):
            output[c, r] = tensor[r, c]
    return output

fn naiveAttention[layout0: Layout,
                  layout1: Layout,
                  layout2: Layout,
                  layout3: Layout,](
                          Q: LayoutTensor[ftype, layout0, MutAnyOrigin],
                          K: LayoutTensor[ftype, layout0, MutAnyOrigin],
                          V: LayoutTensor[ftype, layout1, MutAnyOrigin],
                          mut scores: LayoutTensor[ftype, layout2, MutAnyOrigin],
                          mut scores_probs: LayoutTensor[ftype, layout2, MutAnyOrigin],
                          output: LayoutTensor[ftype, layout3, MutAnyOrigin]) -> None:
    comptime d_k = Q.shape[1]()
    # modifes scores in-place
    
    #var KT = layoutTensorDataTranspose2D[K.shape[0](), K.shape[1]()](K)
    var KT = LayoutTensor[ftype, K.layout.transpose(), MutAnyOrigin].stack_allocation().fill(0.0)
    _ = """
    for i in range(K.shape[0]()):
        for j in range(K.shape[1]()):
            KT[j, i] = K[i, j]
    """
    KT.copy_from(K)
    
    #dotProductTiledVectorizedParallelized(Q, KT, scores)
    naiveDotProduct(Q, KT, scores)
    scores = scores / sqrt(d_k)
    scores_probs.copy_from(scores)
    # modifies scores_probs in-place
    naiveSoftmax(scores_probs)

    # modifies output in-place
    #dotProductTiledVectorizedParallelized(scores_probs, V, output)
    naiveDotProduct(scores_probs, V, output)

    # AT A HIGH LEVEL THIS PERFORMED:
    #var scores = Q @ KT / sqrt(d_k)
    #var atten = softmax(scores)
    #return attn @ V
