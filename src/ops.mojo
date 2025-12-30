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
        layout.output: Layout](
                input: LayoutTensor[ftype, layout_input, MutAnyOrigin],
                weights: LayoutTensor[ftype, layout_weights, MutAnyOrigin],
                biases: LayoutTensor[ftype, layout_bias, MutAnyOrigin],
                output: LayoutTensor[ftype, layout_output, MutAnyOrigin]) -> None:

    dotProductTiledVectorizedParallelized(input, weights, output)
    for 

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
        
        var c_tile = c.tile[tile_size, tile_size](ci, cj).fill(bias)
        
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
            mut c: LayoutTensor[ftype, layout_c, MutAnyOrigin]) -> None:
    comptime rows = a.shape[0]()
    comptime inner = a.shape[1]() # == b.shape[0]()
    comptime cols = b.shape[1]()

    for i in range(rows):
        for j in range(cols):
            var temp: sftype = 0.0
            for k in range(inner):
                temp += rebind[sftype](a[b_num, i, k] * b[k, j])
            c[b_num, i, j] = temp

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
            if x[b, i, j] > max_val:
                max_val = rebind[sftype](x[i, j])

        var sum_exp: sftype = 0
        for j in range(cols):
            var exp_val = exp(x[i, j] - max_val)
            x[i, j] = exp_val
            sum_exp += rebind[sftype](exp_val)

        for j in range(cols):
            x[i, j] /= sum_exp

fn layerNorm[layout0: Layout,
             layout1: Layout](
                x: LayoutTensor[ftype, layout0, MutAnyOrigin],
                gamma: LayoutTensor[ftype, layout1],
                beta: LayoutTensor[ftype, layout1]) -> None:
    """
    For a tensor of shape (seq_len, d_model),
    this normalizes each d_model separately.
    Takes in a single input 'x' from the batch and has
    a shape of (seq_len, d_model).
    MODIFIES 'x' in-place.
    """
    comptime seq_len = x.shape[0]()
    comptime d_model = x.shape[1]()

    @parameter
    for sl in range(seq_len):
        var sum: sftype = 0.0
        @parameter
        for i in range(d_model):
            sum += rebind[sftype](x[sl, i])
        var mean = sum / d_model

        var variance: sftype = 0.0
        @parameter
        for i in range(d_model):
            diff = rebind[sftype](x[sl, i] - mean)
            variance += diff * diff
        variance = variance / d_model

        comptime epsilon = 1e-5 # for stability near 0
        var std_dev = sqrt(variance + epsilon)
        for i in range(d_model):
            var normed = (x[sl, i] - mean) / std_dev
            x[sl, i] = gamma[i] * normed + beta[i]

fn feedForward[layout0: Layout,
               layout1: Layout,
               layout2: Layout,
               act_fn: ActivationFunction = ReLU](
                x: LayoutTensor[ftype, layout0, MutAnyOrigin],
                w0: LayoutTensor[ftype, layout1, MutAnyOrigin],
                b0: LayoutTensor[ftype, layout1, MutAnyOrigin],
                w1: LayoutTensor[ftype, layout2, MutAnyOrigin],
                b1: LayoutTensor[ftype, layout2, MutAnyOrigin],
                hidden: LayoutTensor[ftype, layout3, MutAnyOrigin], # internal buffer
                mut output: LayoutTensor[ftype, layout0, MutAnyOrigin]):
    """
    No batching here - while we're just on CPU we'll handle that elsewhere.
    General batched form (Claude Sonnet 4.5):
    Input: x_normalized with shape (batch, seq_len, d_model)
        ↓
    Linear 1: matmul with W1 of shape (d_model, d_ff)  where d_ff = 4 * d_model
        → output shape: (batch, seq_len, d_ff)
        ↓
    Activation (GELU or ReLU) - applied element-wise
        → output shape: (batch, seq_len, d_ff)
        ↓
    Linear 2: matmul with W2 of shape (d_ff, d_model)
        → output shape: (batch, seq_len, d_model)
    Output is modified in-place.
    """
    #comptime seq_len = x.shape[0]()
    #comptime d_model = x.shape[1]()
    #comptime d_ff = w0.shape[1]() # == w1.shape[0]()
    #comptime hidden_layout = Layout.row_major(seq_len, d_ff)

    #var hidden_storage = alloc[sftype](hidden_layout.size()) # could use heap
    #var hidden = LayoutTensor[ftype, hidden_layout, MutAnyOrigin].stack_allocation()
    #dotProductSlices(x, w0, hidden) # c = a*b
    dotProductTiledVectorizedParallelized(x, w0, hidden)
    hidden += b0
    act_fn.forward(hidden)
    #dotProductSlices(hidden, w1, output)
    dotProductTiledVectorizedParallelized(hidden, w1, output)
    output += b1

@always_inline("nodebug")
fn layoutTensorDataTranspose2D[rows: Int, cols: Int](tensor: LayoutTensor[ftype, Layout.row_major(rows, cols), MutAnyOrigin]) -> LayoutTensor[ftype, Layout.row_major(cols, rows), MutAnyOrigin]:
    #comptime rows = tensor.shape[0]()
    #comptime cols = tensor.shape[1]()

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
                          scores: LayoutTensor[ftype, layout3, MutAnyOrigin],
                          mut output: LayoutTensor[ftype, layout2]) -> None:
    comptime seq_len = Q.shape[0]()
    comptime d_k = Q.shape[1]()
    comptime d_v = V.shape[2]()

    #comptime scores_layout = Layout.row_major(seq_len, seq_len)
    #var scores_buffer = alloc[sftype](scores_layout.size())
    #var scores = LayoutTensor[ftype, scores_layout, MutAnyOrigin].stack_allocation()#(scores_buffer)
    
    #dotProductSlices(Q, K, scores)
    dotProductTiledVectorizedParallelized(Q, K, scores)
    scores = scores / sqrt(d_k)
    naiveSoftmax(scores)

    #dotProductSlices(scores, V, output)
    dotProductTiledVectorizedParallelized(scores, V, output)
    # AT A HIGH LEVEL THIS PERFORMED:
    #var scores = Q @ KT / sqrt(d_k)
    #var atten = softmax(scores)
    #return attn @ V
