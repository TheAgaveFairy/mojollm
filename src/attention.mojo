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
from compile.reflection import get_linkage_name
from compile import compile_info
import benchmark # run, Unit.ms
#from kernels.nn.softmax import softmax

from helpers import showProgress, cleanFunctionName, systemInfo, randTensorHeap, zeroTensorHeap, compareBuffers
from activation_fn import ActivationFunction, ReLU
from ops import weightAndBias, naiveSoftmax, layerNorm, feedForward, naiveAttention

comptime ftype = DType.float32
comptime sftype = Scalar[ftype] # 's' prefix = 'S'calar
comptime nelts = simd_width_of[ftype]()

struct ModelParams():
    comptime vocab_size = 1 << 8

    comptime max_batch_size = 1 << 5 # hmm
    comptime seq_len = 1 << 3
    comptime d_model = 1 << 6

    comptime d_k = Self.d_model
    comptime d_v = Self.d_model

    comptime d_ff = Self.d_model << 2 # d_model * 4 is common, apparently

trait Weights:
    @staticmethod
    fn initRandom(out self: Self):
        ...
    # TODO: implement saving / loading
    # @staticmethod
    # fn initFromFile(out self):
    #     pass
    # fn saveToFile(self):
    #    pass

struct AttentionWeights(Weights, Copyable, Movable):
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    # learned weights
    comptime W_q_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_k)
    # comptime W_k_layout = W_q_layout
    comptime W_v_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_v)
    var W_q: LayoutTensor[ftype, Self.W_q_layout, MutAnyOrigin]
    var W_k: LayoutTensor[ftype, Self.W_q_layout, MutAnyOrigin]
    var W_v: LayoutTensor[ftype, Self.W_v_layout, MutAnyOrigin]

    comptime b_q_layout = Layout.row_major(ModelParams.d_k)
    comptime b_v_layout = Layout.row_major(ModelParams.d_v)
    var b_q: LayoutTensor[ftype, Self.b_q_layout, MutAnyOrigin]
    var b_k: LayoutTensor[ftype, Self.b_q_layout, MutAnyOrigin]
    var b_v: LayoutTensor[ftype, Self.b_v_layout, MutAnyOrigin]
    fn __init__(out self):
        print("AttentionWeights __init__() ERROR", file = stderr)

    @staticmethod
    fn initRandom(out self: Self):
        self.W_q = randTensorHeap[Self.W_q_layout]()
        self.W_k = randTensorHeap[Self.W_q_layout]()
        self.W_v = randTensorHeap[Self.W_v_layout]()
    
        self.b_q = randTensorHeap[Self.b_q_layout]()
        self.b_k = randTensorHeap[Self.b_q_layout]()
        self.b_v = randTensorHeap[Self.b_v_layout]()

    fn __del__(deinit self):
        self.W_q.ptr.free()
        self.W_k.ptr.free()
        self.W_v.ptr.free()

        self.b_q.ptr.free()
        self.b_k.ptr.free()
        self.b_v.ptr.free()
    
struct FFWeights(Weights, Copyable, Movable):
    comptime w0_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_ff)
    comptime w1_layout = Layout.row_major(ModelParams.d_ff, ModelParams.d_model)
    var w0: LayoutTensor[ftype, Self.w0_layout, MutAnyOrigin]
    var w1: LayoutTensor[ftype, Self.w1_layout, MutAnyOrigin]

    comptime b0_layout = Layout.row_major(ModelParams.d_ff)
    comptime b1_layout = Layout.row_major(ModelParams.d_model)
    var b0: LayoutTensor[ftype, Self.b0_layout, MutAnyOrigin]
    var b1: LayoutTensor[ftype, Self.b1_layout, MutAnyOrigin]

    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    fn __init__(out self):
        print("FFWeights __init__() ERROR", file = stderr)

    @staticmethod
    fn initRandom(out self: Self):
        self.w0 = randTensorHeap[Self.w0_layout]()
        self.w1 = randTensorHeap[Self.w1_layout]()

        self.b0 = randTensorHeap[Self.b0_layout]()
        self.b1 = randTensorHeap[Self.b1_layout]()

    fn __del__(deinit self):
        self.w0.ptr.free()
        self.w1.ptr.free()
        self.b0.ptr.free()
        self.b1.ptr.free()

struct LayerNormWeights(Weights, Copyable, Movable):
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True
    comptime gamma_layout = Layout.row_major(ModelParams.d_model)
    var gamma: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]
    var beta: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]
    fn __init__(out self):
        print("LayerNormWeights __init__() ERROR", file = stderr)

    @staticmethod
    fn initRandom(out self: Self):
        self.gamma = randTensorHeap[Self.gamma_layout]()
        self.beta = randTensorHeap[Self.gamma_layout]()

    fn __del__(deinit self):
        self.gamma.ptr.free()
        self.beta.ptr.free()

struct OutputWeights(Weights, Copyable, Movable):
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True
    comptime W_layout = Layout.row_major(ModelParams.d_model, ModelParams.vocab_size)
    comptime b_layout = Layout.row_major(ModelParams.vocab_size)
    var W: LayoutTensor[ftype, Self.W_layout, MutAnyOrigin]
    var b: LayoutTensor[ftype, Self.b_layout, MutAnyOrigin]

    fn __init__(out self):
        print("OutputWeights __init__() ERROR", file = stderr)
        pass

    @staticmethod
    fn initRandom(out self: Self):
        self.W = randTensorHeap[Self.W_layout]()
        self.b = randTensorHeap[Self.b_layout]()

    fn __del__(deinit self):
        self.W.ptr.free()
        self.b.ptr.free()

struct TransformerBlock(Copyable): # decoder, also would take Layer and/or Weights trait
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True
    var ln_attn: LayerNormWeights
    var attn_weights: AttentionWeights # single head, causal masking
    var ffn_weights: FFWeights
    var ln_ffn: LayerNormWeights

    comptime X_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    var X_pre_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]
    var X_post_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]

    # intermediate buffers
    comptime Q_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_k) # is K_Layout
    comptime V_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_v)
    var Q: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin] # intermediate buffers
    var K: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var V: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]

    comptime attn_scores_layout = Layout.row_major(ModelParams.seq_len, ModelParams.seq_len)
    var attn_scores: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var attn_probs: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]

    var attn_out_pre_residual: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]
    var attn_out_post_residual: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]
    
    comptime ffn_hidden_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_ff)
    comptime ffn_out_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    var ffn_input: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]
    var ffn_hidden: LayoutTensor[ftype, Self.ffn_hidden_layout, MutAnyOrigin]
    var ffn_out: LayoutTensor[ftype, Self.ffn_out_layout, MutAnyOrigin]

    @staticmethod
    fn initRandom(out self: Self):
        self.ln_attn = LayerNormWeights.initRandom()
        self.attn_weights = AttentionWeights.initRandom()
        self.ln_ffn = LayerNormWeights.initRandom()
        self.ffn_weights = FFWeights.initRandom()

        self.X_pre_ln_attn = zeroTensorHeap[Self.X_layout]()
        self.X_post_ln_attn = zeroTensorHeap[Self.X_layout]()
        
        self.Q = zeroTensorHeap[Self.Q_layout]()
        self.K = zeroTensorHeap[Self.Q_layout]()
        self.V = zeroTensorHeap[Self.V_layout]()
        
        # initialize attn_scores, attn_out, ffn_hidden, ffn_output to zeros ...
        self.attn_scores = zeroTensorHeap[Self.attn_scores_layout]()
        self.attn_probs = zeroTensorHeap[Self.attn_scores_layout]()
        self.attn_out_pre_residual = zeroTensorHeap[Self.V_layout]()
        self.attn_out_post_residual = zeroTensorHeap[Self.V_layout]()
        self.ffn_input = zeroTensorHeap[Self.V_layout]()
        self.ffn_hidden = zeroTensorHeap[Self.ffn_hidden_layout]()
        self.ffn_out = zeroTensorHeap[Self.ffn_out_layout]()

    fn forward[layout: Layout,
               out_layout: Layout](
                    mut self,
                    X: LayoutTensor[ftype, layout, MutAnyOrigin],
                    mut output: LayoutTensor[ftype, out_layout, MutAnyOrigin]):

        # layerNorm(input, gamma, beta, output)
        self.X_pre_ln_attn.copy_from(X)
        layerNorm(X,
                  self.ln_attn.gamma,
                  self.ln_attn.beta,
                  self.X_post_ln_attn)

        # weightAndBias(input, weight, bias, output)
        weightAndBias(self.X_post_ln_attn,
                      self.attn_weights.W_q,
                      self.attn_weights.b_q,
                      self.Q)
        weightAndBias(self.X_post_ln_attn,
                      self.attn_weights.W_k,
                      self.attn_weights.b_k,
                      self.K)
        weightAndBias(self.X_post_ln_attn,
                      self.attn_weights.W_v,
                      self.attn_weights.b_v,
                      self.V)

        # causal masking not implemented YET
        naiveAttention(self.Q, self.K, self.V,
                       self.attn_scores,    # intermediate buffer # backprop
                       self.attn_probs,     # intermediate buffer # backprop
                       self.attn_out_pre_residual)

        # residual conn #1
        self.attn_out_post_residual.copy_from(self.attn_out_pre_residual)
        self.attn_out_post_residual += self.X_pre_ln_attn

        layerNorm(self.attn_out_post_residual,
                  self.ln_ffn.gamma,
                  self.ln_ffn.beta,
                  self.ffn_input)

        # feedForward(input, w0, b0, w1, b1, hidden_buffer, output)
        feedForward(self.ffn_input,
                    self.ffn_weights.w0,
                    self.ffn_weights.b0,
                    self.ffn_weights.w1,
                    self.ffn_weights.b1,
                    self.ffn_hidden,
                    self.ffn_out)

        output.copy_from(self.ffn_out)
        output += self.attn_out_post_residual

    fn __del__(deinit self):
        # not sure since nested, leave structs alone?
        self.X_pre_ln_attn.ptr.free()
        self.X_post_ln_attn.ptr.free()

        self.Q.ptr.free()
        self.K.ptr.free()
        self.V.ptr.free()
        
        self.attn_scores.ptr.free()
        self.attn_probs.ptr.free()
        self.attn_out_pre_residual.ptr.free()
        self.attn_out_post_residual.ptr.free()
        self.ffn_input.ptr.free()
        self.ffn_hidden.ptr.free()
        self.ffn_out.ptr.free()

struct LLM():
    #comptime params = ModelParams
    comptime num_transformer_blocks = 1 << 2
    var blocks: InlineArray[TransformerBlock, Self.num_transformer_blocks]
    var ln_final_weights: LayerNormWeights
    var output_weights: OutputWeights
    comptime output_layout = Layout.row_major(ModelParams.seq_len, ModelParams.vocab_size)
    var logits: LayoutTensor[ftype, Self.output_layout, MutAnyOrigin]
    var output: LayoutTensor[ftype, Self.output_layout, MutAnyOrigin]

    fn __init__(out self): # allocate buffers and pre-fill / load etc.
        self.blocks = type_of(self.blocks)(fill = TransformerBlock.initRandom())
        self.ln_final_weights = LayerNormWeights.initRandom()
        self.output_weights = OutputWeights.initRandom()
        self.logits = zeroTensorHeap[Self.output_layout]()
        self.output = zeroTensorHeap[Self.output_layout]()

    fn forward(mut self, X: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin],
                out output: LayoutTensor[ftype, self.output_layout, MutAnyOrigin]):
        """
        Caller owns memory of logits.
        """
        var block_output = X
        for i in range(len(self.blocks)):
            self.blocks[i].forward(block_output, self.blocks[i].ffn_out)
            block_output = self.blocks[i].ffn_out

        layerNorm(block_output, self.ln_final_weights.gamma, self.ln_final_weights.beta, self.logits)

        weightAndBias(self.logits, self.output_weights.W, self.output_weights.b, output)
        #naiveSoftmax(output)

    fn __del__(deinit self):
        self.logits.ptr.free()
        self.output.ptr.free()
