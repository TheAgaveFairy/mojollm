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
from ops import naiveDotProductBatched, dotProductSlices, naiveSoftmaxBatched, layerNorm, feedForward, naiveAttention

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
    comptime wff0_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_ff)
    comptime wff1_layout = Layout.row_major(ModelParams.d_ff, ModelParams.d_model)
    var wff0: LayoutTensor[ftype, Self.wff0_layout, MutAnyOrigin]
    var wff1: LayoutTensor[ftype, Self.wff1_layout, MutAnyOrigin]

    comptime bff0_layout = Layout.row_major(ModelParams.d_ff)
    comptime bff1_layout = Layout.row_major(ModelParams.d_model)
    var bff0: LayoutTensor[ftype, Self.bff0_layout, MutAnyOrigin]
    var bff1: LayoutTensor[ftype, Self.bff1_layout, MutAnyOrigin]

    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    fn __init__(out self):
        print("FFWeights __init__() ERROR", file = stderr)

    @staticmethod
    fn initRandom(out self: Self):
        self.wff0 = randTensorHeap[Self.wff0_layout]()
        self.wff1 = randTensorHeap[Self.wff1_layout]()

        self.bff0 = randTensorHeap[Self.bff0_layout]()
        self.bff1 = randTensorHeap[Self.bff1_layout]()

    fn __del__(deinit self):
        self.wff0.ptr.free()
        self.wff1.ptr.free()
        self.bff0.ptr.free()
        self.bff1.ptr.free()

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
    comptime W_output_layout = Layout.row_major(ModelParams.d_model, ModelParams.vocab_size)
    comptime b_output_layout = Layout.row_major(ModelParams.vocab_size)
    var W_output: LayoutTensor[ftype, Self.W_output_layout, MutAnyOrigin]
    var b_output: LayoutTensor[ftype, Self.b_output_layout, MutAnyOrigin]

    fn __init__(out self):
        print("OutputWeights __init__() ERROR", file = stderr)
        pass

    @staticmethod
    fn initRandom(out self: Self):
        self.W_output = randTensorHeap[Self.W_output_layout]()
        self.b_output = randTensorHeap[Self.b_output_layout]()

    fn __del__(deinit self):
        self.W_output.ptr.free()
        self.B_output.ptr.free()

struct TransformerBlock(Copyable): # decoder, also would take Layer and/or Weights trait
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True
    var ln_attn: LayerNormWeights
    var attn_weights: AttentionWeights # single head, causal masking
    var ffn_weights: FFWeights
    var ln_ffn: LayerNormWeights

    # intermediate buffers
    comptime Q_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_k) # is K_Layout
    comptime V_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_v)
    var Q: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin] # intermediate buffers
    var K: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var V: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]

    var attn_scores: LayoutTensor[ftype, Self., MutAnyOrigin]
    var attn_probs: LayoutTensor[ftype, Self., MutAnyOrigin]
    var attn_out: LayoutTensor[ftype, Self., MutAnyOrigin]
    var ffn_hidden: LayoutTensor[ftype, Self., MutAnyOrigin]

    @staticmethod
    fn initRandom(out self: Self):
        self.ln_attn = LayerNormWeights.initRandom()
        self.attn_weights = AttentionWeights.initRandom()
        self.ffn_weights = FFWeights.initRandom()
        self.ln_ffn = LayerNormWeights.initRandom()

    # could this make sense as a pattern?
    fn forward[layout: Layout,
               out_layout: Layout](
                    self,
                    X: LayoutTensor[ftype, layout, MutAnyOrigin],
                    output: LayoutTensor[ftype, out_layout, MutAnyOrigin]):

        # layerNorm input
        # mha w/ causal masking
        # residual conn
        # layerNorm again
        # ffn
        # residual
        layerNorm(X, self.ln_attn.gamma, self.ln_attn.beta) # 'x' is inout
        naiveAttention(self.attn_weights.Q,
                       self.attn_weights.K,
                       self.attn_weights.V,
                       self.attn_weights.scores,
                       output)


        naiveAttention(X, self.att)


    fn __del__(deinit self):
        # not sure since nested, leave structs alone?
        # clear buffers
        self.Q.ptr.free()
        self.K.ptr.free()
        self.V.ptr.free()
        self.attn_scores.ptr.free()
        self.attn_probs.ptr.free()
        self.attn_out.ptr.free()
        self.ffn_hidden.ptr.free()

struct LLM():
    #comptime params = ModelParams
    comptime num_transformer_blocks = 1 << 2
    var blocks: InlineArray[TransformerBlock, Self.num_transformer_blocks]
    var ln_final_weights: LayerNormWeights
    var output_weights: OutputWeights

    fn __init__(out self): # allocate buffers and pre-fill / load etc.
        self.blocks = type_of(self.blocks)(fill = TransformerBlock.initRandom())
        self.ln_final_weights = LayerNormWeights.initRandom()
        self.output_weights = OutputWeights.initRandom()

    fn forward[layout: Layout](mut self, X: LayoutTensor[ftype, layout, MutAnyOrigin],
                               out logits: LayoutTensor[ftype, Layout.row_major(ModelParams.seq_len, ModelParams.vocab_size), MutAnyOrigin]):
        """
        Caller owns memory of logits.
        """
        var logits_storage = alloc[sftype](logits.layout.size())
        logits = type_of(logits)(logits_storage)

        for block in range(len(self.blocks)):
            blocks[block].forward(X)

        layerNorm(X, ln_final_weights)
        gemm(X, output_weights, logits)
    
        _ = """
        # some functions I wrote weren't setup for batching so you'll see them handled as such
        @parameter
        for b in range(ModelParams.batch_size):
            var x_slice_temp = X.slice[Slice(0, X.shape[1]()), Slice(0, X.shape[2]()), IndexList[2](1,2)](b)
            var x_slice = rebind[LayoutTensor[ftype, Layout.row_major(ModelParams.seq_len, ModelParams.d_model), MutAnyOrigin]](x_slice_temp)
            layerNorm(x_slice, self.ln0_weights.gamma, self.ln0_weights.beta)

        # TODO: add biases
        naiveDotProductBatched(X, self.attn_weights.W_q, self.Q)
        naiveDotProductBatched(X, self.attn_weights.W_k, self.K)
        naiveDotProductBatched(X, self.attn_weights.W_v, self.V)

        # BEGIN TRANSFORMER BLOCK, WE WOULD STACK THIS MANY TIMES (how exactly, I don't know)
        #for block in self.blocks:
            #block.forward(input_here)
        comptime attn_output_layout = Layout.row_major(ModelParams.batch_size, ModelParams.seq_len, ModelParams.d_v)
        #var output_buffer = alloc[sftype](output_layout.size())
        var attn_output = LayoutTensor[ftype, attn_output_layout, MutAnyOrigin].stack_allocation()#(output_buffer)

        # takes the whole batch
        naiveAttention(self.Q, self.K, self.V, attn_output)
        
        # residual connection
        attn_output += X

        # allocate the output_buffer for the feedForward as ff_output ?
        comptime ff_output_layout = Layout.row_major(ModelParams.batch_size, ModelParams.seq_len, ModelParams.d_model)
        var ff_output = LayoutTensor[ftype, ff_output_layout, MutAnyOrigin].stack_allocation()

        @parameter
        for b in range(ModelParams.batch_size):
            # get the attn_output slice
            layerNorm(attn_output_slice, self.ln1_weights.gamma, self.ln1_weights.beta)
            feedForward(attn_output_slice, self.wffn_weights_and_biases, ff_output_slice) # etc - i'll change signatures later to accept the weight-holding structs, to become consistent on what takes batches vs slices, etc)
        ff_output += attn_output
        # END TRANSFORMER BLOCK "LOOP" HERE.
        
        # final Layer norm -> linear projection to vocab, output logits(?), final softmax, walk me through this
        @parameter
        for b in range(ModelParams.batch_size):
            # get slices as needed for now
            layerNorm(ff_output_slice, self.ln2_weights)
        
            comptime logits_layout = Layout.row_major(ModelParams.seq_len, ModelParams.vocab_size)
            var logits = LayoutTensor[ftype, logits_layout, MutAnyOrigin].stack_allocation()
        
            dotProductSlices(ff_output_slice, self.output_weights.W_output, logits)
            logits += self.output_weights.b_output
        
        naiveSoftmaxBatched(final_output)

        # do something with final output
        """

fn main():
    seed(42)
    systemInfo[ftype]()

    var llm = LLM()
    benchmark.compiler.keep(llm)
