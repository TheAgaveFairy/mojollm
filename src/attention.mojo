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
from reflection import get_linkage_name
from compile import compile_info
import benchmark  # run, Unit.ms

# from kernels.nn.softmax import softmax

from helpers import (
    showProgress,
    cleanFunctionName,
    systemInfo,
    randTensorHeap,
    zeroTensorHeap,
    compareBuffers,
    fillTensorRand,
)
from activation_fn import ActivationFunction, ReLU
from ops import (
    weightAndBias,
    naiveSoftmax,
    layerNorm,
    feedForward,
    naiveAttention,
)

comptime ftype = DType.float32
comptime sftype = Scalar[ftype]  # 's' prefix = 'S'calar
comptime nelts = simd_width_of[ftype]()

# token IDs stored as:
comptime token_itype = DType.uint16


struct ModelParams:
    comptime num_transformer_blocks = 1 << 2
    comptime vocab_size = 1 << 10

    comptime max_batch_size = 1 << 5  # hmm
    comptime seq_len = 1 << 3
    comptime d_model = 1 << 6

    comptime d_k = Self.d_model
    comptime d_v = Self.d_model

    comptime d_ff = Self.d_model << 2  # d_model * 4 is common, apparently


trait Weights(Defaultable):
    @staticmethod
    fn initRandom(out self: Self):
        ...

    # TODO: implement saving / loading
    # @staticmethod
    # fn initFromFile(out self):
    #     pass
    # fn saveToFile(self):
    #    pass


struct EmbeddingWeights(Copyable, Weights):
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    comptime token_embeddings_layout = Layout.row_major(
        ModelParams.vocab_size, ModelParams.d_model
    )
    comptime position_embeddings_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )
    var token_embeddings: LayoutTensor[
        ftype, Self.token_embeddings_layout, MutAnyOrigin
    ]
    var position_embeddings: LayoutTensor[
        ftype, Self.position_embeddings_layout, MutAnyOrigin
    ]

    var token_embeddings_grad: LayoutTensor[
        ftype, Self.token_embeddings_layout, MutAnyOrigin
    ]
    var position_embeddings_grad: LayoutTensor[
        ftype, Self.position_embeddings_layout, MutAnyOrigin
    ]

    @always_inline("nodebug")
    fn embedTokens(
        self,
        token_ids: List[Int],
        mut X: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin],
    ):
        """
        Helper function for the forward pass of my LLM. The rebinds etc are
        just a bit much, and I'd prefer that forward to abstract away the
        deepest of tedium, at least for now.
        """
        comptime d_model_slice = Slice(0, ModelParams.d_model)
        for i in range(ModelParams.seq_len):
            var tok_emb = self.token_embeddings.slice_1d[
                d_model_slice, IndexList[1](1)
            ](IndexList[1](token_ids[i]))
            var pos_emb = self.position_embeddings.slice_1d[
                d_model_slice, IndexList[1](1)
            ](IndexList[1](i))

            var output_slice = X.slice_1d[d_model_slice, IndexList[1](1)](
                IndexList[1](i)
            )
            output_slice += tok_emb
            output_slice += pos_emb

    fn __init__(out self):
        self.token_embeddings = type_of(self.token_embeddings)(
            alloc[sftype](Self.token_embeddings_layout.size())
        ).fill(0.0)
        self.position_embeddings = type_of(self.position_embeddings)(
            alloc[sftype](Self.position_embeddings_layout.size())
        ).fill(0.0)
        self.token_embeddings_grad = type_of(self.token_embeddings_grad)(
            alloc[sftype](Self.token_embeddings_layout.size())
        ).fill(0.0)
        self.position_embeddings_grad = type_of(self.position_embeddings_grad)(
            alloc[sftype](Self.position_embeddings_layout.size())
        ).fill(0.0)

    @staticmethod
    fn initRandom(out self: Self):
        self = Self()
        fillTensorRand(self.token_embeddings)
        fillTensorRand(self.position_embeddings)

    fn __del__(deinit self):
        print("EmbeddingWeights __del__()")
        self.token_embeddings.ptr.free()
        self.position_embeddings.ptr.free()
        self.token_embeddings_grad.ptr.free()
        self.position_embeddings_grad.ptr.free()


struct AttentionWeights(Copyable, Movable, Weights):
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
        self.W_q = type_of(self.W_q)(
            alloc[sftype](Self.W_q_layout.size())
        ).fill(0.0)
        self.W_k = type_of(self.W_k)(
            alloc[sftype](Self.W_q_layout.size())
        ).fill(0.0)
        self.W_v = type_of(self.W_v)(
            alloc[sftype](Self.W_v_layout.size())
        ).fill(0.0)
        self.b_q = type_of(self.b_q)(
            alloc[sftype](Self.b_q_layout.size())
        ).fill(0.0)
        self.b_k = type_of(self.b_k)(
            alloc[sftype](Self.b_q_layout.size())
        ).fill(0.0)
        self.b_v = type_of(self.b_v)(
            alloc[sftype](Self.b_v_layout.size())
        ).fill(0.0)

    @staticmethod
    fn initRandom(out self: Self):
        self = Self()
        fillTensorRand(self.W_q)
        fillTensorRand(self.W_q)
        fillTensorRand(self.W_v)

        fillTensorRand(self.b_q)
        fillTensorRand(self.b_q)
        fillTensorRand(self.b_v)

    fn __del__(deinit self):
        print("AttentionWeights __del__()")
        self.W_q.ptr.free()
        self.W_k.ptr.free()
        self.W_v.ptr.free()

        self.b_q.ptr.free()
        self.b_k.ptr.free()
        self.b_v.ptr.free()


struct FFWeights(Copyable, Movable, Weights):
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
        self.w0 = type_of(self.w0)(alloc[sftype](Self.w0_layout.size())).fill(
            0.0
        )
        self.b0 = type_of(self.b0)(alloc[sftype](Self.b0_layout.size())).fill(
            0.0
        )
        self.w1 = type_of(self.w1)(alloc[sftype](Self.w1_layout.size())).fill(
            0.0
        )
        self.b1 = type_of(self.b1)(alloc[sftype](Self.b1_layout.size())).fill(
            0.0
        )

    @staticmethod
    fn initRandom(out self: Self):
        self = Self()
        fillTensorRand(self.w0)
        fillTensorRand(self.b0)
        fillTensorRand(self.w1)
        fillTensorRand(self.b1)

    fn __del__(deinit self):
        print("FFWeights __del__()")
        self.w0.ptr.free()
        self.b0.ptr.free()
        self.w1.ptr.free()
        self.b1.ptr.free()


struct LayerNormWeights(Copyable, Movable, Weights):
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True
    comptime gamma_layout = Layout.row_major(ModelParams.d_model)
    var gamma: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]
    var beta: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]

    fn __init__(out self):
        self.gamma = type_of(self.gamma)(
            alloc[sftype](Self.gamma_layout.size())
        ).fill(0.0)
        self.beta = type_of(self.beta)(
            alloc[sftype](Self.gamma_layout.size())
        ).fill(0.0)

    @staticmethod
    fn initRandom(out self: Self):
        self = Self()
        fillTensorRand(self.gamma)
        fillTensorRand(self.beta)

    fn __del__(deinit self):
        print("LayerNormWeights __del__()")
        self.gamma.ptr.free()
        self.beta.ptr.free()


struct OutputWeights(Copyable, Movable, Weights):
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True
    comptime W_layout = Layout.row_major(
        ModelParams.d_model, ModelParams.vocab_size
    )
    comptime b_layout = Layout.row_major(ModelParams.vocab_size)
    var W: LayoutTensor[ftype, Self.W_layout, MutAnyOrigin]
    var b: LayoutTensor[ftype, Self.b_layout, MutAnyOrigin]

    fn __init__(out self):
        self.W = type_of(self.W)(alloc[sftype](Self.W_layout.size())).fill(0.0)
        self.b = type_of(self.b)(alloc[sftype](Self.b_layout.size())).fill(0.0)

    @staticmethod
    fn initRandom(out self: Self):
        self = Self()
        fillTensorRand(self.W)
        fillTensorRand(self.b)

    fn __del__(deinit self):
        print("OutputWeights __del__()")
        self.W.ptr.free()
        self.b.ptr.free()


struct TransformerBlock(
    Copyable
):  # decoder, also would take Layer and/or Weights trait
    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True
    var ln_attn: LayerNormWeights
    var attn_weights: AttentionWeights  # single head, causal masking
    var ffn_weights: FFWeights
    var ln_ffn: LayerNormWeights

    comptime X_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )
    var X_pre_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]
    var X_post_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]

    # intermediate buffers
    comptime Q_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_k
    )  # is K_Layout
    comptime V_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_v)
    var Q: LayoutTensor[
        ftype, Self.Q_layout, MutAnyOrigin
    ]  # intermediate buffers
    var K: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var V: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]

    comptime attn_scores_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.seq_len
    )
    var attn_scores: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var attn_probs: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]

    var attn_out_pre_residual: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]
    var attn_out_post_residual: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]

    comptime ffn_hidden_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_ff
    )
    comptime ffn_out_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )
    var ffn_input: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]
    var ffn_hidden: LayoutTensor[ftype, Self.ffn_hidden_layout, MutAnyOrigin]
    var ffn_out: LayoutTensor[ftype, Self.ffn_out_layout, MutAnyOrigin]

    fn __init__(out self):
        # TODO: setup zeroTensorRand for better readability
        self.ln_attn = LayerNormWeights()
        self.attn_weights = AttentionWeights()
        self.ln_ffn = LayerNormWeights()
        self.ffn_weights = FFWeights()

        self.X_pre_ln_attn = type_of(self.X_pre_ln_attn)(
            alloc[sftype](Self.X_layout.size())
        ).fill(0.0)
        self.X_post_ln_attn = type_of(self.X_post_ln_attn)(
            alloc[sftype](Self.X_layout.size())
        ).fill(0.0)

        self.Q = type_of(self.Q)(alloc[sftype](Self.Q_layout.size())).fill(0.0)
        self.K = type_of(self.K)(alloc[sftype](Self.Q_layout.size())).fill(0.0)
        self.V = type_of(self.V)(alloc[sftype](Self.V_layout.size())).fill(0.0)

        # initialize attn_scores, attn_out, ffn_hidden, ffn_output to zeros ...
        self.attn_scores = type_of(self.attn_scores)(
            alloc[sftype](Self.attn_scores_layout.size())
        ).fill(0.0)
        self.attn_probs = type_of(self.attn_probs)(
            alloc[sftype](Self.attn_scores_layout.size())
        ).fill(0.0)
        self.attn_out_pre_residual = type_of(self.attn_out_pre_residual)(
            alloc[sftype](Self.V_layout.size())
        ).fill(0.0)
        self.attn_out_post_residual = type_of(self.attn_out_post_residual)(
            alloc[sftype](Self.V_layout.size())
        ).fill(0.0)
        self.ffn_input = type_of(self.ffn_input)(
            alloc[sftype](Self.V_layout.size())
        ).fill(0.0)
        self.ffn_hidden = type_of(self.ffn_hidden)(
            alloc[sftype](Self.ffn_hidden_layout.size())
        ).fill(0.0)
        self.ffn_out = type_of(self.ffn_out)(
            alloc[sftype](Self.ffn_out_layout.size())
        ).fill(0.0)

    @staticmethod
    fn initRandom(out self: Self):
        self = Self()
        # TODO : old memory should be set in-place, this is awful
        self.ln_attn = LayerNormWeights.initRandom()
        self.attn_weights = AttentionWeights.initRandom()
        self.ln_ffn = LayerNormWeights.initRandom()
        self.ffn_weights = FFWeights.initRandom()
        # /END AWFUL

        fillTensorRand(self.X_pre_ln_attn)
        fillTensorRand(self.X_post_ln_attn)

        fillTensorRand(self.Q)
        fillTensorRand(self.K)
        fillTensorRand(self.V)

        # initialize attn_scores, attn_out, ffn_hidden, ffn_output to zeros ...
        fillTensorRand(self.attn_scores)
        fillTensorRand(self.attn_probs)
        fillTensorRand(self.attn_out_pre_residual)
        fillTensorRand(self.attn_out_post_residual)
        fillTensorRand(self.ffn_input)
        fillTensorRand(self.ffn_hidden)
        fillTensorRand(self.ffn_out)

    fn forward[
        layout: Layout, out_layout: Layout
    ](
        mut self,
        X: LayoutTensor[ftype, layout, MutAnyOrigin],
        mut output: LayoutTensor[ftype, out_layout, MutAnyOrigin],
        display: Bool = False,
    ):
        if display:
            print("begin tb forward")
            print("\tlayerNorm1")
        # layerNorm(input, gamma, beta, output)
        self.X_pre_ln_attn.copy_from(X)
        layerNorm(X, self.ln_attn.gamma, self.ln_attn.beta, self.X_post_ln_attn)

        if display:
            print("\tgenerate QKV")
        # weightAndBias(input, weight, bias, output)
        weightAndBias(
            self.X_post_ln_attn,
            self.attn_weights.W_q,
            self.attn_weights.b_q,
            self.Q,
        )
        weightAndBias(
            self.X_post_ln_attn,
            self.attn_weights.W_k,
            self.attn_weights.b_k,
            self.K,
        )
        weightAndBias(
            self.X_post_ln_attn,
            self.attn_weights.W_v,
            self.attn_weights.b_v,
            self.V,
        )
        
        if display:
            print("\tnaive attention")
        # causal masking not implemented YET
        naiveAttention(
            self.Q,
            self.K,
            self.V,
            self.attn_scores,  # intermediate buffer # backprop
            self.attn_probs,  # intermediate buffer # backprop
            self.attn_out_pre_residual,
        )

        if display:
            print("\tresidual conn #1")
        # residual conn #1
        self.attn_out_post_residual.copy_from(self.attn_out_pre_residual)
        self.attn_out_post_residual += self.X_pre_ln_attn

        if display:
            print("\tlayerNorm2")
        layerNorm(
            self.attn_out_post_residual,
            self.ln_ffn.gamma,
            self.ln_ffn.beta,
            self.ffn_input,
        )

        if display:
            print("\tfeedForward")
        # feedForward(input, w0, b0, w1, b1, hidden_buffer, output)
        feedForward(
            self.ffn_input,
            self.ffn_weights.w0,
            self.ffn_weights.b0,
            self.ffn_weights.w1,
            self.ffn_weights.b1,
            self.ffn_hidden,
            self.ffn_out,
        )
        if display:
            print("final residual")
        output.copy_from(self.ffn_out)
        output += self.attn_out_post_residual

    fn __del__(deinit self):
        print("TransformerBlock __del__()")
        # not sure since nested, leave structs alone?
        print("LET OS HANDLE TransformerBlock __del__() FOR NOW", file=stderr)
        _ = """
        self.X_pre_ln_attn.ptr.free()
        self.X_post_ln_attn.ptr.free()
        print("DEBUG Q")
        self.Q.ptr.free()
        print("DEBUG K")
        self.K.ptr.free()
        print("DEBUG V")
        self.V.ptr.free()

        print("DEBUG scores")
        self.attn_scores.ptr.free()
        print("DEBUG probs")
        self.attn_probs.ptr.free()
        print("DEBUG attn_out_pre")
        self.attn_out_pre_residual.ptr.free()
        print("DEBUG attn_out_post")
        self.attn_out_post_residual.ptr.free()
        print("DEBUG ffn_input")
        self.ffn_input.ptr.free()
        print("DEBUG ffn_hidden")
        self.ffn_hidden.ptr.free()
        print("DEBUG ffn_out")
        self.ffn_out.ptr.free()
        print("DEBUG DONE")
        """


struct LLM:
    # store token ids so we can update token embeddings
    var input_token_ids: LayoutTensor[
        token_itype, Layout.row_major(ModelParams.seq_len), MutAnyOrigin
    ]
    var embedded_X: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin]
    # to take tokens and create a valid input:
    var embedding_weights: EmbeddingWeights
    # attention blocks
    var blocks: InlineArray[
        TransformerBlock, ModelParams.num_transformer_blocks
    ]
    # layer norm -> output layer -> final_ln_output
    var ln_final_weights: LayerNormWeights
    var output_weights: OutputWeights
    comptime output_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.vocab_size
    )
    var final_ln_output: LayoutTensor[
        ftype, TransformerBlock.X_layout, MutAnyOrigin
    ]
    var logits: LayoutTensor[ftype, Self.output_layout, MutAnyOrigin]

    fn __init__(out self):  # allocate buffers and pre-fill / load etc.
        self.input_token_ids = type_of(self.input_token_ids)(
            alloc[Scalar[token_itype]](ModelParams.seq_len)
        ).fill(0)
        self.embedded_X = zeroTensorHeap[TransformerBlock.X_layout]()
        self.embedding_weights = EmbeddingWeights.initRandom()
        self.blocks = type_of(self.blocks)(fill=TransformerBlock.initRandom())
        self.ln_final_weights = LayerNormWeights.initRandom()
        self.output_weights = OutputWeights.initRandom()
        self.final_ln_output = zeroTensorHeap[TransformerBlock.X_layout]()
        self.logits = zeroTensorHeap[Self.output_layout]()

    fn forward(
        mut self,
        token_ids: List[Int],  # TODO: should this be InlineArray[Int, seq_len]
        out output: LayoutTensor[ftype, Self.output_layout, MutAnyOrigin],
        display: Bool = False
    ):
        """
        Caller needs to free memory of final output.
        """
        fn printTensorSlice[layout: Layout](tensor: LayoutTensor[ftype, layout, MutAnyOrigin], name: String):
            print("Tensor", name, ":\n")
            comptime tile = 4
            var tslice = tensor.slice[Slice(0, tile), Slice(0, tile), IndexList[2](0,1)](IndexList[1](0))
            print(tslice, "\n")
        # assert_equal(ModelParams.seq_len, len(tokens))
        self.embedding_weights.embedTokens(token_ids, self.embedded_X)
        #print("EMBEDDED X:\n", self.embedded_X,"\n\n")
        printTensorSlice(self.embedded_X, "embedded X")

        var block_output = self.embedded_X.copy()
        for i in range(len(self.blocks)):
            if display:
                print("LLM TransformerBlock", i)
            var ffn_out_temp = LayoutTensor[
                ftype, TransformerBlock.ffn_out_layout, MutAnyOrigin
            ].stack_allocation()
            self.blocks[i].forward(block_output, ffn_out_temp)
            self.blocks[i].ffn_out.copy_from(ffn_out_temp)
            block_output = self.blocks[i].ffn_out
            printTensorSlice(self.blocks[i].ffn_out, "block " + String(i) + " ffn_out")

        if display:
            print("LLM forward blocks done")
        layerNorm(
            block_output,
            self.ln_final_weights.gamma,
            self.ln_final_weights.beta,
            self.final_ln_output,
        )
        printTensorSlice(self.final_ln_output, "final ln output")

        if display:
            print("Final linear layer...")
        output = zeroTensorHeap[Self.output_layout]()
        weightAndBias(
            self.final_ln_output,
            self.output_weights.W,
            self.output_weights.b,
            output,
        )
        printTensorSlice(output, "final output")
        self.logits = output.copy()
        # naiveSoftmax(output)

    fn getNextTokenGreedy(self) -> Int:
        """
        Call *after* a forward pass. This is not an end-to-end prediction path,
        just an abstraction so we can do temperature based, top-K, etc. types
        of token selection.
        """
        var last_row = ModelParams.seq_len - 1
        var max_idx = 0
        var max_val = self.logits[last_row, 0]

        for i in range(ModelParams.vocab_size):
            var temp_val = self.logits[last_row, i]
            if temp_val > max_val:
                max_val = temp_val
                max_idx = i
        return max_idx

    fn __del__(deinit self):
        print("LLM __del__()")
        self.input_token_ids.ptr.free()
        self.logits.ptr.free()
