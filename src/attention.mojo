from layout import Layout, LayoutTensor
from math import sqrt, exp, log, ceildiv
from random import random_float64, random_si64, randint, randn, rand, seed
from sys.info import (
    simd_bit_width,
    simd_byte_width,
    simd_width_of,
    num_logical_cores,
    size_of,
    align_of,
)
from sys import stderr, is_big_endian, is_defined
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
    ColorsEnum,
    coloredString,
)
from activation_fn import ActivationFunction, ReLU
from ops import (
    weightAndBias,
    naiveSoftmax,
    layerNorm,
    feedForward,
    naiveAttention,
    applyCausalMask,
)
from arena import BumpArenaAllocator

comptime ftype = DType.float32
comptime sftype = Scalar[ftype]  # 's' prefix = 'S'calar
comptime nelts = simd_width_of[ftype]()

# token IDs stored as:
comptime token_itype = DType.uint16 # needs to fit vocab_size (~50k for GPT-2)
comptime display = True if is_defined["DISPLAY"]() else False


fn _myTensorCopyFrom[
    layout_a: Layout, layout_b: Layout, d_type: DType
](
    *,
    src: LayoutTensor[d_type, layout_a],
    dest: LayoutTensor[d_type, layout_b, MutAnyOrigin],
    transposed: Bool = False
):
    """Automagically handles transposition."""
    constrained[
        layout_a.rank() == layout_b.rank(),
        "Invalid tensor ranks at _myTensorCopyFrom",
    ]()
    comptime m = src.shape[0]()
    comptime n = src.shape[1]()
    comptime mm = dest.shape[0]()
    comptime nn = dest.shape[1]()
    comptime equal = m == mm and n == nn
    comptime transposed_valid = m == nn and n == mm
    comptime valid = equal or transposed_valid and layout_a.size() == layout_b.size()  # just to be safe
    constrained[valid, "Invalid tensor shapes at _myTensorCopyFrom"]()

    if not transposed:
        memcpy(dest=dest.ptr, src=src.ptr, count=comptime(layout_a.size()))
    else:
        # DO NOT PARAMETERIZE THE FOR LOOPS!
        for i in range(m):
            for j in range(n):
                dest[j, i] = src[i, j]


fn _arenaTensorHelper[
    layout: Layout, d_type: DType
](
    mut arena: BumpArenaAllocator, *, random: Bool = False, std: Float64 = 0.02
) -> LayoutTensor[d_type, layout, MutAnyOrigin]:
    var offset_before = arena.offset
    var ptr = arena.alloc[Scalar[d_type]](comptime(layout.size()))
    var offset_after = arena.offset
    var expected = comptime(layout.size()) * size_of[Scalar[d_type]]()
    var actual = offset_after - offset_before
    if expected != actual: # TODO: not sure this should be needed, Arena should handle?
        os.abort(
            "Allocation failure! Expected: {} != Actual: {}".format(
                expected, actual
            )
        )
    var tensor = LayoutTensor[d_type, layout, MutAnyOrigin](ptr)
    if random:
        randn(ptr, comptime(layout.size()), 0, std)
    return tensor


struct ModelParams(Writable):
    comptime num_transformer_blocks = 1 << 4
    comptime vocab_size = 1 << 13

    comptime max_batch_size = 1 << 4  # hmm
    comptime seq_len = 1 << 5
    comptime d_model = 1 << 6

    comptime d_k = Self.d_model
    comptime d_v = Self.d_model

    comptime d_ff = Self.d_model << 2  # d_model * 4 is common, apparently

    @staticmethod
    fn __str__() -> String:
        var result = """
ModelParams:
    comptime num_transformer_blocks = {}
    comptime vocab_size = {}

    comptime max_batch_size = {} # not in use right now
    comptime seq_len = {}
    comptime d_model = {}

    comptime d_k = Self.d_model
    comptime d_v = Self.d_model

    comptime d_ff = Self.d_model << 2  # d_model * 4 is common, apparently\n""".format(
            ModelParams.num_transformer_blocks,
            ModelParams.vocab_size,
            ModelParams.max_batch_size,
            ModelParams.seq_len,
            ModelParams.d_model,
        )
        return result


trait Weights:
    """Defaultable removed b/c arena."""

    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    @staticmethod
    fn initRandom(out self: Self, mut arena: BumpArenaAllocator, std: Float64):
        ...

    @staticmethod
    fn sizeInBytes() -> Int:
        ...

    # TODO: implement saving / loading
    # @staticmethod
    # fn initFromFile(out self):
    #     pass
    # fn saveToFile(self):
    #    pass


struct EmbeddingWeights(Copyable, Weights):
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
        token_ids: InlineArray[
            Int, ModelParams.seq_len
        ],  # TODO: should this be InlineArray[Int, seq_len]
        mut X: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin],
    ):
        """
        Helper function for the forward pass of my LLM. Just wanted where this
        is called to fit onto a single screen, honestly.
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
            # TODO: __add__ explodes compile time bug (_elementwise_binary_with_broadcast)
            for i in range(ModelParams.d_model):
                output_slice[i] += tok_emb[i] + pos_emb[i]
            # output_slice += tok_emb
            # output_slice += pos_emb

    fn __init__(out self, mut arena: BumpArenaAllocator):
        self.token_embeddings = _arenaTensorHelper[
            Self.token_embeddings_layout, ftype
        ](arena)
        self.position_embeddings = _arenaTensorHelper[
            Self.position_embeddings_layout, ftype
        ](arena)
        self.token_embeddings_grad = _arenaTensorHelper[
            Self.token_embeddings_layout, ftype
        ](arena)
        self.position_embeddings_grad = _arenaTensorHelper[
            Self.position_embeddings_layout, ftype
        ](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime(Self.token_embeddings_layout.size()) * 2
        result_in_sftype += comptime(Self.position_embeddings_layout.size()) * 2
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(
        out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02
    ):
        self = Self(arena)
        fillTensorRand(self.token_embeddings, std)
        fillTensorRand(self.position_embeddings, std)


struct AttentionWeights(Copyable, Movable, Weights):
    comptime W_q_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_k)
    comptime W_v_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_v)
    var W_q: LayoutTensor[ftype, Self.W_q_layout, MutAnyOrigin]
    var W_k: LayoutTensor[ftype, Self.W_q_layout, MutAnyOrigin]
    var W_v: LayoutTensor[ftype, Self.W_v_layout, MutAnyOrigin]

    comptime b_q_layout = Layout.row_major(ModelParams.d_k)
    comptime b_v_layout = Layout.row_major(ModelParams.d_v)
    var b_q: LayoutTensor[ftype, Self.b_q_layout, MutAnyOrigin]
    var b_k: LayoutTensor[ftype, Self.b_q_layout, MutAnyOrigin]
    var b_v: LayoutTensor[ftype, Self.b_v_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: BumpArenaAllocator):
        # weights
        self.W_q = _arenaTensorHelper[Self.W_q_layout, ftype](arena)
        self.W_k = _arenaTensorHelper[Self.W_q_layout, ftype](arena)
        self.W_v = _arenaTensorHelper[Self.W_v_layout, ftype](arena)
        # biases
        self.b_q = _arenaTensorHelper[Self.b_q_layout, ftype](arena)
        self.b_k = _arenaTensorHelper[Self.b_q_layout, ftype](arena)
        self.b_v = _arenaTensorHelper[Self.b_v_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime(Self.W_q_layout.size()) * 2
        result_in_sftype += comptime(Self.W_v_layout.size())
        result_in_sftype += comptime(Self.b_q_layout.size()) * 2
        result_in_sftype += comptime(Self.b_v_layout.size())
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(
        out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02
    ):
        self = Self(arena)
        fillTensorRand(self.W_q, std)
        fillTensorRand(self.W_k, std)
        fillTensorRand(self.W_v, std)
        # biases stay at zeros


struct FFWeights(Copyable, Movable, Weights):
    comptime w0_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_ff)
    comptime w1_layout = Layout.row_major(ModelParams.d_ff, ModelParams.d_model)
    var w0: LayoutTensor[ftype, Self.w0_layout, MutAnyOrigin]
    var w1: LayoutTensor[ftype, Self.w1_layout, MutAnyOrigin]

    comptime b0_layout = Layout.row_major(ModelParams.d_ff)
    comptime b1_layout = Layout.row_major(ModelParams.d_model)
    var b0: LayoutTensor[ftype, Self.b0_layout, MutAnyOrigin]
    var b1: LayoutTensor[ftype, Self.b1_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: BumpArenaAllocator):
        self.w0 = _arenaTensorHelper[Self.w0_layout, ftype](arena)
        self.b0 = _arenaTensorHelper[Self.b0_layout, ftype](arena)
        self.w1 = _arenaTensorHelper[Self.w1_layout, ftype](arena)
        self.b1 = _arenaTensorHelper[Self.b1_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime(Self.w0_layout.size())
        result_in_sftype += comptime(Self.b0_layout.size())
        result_in_sftype += comptime(Self.w1_layout.size())
        result_in_sftype += comptime(Self.b1_layout.size())
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(
        out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02
    ):
        self = Self(arena)
        fillTensorRand(self.w0, std)
        fillTensorRand(self.w1, std)
        # biases stay at zeros


struct LayerNormWeights(Copyable, Movable, Weights):
    comptime gamma_layout = Layout.row_major(ModelParams.d_model)
    var gamma: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]
    var beta: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: BumpArenaAllocator):
        self.gamma = _arenaTensorHelper[Self.gamma_layout, ftype](arena)
        self.beta = _arenaTensorHelper[Self.gamma_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime(Self.gamma_layout.size()) * 2
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(
        out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.1
    ):
        self = Self(arena)
        fillTensorRand(self.gamma, std)
        # biases stay at zeros


struct OutputWeights(Copyable, Movable, Weights):
    comptime W_layout = Layout.row_major(
        ModelParams.d_model, ModelParams.vocab_size
    )
    comptime b_layout = Layout.row_major(ModelParams.vocab_size)
    var W: LayoutTensor[ftype, Self.W_layout, MutAnyOrigin]
    var b: LayoutTensor[ftype, Self.b_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: BumpArenaAllocator):
        self.W = _arenaTensorHelper[Self.W_layout, ftype](arena)
        self.b = _arenaTensorHelper[Self.b_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime(Self.W_layout.size())
        result_in_sftype += comptime(Self.b_layout.size())
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(
        out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02
    ):
        self = Self(arena)
        fillTensorRand(self.W, std)
        # biases stay at zeros

struct TransformerBlock(Copyable):  # decoder, should this take Weights trait?
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
    var Q: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
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
    var ffn_input: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]
    var ffn_hidden: LayoutTensor[ftype, Self.ffn_hidden_layout, MutAnyOrigin]
    var ffn_out: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: BumpArenaAllocator):
        self.ln_attn = LayerNormWeights(arena)
        self.attn_weights = AttentionWeights(arena)
        self.ln_ffn = LayerNormWeights(arena)
        self.ffn_weights = FFWeights(arena)

        self.X_pre_ln_attn = _arenaTensorHelper[Self.X_layout, ftype](arena)
        self.X_post_ln_attn = _arenaTensorHelper[Self.X_layout, ftype](arena)

        self.Q = _arenaTensorHelper[Self.Q_layout, ftype](arena)
        self.K = _arenaTensorHelper[Self.Q_layout, ftype](arena)
        self.V = _arenaTensorHelper[Self.V_layout, ftype](arena)

        self.attn_scores = _arenaTensorHelper[Self.attn_scores_layout, ftype](
            arena
        )
        self.attn_probs = _arenaTensorHelper[Self.attn_scores_layout, ftype](
            arena
        )
        self.attn_out_pre_residual = _arenaTensorHelper[Self.V_layout, ftype](
            arena
        )
        self.attn_out_post_residual = _arenaTensorHelper[Self.V_layout, ftype](
            arena
        )
        self.ffn_input = _arenaTensorHelper[Self.V_layout, ftype](arena)
        self.ffn_hidden = _arenaTensorHelper[Self.ffn_hidden_layout, ftype](
            arena
        )
        self.ffn_out = _arenaTensorHelper[Self.X_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        var result_in_bytes = 0
        result_in_bytes += LayerNormWeights.sizeInBytes()
        result_in_bytes += AttentionWeights.sizeInBytes()
        result_in_bytes += LayerNormWeights.sizeInBytes()
        result_in_bytes += FFWeights.sizeInBytes()

        result_in_sftype += comptime(Self.X_layout.size()) * 2  # pre & post ln attn
        result_in_sftype += comptime(Self.Q_layout.size()) * 2  # Q, K
        result_in_sftype += comptime(Self.V_layout.size())  # V

        result_in_sftype += comptime(Self.attn_scores_layout.size()) * 2  # scores, probs
        result_in_sftype += comptime(Self.attn_scores_layout.size()) * 2  # scores, probs
        result_in_sftype += (
            comptime(Self.V_layout.size()) * 3
        )  # pre and post residuals, into ffn
        result_in_sftype += comptime(Self.ffn_hidden_layout.size())
        result_in_sftype += comptime(Self.X_layout.size())
        return result_in_sftype * size_of[sftype]() + result_in_bytes

    @staticmethod
    fn initRandom(
        out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02
    ):
        self = Self(arena)
        # TODO: could reduce startup time by initing to random directly
        self.ln_attn = LayerNormWeights.initRandom(arena)
        self.attn_weights = AttentionWeights.initRandom(arena)
        self.ln_ffn = LayerNormWeights.initRandom(arena)
        self.ffn_weights = FFWeights.initRandom(arena)

        fillTensorRand(self.X_pre_ln_attn, std)
        fillTensorRand(self.X_post_ln_attn, std)

        fillTensorRand(self.Q, std)
        fillTensorRand(self.K, std)
        fillTensorRand(self.V, std)

        # initialize attn_scores, attn_out, ffn_hidden, ffn_output to zeros ...
        fillTensorRand(self.attn_scores, std)
        fillTensorRand(self.attn_probs, std)
        fillTensorRand(self.attn_out_pre_residual, std)
        fillTensorRand(self.attn_out_post_residual, std)
        fillTensorRand(self.ffn_input, std)
        fillTensorRand(self.ffn_hidden, std)
        fillTensorRand(self.ffn_out, std)

    fn forward[
        layout: Layout
    ](
        mut self,
        X: LayoutTensor[ftype, layout, MutAnyOrigin],
        mut output: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        if display:
            print("begin tb forward")
            print("\tlayerNorm1")
        # layerNorm(input, gamma, beta, output)
        _myTensorCopyFrom(src=X, dest=self.X_pre_ln_attn)
        printTensorSlice(self.X_pre_ln_attn, "X_pre_ln_attn")
        layerNorm(
            self.X_pre_ln_attn,
            self.ln_attn.gamma,
            self.ln_attn.beta,
            self.X_post_ln_attn,
        )
        printTensorSlice(self.X_post_ln_attn, "X_post_ln_attn")

        if display:
            print("\tgenerate QKV")
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
        _myTensorCopyFrom(
            src=self.attn_out_pre_residual, dest=self.attn_out_post_residual
        )
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
        _myTensorCopyFrom(src=self.ffn_out, dest=output)
        # output += self.attn_out_post_residual
        for i in range(output.shape[0]()):
            for j in range(output.shape[1]()):
                output[i, j] += self.attn_out_post_residual[i, j]


fn printTensorSlice[
    layout: Layout
](tensor: LayoutTensor[ftype, layout, MutAnyOrigin], name: String):
    if not display:
        return
    print("Tensor", name, ":\n")
    comptime tile = 4
    var tslice = tensor.slice[
        Slice(0, tile), Slice(0, tile), IndexList[2](0, 1)
    ](IndexList[1](0))
    print(tslice, "\n")


struct LLM:
    var arena: BumpArenaAllocator
    # store token ids so we can update token embeddings
    var input_token_ids: LayoutTensor[
        token_itype, Layout.row_major(ModelParams.seq_len), MutAnyOrigin
    ]  # could be an InlineArray etc
    # to take tokens and create a valid input:
    var embedding_weights: EmbeddingWeights
    var embedded_X: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin]
    # attention blocks
    var blocks: List[TransformerBlock]
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
        var arena_size_in_bytes = Self.sizeInBytes()
        self.arena = BumpArenaAllocator(arena_size_in_bytes)
        self.arena.clear()  # memset_zeros the whole thing

        self.input_token_ids = _arenaTensorHelper[
            Layout.row_major(ModelParams.seq_len), token_itype
        ](self.arena)
        self.embedded_X = _arenaTensorHelper[TransformerBlock.X_layout, ftype](
            self.arena
        )
        self.embedding_weights = EmbeddingWeights.initRandom(self.arena)

        # this will create a single temp TransformerBlock, copy it to each location, then delete the temp
        self.blocks = type_of(self.blocks)(
            length=ModelParams.num_transformer_blocks,
            fill=TransformerBlock.initRandom(self.arena),
        )
        self.ln_final_weights = LayerNormWeights.initRandom(self.arena)
        self.output_weights = OutputWeights.initRandom(self.arena)
        self.final_ln_output = _arenaTensorHelper[
            TransformerBlock.X_layout, ftype
        ](self.arena)
        self.logits = _arenaTensorHelper[Self.output_layout, ftype](self.arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        """
        For setting up our Arena.
        All of our model's LayoutTensors will go in this for CPU!
        Doesn't *need* to be comptime, but it provides me some sanity.
        """
        var result_in_sftype = 0
        var result_in_itype = 0
        var result_in_bytes = 0
        result_in_itype += comptime(ModelParams.seq_len)

        result_in_sftype += comptime(TransformerBlock.X_layout.size())

        result_in_bytes += EmbeddingWeights.sizeInBytes()
        result_in_bytes += (
            TransformerBlock.sizeInBytes() * ModelParams.num_transformer_blocks
        )
        result_in_bytes += LayerNormWeights.sizeInBytes()
        result_in_bytes += OutputWeights.sizeInBytes()

        result_in_sftype += comptime(TransformerBlock.X_layout.size())
        result_in_sftype += comptime(Self.output_layout.size())

        return (
            (result_in_sftype * size_of[sftype]())
            + (result_in_itype * size_of[Scalar[token_itype]]())
            + result_in_bytes
        )

    fn forward(
        mut self,
        token_ids: InlineArray[
            Int, ModelParams.seq_len
        ],  # TODO: should this be InlineArray[Int, seq_len]
        out output: LayoutTensor[ftype, Self.output_layout, MutAnyOrigin],
    ):
        """
        Caller needs to free memory of final output.
        """
        print("Start forward:", "+" * 100)
        # assert_equal(ModelParams.seq_len, len(tokens))
        self.embedding_weights.embedTokens(token_ids, self.embedded_X)
        printTensorSlice(self.embedded_X, "embedded X")

        var block_output = self.embedded_X.copy()
        for i in range(len(self.blocks)):
            if display:
                print("LLM TransformerBlock", i)
            var ffn_out_temp = LayoutTensor[
                ftype, TransformerBlock.X_layout, MutAnyOrigin
            ].stack_allocation()
            self.blocks[i].forward(block_output, ffn_out_temp)
            _myTensorCopyFrom(src=ffn_out_temp, dest=self.blocks[i].ffn_out)
            block_output = self.blocks[i].ffn_out
            printTensorSlice(
                self.blocks[i].ffn_out, "block " + String(i) + " ffn_out"
            )

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
        _myTensorCopyFrom(src=output, dest=self.logits)
        # naiveSoftmax(output)

    fn getNextTokenGreedy(self) -> Int:
        """
        Call *after* a forward pass. This is not an end-to-end prediction path,
        just an abstraction so we can do temperature based, top-K, etc. types
        of token selection. DETERMINISTIC.
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
        print(coloredString("LLM __del__()", ColorsEnum.COLOR_PURPLE))
        self.arena.buffer.free()  # clear()
