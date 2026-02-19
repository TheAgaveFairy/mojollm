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
    _trans,
    _myTensorCopyFrom,
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
from arena import BumpArenaAllocator, Allocator

comptime ftype = DType.float32
comptime sftype = Scalar[ftype]  # 's' prefix = 'S'calar
comptime nelts = simd_width_of[ftype]()

# token IDs stored as:
comptime token_itype = DType.uint16  # needs to fit vocab_size (~50k for GPT-2)
comptime display = True if is_defined["DISPLAY"]() else False

# CLAUDE 4.5 Sonnet ASSISTED REFACTOR FROM "attention.mojo"

# ============================================================================
# WEIGHTS ONLY - Learnable parameters that persist across batches
# ============================================================================


struct EmbeddingWeights(Copyable, Weights):
    """Just the learnable parameters for embeddings"""

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

    fn __init__(out self, mut arena: Allocator):
        self.token_embeddings = _arenaTensorHelper[
            Self.token_embeddings_layout, ftype
        ](arena)
        self.position_embeddings = _arenaTensorHelper[
            Self.position_embeddings_layout, ftype
        ](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.token_embeddings_layout.size())
        result_in_sftype += comptime (Self.position_embeddings_layout.size())
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(out self: Self, mut arena: Allocator, std: Float64 = 0.02):
        self = Self(arena)
        fillTensorRand(self.token_embeddings, std)
        fillTensorRand(self.position_embeddings, std)

    @always_inline("nodebug")
    fn embedTokens(
        self,
        token_ids: InlineArray[Int, ModelParams.seq_len],
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
            # FIXME: __add__ explodes compile time bug (_elementwise_binary_with_broadcast)
            for i in range(ModelParams.d_model):
                output_slice[i] += tok_emb[i] + pos_emb[i]
            # output_slice += tok_emb
            # output_slice += pos_emb


struct AttentionWeights(Copyable, Movable, Weights):
    """Just the learnable parameters for attention"""

    comptime W_q_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_k)
    comptime W_v_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_v)
    comptime b_q_layout = Layout.row_major(ModelParams.d_k)
    comptime b_v_layout = Layout.row_major(ModelParams.d_v)

    var W_q: LayoutTensor[ftype, Self.W_q_layout, MutAnyOrigin]
    var W_k: LayoutTensor[ftype, Self.W_q_layout, MutAnyOrigin]
    var W_v: LayoutTensor[ftype, Self.W_v_layout, MutAnyOrigin]
    var b_q: LayoutTensor[ftype, Self.b_q_layout, MutAnyOrigin]
    var b_k: LayoutTensor[ftype, Self.b_q_layout, MutAnyOrigin]
    var b_v: LayoutTensor[ftype, Self.b_v_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
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
        result_in_sftype += comptime (Self.W_q_layout.size()) * 2
        result_in_sftype += comptime (Self.W_v_layout.size())
        result_in_sftype += comptime (Self.b_q_layout.size()) * 2
        result_in_sftype += comptime (Self.b_v_layout.size())

        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(out self: Self, mut arena: Allocator, std: Float64 = 0.02):
        self = Self(arena)
        fillTensorRand(self.W_q, std)
        fillTensorRand(self.W_k, std)
        fillTensorRand(self.W_v, std)
        # biases stay at zeros


struct FFNWeights(Copyable, Movable, Weights):
    """Just the learnable parameters for feed-forward network"""

    comptime w0_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_ff)
    comptime w1_layout = Layout.row_major(ModelParams.d_ff, ModelParams.d_model)
    comptime b0_layout = Layout.row_major(ModelParams.d_ff)
    comptime b1_layout = Layout.row_major(ModelParams.d_model)

    var w0: LayoutTensor[ftype, Self.w0_layout, MutAnyOrigin]
    var w1: LayoutTensor[ftype, Self.w1_layout, MutAnyOrigin]
    var b0: LayoutTensor[ftype, Self.b0_layout, MutAnyOrigin]
    var b1: LayoutTensor[ftype, Self.b1_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.w0 = _arenaTensorHelper[Self.w0_layout, ftype](arena)
        self.b0 = _arenaTensorHelper[Self.b0_layout, ftype](arena)
        self.w1 = _arenaTensorHelper[Self.w1_layout, ftype](arena)
        self.b1 = _arenaTensorHelper[Self.b1_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.w0_layout.size())
        result_in_sftype += comptime (Self.b0_layout.size())
        result_in_sftype += comptime (Self.w1_layout.size())
        result_in_sftype += comptime (Self.b1_layout.size())
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(out self: Self, mut arena: Allocator, std: Float64 = 0.02):
        self = Self(arena)
        fillTensorRand(self.w0, std)
        fillTensorRand(self.w1, std)
        # biases stay at zeros


struct LayerNormWeights(Copyable, Movable, Weights):
    """Just the learnable parameters for layer normalization"""

    comptime gamma_layout = Layout.row_major(ModelParams.d_model)
    comptime beta_layout = Layout.row_major(ModelParams.d_model)

    var gamma: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]
    var beta: LayoutTensor[ftype, Self.beta_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.gamma = _arenaTensorHelper[Self.gamma_layout, ftype](arena)
        self.beta = _arenaTensorHelper[Self.gamma_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.gamma_layout.size()) * 2
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(out self: Self, mut arena: Allocator, std: Float64 = 0.02):
        self = Self(arena)
        fillTensorRand(self.gamma, std)
        # biases stay at zeros


struct OutputWeights(Copyable, Movable, Weights):
    """Just the learnable parameters for final output projection"""

    comptime W_layout = Layout.row_major(
        ModelParams.d_model, ModelParams.vocab_size
    )
    comptime b_layout = Layout.row_major(ModelParams.vocab_size)

    var W: LayoutTensor[ftype, Self.W_layout, MutAnyOrigin]
    var b: LayoutTensor[ftype, Self.b_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.W = _arenaTensorHelper[Self.W_layout, ftype](arena)
        self.b = _arenaTensorHelper[Self.b_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.W_layout.size())
        result_in_sftype += comptime (Self.b_layout.size())
        return result_in_sftype * size_of[sftype]()

    @staticmethod
    fn initRandom(out self: Self, mut arena: Allocator, std: Float64 = 0.02):
        self = Self(arena)
        fillTensorRand(self.W, std)
        # biases stay at zeros


struct TransformerBlockWeights(Copyable):
    """All learnable parameters for one transformer block"""

    var ln_attn: LayerNormWeights
    var attn: AttentionWeights
    var ln_ffn: LayerNormWeights
    var ffn: FFNWeights

    fn __init__(out self, mut arena: Allocator):
        self.ln_attn = LayerNormWeights(arena)
        self.attn_weights = AttentionWeights(arena)
        self.ln_ffn = LayerNormWeights(arena)
        self.ffn_weights = FFWeights(arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_bytes = 0

        result_in_bytes += LayerNormWeights.sizeInBytes()
        result_in_bytes += AttentionWeights.sizeInBytes()
        result_in_bytes += LayerNormWeights.sizeInBytes()
        result_in_bytes += FFWeights.sizeInBytes()
        return result_in_bytes

    @staticmethod
    fn initRandom(out self: Self, mut arena: Allocator, std: Float64 = 0.02):
        self = Self(arena)
        self.ln_attn = LayerNormWeights.initRandom(arena)
        self.attn_weights = AttentionWeights.initRandom(arena)
        self.ln_ffn = LayerNormWeights.initRandom(arena)
        self.ffn_weights = FFWeights.initRandom(arena)


struct LLMWeights:
    """All learnable parameters for the entire model - can be shared across batches
    """

    var arena: Allocator
    var embedding: EmbeddingWeights
    var blocks: List[TransformerBlockWeights]
    var ln_final: LayerNormWeights
    var output: OutputWeights

    fn __init__(out self):
        var arena_size_in_bytes = Self.sizeInBytes()
        self.arena = BumpArenaAllocator(arena_size_in_bytes)
        self.arena.clear()  # memset_zeros the whole thing

        self.embedding_weights = EmbeddingWeights.initRandom(self.arena)

        # this will create a single temp TransformerBlock, copy it to each location, then delete the temp
        self.blocks = type_of(self.blocks)(
            length=ModelParams.num_transformer_blocks,
            fill=TransformerBlock.initRandom(self.arena),
        )
        self.ln_final_weights = LayerNormWeights.initRandom(self.arena)
        self.output_weights = OutputWeights.initRandom(self.arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_bytes = 0

        result_in_bytes += EmbeddingWeights.sizeInBytes()
        result_in_bytes += (
            TransformerBlockWeights.sizeInBytes()
            * ModelParams.num_transformer_blocks
        )
        result_in_bytes += LayerNormWeights.sizeInBytes()
        result_in_bytes += OutputWeights.sizeInBytes()
        return result_in_bytes

    fn saveToFile(self, path: String):
        pass

    fn loadFromFile(out self, path: String):
        pass


# ============================================================================
# ACTIVATIONS - Forward pass intermediate values (one per batch element)
# ============================================================================


struct EmbeddingActivations:
    """Forward pass buffers for embeddings"""

    var embedded_X: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.embedded_X = _arenaTensorHelper[TransformerBlock.X_layout, ftype](
            arena
        )

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (TransformerBlock.X_layout.size())
        return result_in_sftype * size_of[sftype]()


struct AttentionActivations:
    """Forward pass buffers for attention - what you need to compute attention
    """

    comptime X_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )
    comptime Q_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_k)
    comptime K_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_k
    )  # Q
    comptime V_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_v)
    comptime attn_scores_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.seq_len
    )
    comptime attn_out_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )

    var X_post_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]
    var Q: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var K: LayoutTensor[ftype, Self.K_layout, MutAnyOrigin]
    var V: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]
    var attn_scores: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var attn_probs: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var attn_out_pre_residual: LayoutTensor[
        ftype, Self.attn_out_layout, MutAnyOrigin
    ]
    var attn_out_post_residual: LayoutTensor[
        ftype, Self.attn_out_layout, MutAnyOrigin
    ]

    fn __init__(out self, mut arena: Allocator):
        self.X_post_ln_attn = _arenaTensorHelper[Self.X_layout](arena)
        self.Q = _arenaTensorHelper[Self.Q_layout](arena)
        self.K = _arenaTensorHelper[Self.K_layout](arena)
        self.V = _arenaTensorHelper[Self.V_layout](arena)

        self.attn_scores = _arenaTensorHelper[Self.attn_scores_layout](arena)
        self.attn_probs = _arenaTensorHelper[Self.attn_scores_layout](arena)
        self.attn_out_pre_residual = _arenaTensorHelper[Self.attn_out_layout](
            arena
        )
        self.attn_out_post_residual = _arenaTensorHelper[Self.attn_out_layout](
            arena
        )

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.Q_layout.size())
        result_in_sftype += comptime (Self.K_layout.size())
        result_in_sftype += comptime (Self.V_layout.size())

        result_in_sftype += comptime (Self.attn_scores_layout.size())
        result_in_sftype += comptime (Self.attn_scores_layout.size())
        result_in_sftype += comptime (Self.attn_out_layout.size())
        result_in_sftype += comptime (Self.attn_out_layout.size())
        return result_in_sftype * size_of[sftype]()


struct FFNActivations:
    """Forward pass buffers for feed-forward network"""

    comptime input_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )
    comptime hidden_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_ff
    )
    comptime output_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )

    var ffn_input: LayoutTensor[ftype, Self.input_layout, MutAnyOrigin]
    var ffn_hidden: LayoutTensor[ftype, Self.hidden_layout, MutAnyOrigin]
    var ffn_out: LayoutTensor[ftype, Self.output_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.ffn_input = _arenaTensorHelper[Self.input_layout](arena)
        self.ffn_hidden = _arenaTensorHelper[Self.hidden_layout](arena)
        self.ffn_out = _arenaTensorHelper[Self.output_layout](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.input_layout.size())
        result_in_sftype += comptime (Self.hidden_layout.size())
        result_in_sftype += comptime (Self.output_layout.size())

        return result_in_sftype * size_of[sftype]()


struct TransformerBlockActivations:
    """All forward pass buffers for one transformer block"""

    comptime X_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )

    var X_pre_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]
    var attn: AttentionActivations
    var ffn: FFNActivations

    fn __init__(out self, mut arena: Allocator):
        self.X_pre_ln_attn = _arenaTensorHelper[Self.X_layout](arena)
        self.attn = AttentionActivations(arena)
        self.ffn = FFNActivations(arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        var result_in_bytes = 0
        result_in_sftype += comptime (Self.X_layout.size())

        result_in_bytes += AttentionActivations.sizeInBytes()
        result_in_bytes += FFNActivations.sizeInBytes()

        return result_in_sftype * size_of[sftype]() + result_in_bytes


struct LLMActivations:
    """All forward pass buffers for one sequence - allocate one per batch element
    """

    comptime input_layout = Layout.row_major(ModelParams.seq_len)
    comptime logits_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.vocab_size
    )

    # var arena: Allocator
    var input_token_ids: LayoutTensor[
        token_itype, Self.input_layout, MutAnyOrigin
    ]
    var embedding: EmbeddingActivations
    var blocks: List[TransformerBlockActivations]
    var final_ln_output: LayoutTensor[
        ftype, TransformerBlock.X_layout, MutAnyOrigin
    ]
    var logits: LayoutTensor[ftype, Self.logits_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.input_token_ids = AttentionActivations(arena)
        self.embedding = EmbeddingActivations(arena)
        # this will create a single temp TransformerBlock, copy it to each location, then delete the temp
        self.blocks = type_of(self.blocks)(
            length=ModelParams.num_transformer_blocks,
            fill=TransformerBlockActivations(self.arena),
        )
        self.final_ln_output = _arenaTensorHelper[TransformerBlock.X_layout](
            arena
        )
        self.logits = _arenaTensorHelper[Self.logits_layout](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        var result_in_bytes = 0
        result_in_sftype += comptime (Self.input_layout.size())

        result_in_bytes += EmbeddingActivations.sizeInBytes()
        result_in_bytes += (
            TransformerBlockActivations.sizeInBytes()
            * ModelParams.num_transformer_blocks
        )

        result_in_sftype += comptime (TransformerBlock.X_layout.size())
        result_in_sftype += comptime (Self.logits_layout.size())

        return result_in_sftype * size_of[sftype]() + result_in_bytes


# ============================================================================
# GRADIENTS - Backward pass gradient accumulators (mirrors Weights structure)
# ============================================================================


struct EmbeddingGradients:
    """Gradient accumulators for embeddings"""

    comptime token_embeddings_layout = Layout.row_major(
        ModelParams.vocab_size, ModelParams.d_model
    )
    comptime position_embeddings_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )

    var token_embeddings_grad: LayoutTensor[
        ftype, Self.token_embeddings_layout, MutAnyOrigin
    ]
    var position_embeddings_grad: LayoutTensor[
        ftype, Self.position_embeddings_layout, MutAnyOrigin
    ]

    fn __init__(out self, mut arena: Allocator):
        self.token_embeddings_grad = _arenaTensorHelper[
            Self.token_embeddings_layout
        ](arena)
        self.position_embeddings_grad = _arenaTensorHelper[
            Self.position_embeddings_layout
        ](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.token_embeddings_layout.size())
        result_in_sftype += comptime (Self.position_embeddings_layout.size())
        return result_in_sftype * size_of[sftype]()

    fn zero(mut self):
        memset_zero(
            self.token_embeddings_grad.ptr,
            comptime (Self.token_embeddings_layout.size()),
        )
        memset_zero(
            self.position_embeddings_grad.ptr,
            comptime (Self.position_embeddings_layout.size()),
        )


struct AttentionGradients:
    """Gradient accumulators for attention"""

    comptime W_q_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_k)
    comptime W_v_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_v)
    comptime b_q_layout = Layout.row_major(ModelParams.d_k)
    comptime b_v_layout = Layout.row_major(ModelParams.d_v)

    var W_q_grad: LayoutTensor[ftype, Self.W_q_layout, MutAnyOrigin]
    var W_k_grad: LayoutTensor[ftype, Self.W_q_layout, MutAnyOrigin]
    var W_v_grad: LayoutTensor[ftype, Self.W_v_layout, MutAnyOrigin]
    var b_q_grad: LayoutTensor[ftype, Self.b_q_layout, MutAnyOrigin]
    var b_k_grad: LayoutTensor[ftype, Self.b_q_layout, MutAnyOrigin]
    var b_v_grad: LayoutTensor[ftype, Self.b_v_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        # weights
        self.W_q_grad = _arenaTensorHelper[Self.W_q_layout, ftype](arena)
        self.W_k_grad = _arenaTensorHelper[Self.W_q_layout, ftype](arena)
        self.W_v_grad = _arenaTensorHelper[Self.W_v_layout, ftype](arena)
        # biases
        self.b_q_grad = _arenaTensorHelper[Self.b_q_layout, ftype](arena)
        self.b_k_grad = _arenaTensorHelper[Self.b_q_layout, ftype](arena)
        self.b_v_grad = _arenaTensorHelper[Self.b_v_layout, ftype](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.W_q_layout.size()) * 2
        result_in_sftype += comptime (Self.W_v_layout.size())
        result_in_sftype += comptime (Self.b_q_layout.size()) * 2
        result_in_sftype += comptime (Self.b_v_layout.size())

        return result_in_sftype * size_of[sftype]()

    fn zero(mut self):
        memset_zero(self.W_q_grad.ptr, comptime (Self.W_q_layout.size()))
        memset_zero(self.W_k_grad.ptr, comptime (Self.W_q_layout.size()))
        memset_zero(self.W_v_grad.ptr, comptime (Self.W_v_layout.size()))

        memset_zero(self.b_q_grad.ptr, comptime (Self.b_q_layout.size()))
        memset_zero(self.b_k_grad.ptr, comptime (Self.b_q_layout.size()))
        memset_zero(self.b_v_grad.ptr, comptime (Self.b_v_layout.size()))


struct FFNGradients:
    """Gradient accumulators for feed-forward network"""

    comptime w0_layout = Layout.row_major(ModelParams.d_model, ModelParams.d_ff)
    comptime w1_layout = Layout.row_major(ModelParams.d_ff, ModelParams.d_model)
    comptime b0_layout = Layout.row_major(ModelParams.d_ff)
    comptime b1_layout = Layout.row_major(ModelParams.d_model)

    var w0_grad: LayoutTensor[ftype, Self.w0_layout, MutAnyOrigin]
    var w1_grad: LayoutTensor[ftype, Self.w1_layout, MutAnyOrigin]
    var b0_grad: LayoutTensor[ftype, Self.b0_layout, MutAnyOrigin]
    var b1_grad: LayoutTensor[ftype, Self.b1_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.w0_grad = _arenaTensorHelper[Self.w0_layout](arena)
        self.w1_grad = _arenaTensorHelper[Self.w1_layout](arena)
        self.b0_grad = _arenaTensorHelper[Self.b0_layout](arena)
        self.b1_grad = _arenaTensorHelper[Self.b1_layout](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.w0_layout.size())
        result_in_sftype += comptime (Self.w1_layout.size())
        result_in_sftype += comptime (Self.b0_layout.size())
        result_in_sftype += comptime (Self.b1_layout.size())
        return result_in_sftype * size_of[sftype]()

    fn zero(mut self):
        memset_zero(self.w0_grad.ptr, comptime (Self.w0_layout.size()))
        memset_zero(self.w1_grad.ptr, comptime (Self.w1_layout.size()))
        memset_zero(self.b0_grad.ptr, comptime (Self.b0_layout.size()))
        memset_zero(self.b1_grad.ptr, comptime (Self.b1_layout.size()))


struct LayerNormGradients:
    """Gradient accumulators for layer normalization"""

    comptime gamma_layout = Layout.row_major(ModelParams.d_model)
    comptime beta_layout = Layout.row_major(ModelParams.d_model)

    var gamma_grad: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]
    var beta_grad: LayoutTensor[ftype, Self.beta_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.gamma_grad = _arenaTensorHelper[Self.gamma_layout](arena)
        self.beta_grad = _arenaTensorHelper[Self.beta_layout](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.gamma_layout.size())
        result_in_sftype += comptime (Self.beta_layout.size())
        return result_in_sftype * size_of[sftype]()

    fn zero(mut self):
        memset_zero(self.gamma_grad.ptr, comptime (Self.gamma_layout.size()))
        memset_zero(self.beta_grad.ptr, comptime (Self.beta_layout.size()))


struct OutputGradients:
    """Gradient accumulators for final output projection"""

    comptime W_layout = Layout.row_major(
        ModelParams.d_model, ModelParams.vocab_size
    )
    comptime b_layout = Layout.row_major(ModelParams.vocab_size)

    var W_grad: LayoutTensor[ftype, Self.W_layout, MutAnyOrigin]
    var b_grad: LayoutTensor[ftype, Self.b_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.W_grad = _arenaTensorHelper[Self.W_layout](arena)
        self.b_grad = _arenaTensorHelper[Self.W_layout](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.W_layout.size())
        result_in_sftype += comptime (Self.b_layout.size())
        return result_in_sftype * size_of[sftype]()

    fn zero(mut self):
        memset_zero(self.W_grad.ptr, comptime (Self.W_layout.size()))
        memset_zero(self.b_grad.ptr, comptime (Self.b_layout.size()))


struct TransformerBlockGradients:
    """All gradient accumulators for one transformer block"""

    var ln_attn: LayerNormGradients
    var attn: AttentionGradients
    var ln_ffn: LayerNormGradients
    var ffn: FFNGradients

    fn __init__(out self, mut arena: Allocator):
        self.ln_attn = LayerNormWeights(arena)
        self.attn_weights = AttentionWeights(arena)
        self.ln_ffn = LayerNormWeights(arena)
        self.ffn_weights = FFWeights(arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_bytes = 0

        result_in_bytes += LayerNormGradients.sizeInBytes()
        result_in_bytes += AttentionGradients.sizeInBytes()
        result_in_bytes += LayerNormGradients.sizeInBytes()
        result_in_bytes += FFNGradients.sizeInBytes()
        return result_in_bytes

    fn zero(mut self):
        self.ln_attn.zero()
        self.attn.zero()
        self.ln_ffn.zero()
        self.ffn.zero()


struct LLMGradients:
    """All gradient accumulators for the entire model - accumulate over batch"""

    # var arena: Allocator
    var embedding: EmbeddingGradients
    var blocks: List[TransformerBlockGradients]
    var ln_final: LayerNormGradients
    var output: OutputGradients

    fn __init__(out self, mut arena: Allocator):
        self.embedding = EmbeddingGradients(arena)
        self.blocks = type_of(self.blocks)(
            length=ModelParams.num_transformer_blocks,
            fill=TransformerBlockGradients(arena),
        )
        self.ln_final = LayerNormGradients(arena)
        self.output = OutputGradients(arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_bytes = 0
        result_in_bytes += EmbeddingGradients.sizeInBytes()
        result_in_bytes += (
            TransformerBlockGradients.sizeInBytes()
            * ModelParams.num_transformer_blocks
        )
        result_in_bytes += LayerNormGradients.sizeInBytes()
        result_in_bytes += OutputGradients.sizeInBytes()
        return result_in_bytes

    fn zero(mut self):
        self.embedding.zero()
        for block in self.blocks:
            block.zero()
        self.ln_final.zero()
        self.output.zero()


# ============================================================================
# BACKWARD BUFFERS - Intermediate buffers needed during backprop
# These are like "activations" but for the backward pass
# ============================================================================


struct AttentionBackwardBuffers:
    """Buffers needed during attention backward pass"""

    comptime X_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )
    comptime Q_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_k)
    # FIXME: V_layout = Layout.row_major(ModelParams., ModelParams.)
    comptime attn_scores_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.seq_len
    )

    var d_Q: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var d_K: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var d_V: LayoutTensor[
        ftype, Self.Q_layout, MutAnyOrigin
    ]  # TODO: LAYOUT WRONG!?
    var d_attn_scores: LayoutTensor[
        ftype, Self.attn_scores_layout, MutAnyOrigin
    ]
    var d_attn_probs: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var d_X_post_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.d_Q = _arenaTensorHelper[Self.Q_layout](arena)
        self.d_K = _arenaTensorHelper[Self.Q_layout](arena)
        self.d_V = _arenaTensorHelper[Self.Q_layout](arena)

        self.d_attn_scores = _arenaTensorHelper[Self.attn_scores_layout](arena)
        self.d_attn_probs = _arenaTensorHelper[Self.attn_scores_layout](arena)
        self.d_X_post_ln_attn = _arenaTensorHelper[Self.X_layout](arena)

    @staticmethod
    fn sizeInBytes():
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.Q_layout.size()) * 3
        result_in_sftype += comptime (Self.attn_scores_layout.size()) * 2
        result_in_sftype += comptime (Self.X_layout.size())
        return result_in_sftype * size_of[sftype]()


struct FFNBackwardBuffers:
    """Buffers needed during FFN backward pass"""

    comptime input_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_model
    )
    comptime hidden_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.d_ff
    )

    var d_ffn_input: LayoutTensor[ftype, Self.input_layout, MutAnyOrigin]
    var d_ffn_hidden: LayoutTensor[ftype, Self.hidden_layout, MutAnyOrigin]

    fn __init__(out self, mut arena: Allocator):
        self.d_ffn_input = _arenaTensorHelper[Self.input_layout](arena)
        self.d_ffn_input = _arenaTensorHelper[Self.hidden_layout](arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_sftype = 0
        result_in_sftype += comptime (Self.input_layout.size())
        result_in_sftype += comptime (Self.hidden_layout.size())
        return result_in_sftype * size_of[sftype]()


struct TransformerBlockBackwardBuffers:
    """All backward buffers for one transformer block"""

    var attn: AttentionBackwardBuffers
    var ffn: FFNBackwardBuffers

    fn __init__(out self, mut arena: Allocator):
        self.attn = AttentionBackwardBuffers(arena)
        self.ffn = FFNBackwardBuffers(arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_bytes = 0
        result_in_bytes += AttentionBackwardBuffers(arena)
        result_in_bytes += FFNBackwardBuffers(arena)
        return result_in_bytes


struct LLMBackwardBuffers:
    """All backward pass buffers - allocate once for training"""

    var blocks: List[TransformerBlockBackwardBuffers]
    var d_final_ln_input: LayoutTensor[
        ftype, TransformerBlock.X_layout, MutAnyOrigin
    ]

    fn __init__(out self):
        self.blocks = type_of(self.blocks)(
            length=ModelParams.num_transformer_blocks,
            fill=TransformerBlockBackwardBuffers(arena),
        )
        self.d_final_ln_input = _arenaTensorHelper[TransformerBlock.X_layout](
            arena
        )

    @staticmethod
    fn sizeInBytes() -> Int:
        var result_in_bytes = 0
        var result_in_sftype = 0
        result_in_bytes += (
            TransformerBlockBackwardBuffers.sizeInBytes()
            * ModelParams.num_transformer_blocks
        )
        result_in_sftype += comptime (TransformerBlock.X_layout.size())
        return result_in_sftype * size_of[sftype]() + result_in_bytes


# ============================================================================
# OPTIMIZER STATE - For algorithms like Adam that need extra buffers
# ============================================================================


struct AdamOptimizerState:
    """First and second moment estimates for Adam optimizer"""

    # Mirrors the structure of LLMGradients, but stores momentum
    var momentum: LLMGradients  # Only need one copy for momentum
    var velocity: LLMGradients  # Only need one copy for momentum

    var t: Int  # timestep for bias correction

    fn __init__(out self, mut arena: Allocator):
        self.momentum = LLMGradients(arena)
        self.velocity = LLMGradients(arena)
        self.t = 0

    @staticmethod
    fn sizeInBytes() -> Int:
        return 2 * LLMGradients.sizeInBytes()  # lazy today

    fn step(
        mut self,
        mut weights: LLMWeights,
        gradients: LLMGradients,
        lr: Float64,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        eps: Float64 = 1e-8,  # for stability
    ):
        ...


struct SGDOptimizerState:
    """For SGD with momentum (optional)"""

    var velocity: LLMGradients  # Only need one copy for momentum

    fn __init__(out self, mut arena: Allocator):
        self.velocity = LLMGradients(arena)

    @staticmethod
    fn sizeInBytes() -> Int:
        return LLMGradients.sizeInBytes()

    fn step(
        mut self,
        mut weights: LLMWeights,
        gradients: LLMGradients,
        lr: Float64,
        momentum: Float64 = 0.9,
    ):
        ...


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


fn inference_example():
    """How you'd use this for inference with batching"""
    # Load weights once
    var weights = LLMWeights()
    weights.loadFromFile("model.weights")

    # Allocate activations for each batch slot
    var batch_size = 4
    var batch_activations = List[LLMActivations](capacity=batch_size)
    for i in range(batch_size):
        batch_activations.append(LLMActivations())

    # Run inference on each sequence
    for i in range(batch_size):
        var tokens = get_input_tokens(i)
        forward(weights, batch_activations[i], tokens)
        var next_token = greedy_decode(batch_activations[i].logits)


fn training_example():
    """How you'd use this for training"""
    # Initialize everything
    var weights = LLMWeights()  # or load from checkpoint
    var gradients = LLMGradients()
    var optimizer = AdamOptimizerState()

    # For each training step
    var batch_size = 8
    for step in range(num_steps):
        # Zero gradients at start of batch
        gradients.zero()

        # Accumulate gradients over batch
        for i in range(batch_size):
            var activations = LLMActivations()
            var backward_buffers = LLMBackwardBuffers()

            var tokens = get_training_tokens(i)
            forward(weights, activations, tokens)

            var loss_grad = compute_loss_gradient(
                activations.logits, targets[i]
            )
            backward(
                weights, activations, backward_buffers, loss_grad, gradients
            )

        # Update weights with accumulated gradients
        optimizer.step(weights, gradients, lr=0.001)


fn memory_efficient_inference():
    """For inference only, you don't need gradients or optimizer state"""
    var weights = LLMWeights()
    var activations = LLMActivations()

    ...
