from layout import Layout, LayoutTensor
from arena import BumpArenaAllocator

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
    var token_embeddings: LayoutTensor[ftype, Self.token_embeddings_layout, MutAnyOrigin]
    var position_embeddings: LayoutTensor[ftype, Self.position_embeddings_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    
    @staticmethod
    fn sizeInBytes() -> Int: ...
    
    @staticmethod
    fn initRandom(out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02): ...


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
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    @staticmethod
    fn initRandom(out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02): ...


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
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    @staticmethod
    fn initRandom(out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02): ...


struct LayerNormWeights(Copyable, Movable, Weights):
    """Just the learnable parameters for layer normalization"""
    comptime gamma_layout = Layout.row_major(ModelParams.d_model)
    comptime beta_layout = Layout.row_major(ModelParams.d_model)
    
    var gamma: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]
    var beta: LayoutTensor[ftype, Self.beta_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    @staticmethod
    fn initRandom(out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02): ...


struct OutputWeights(Copyable, Movable, Weights):
    """Just the learnable parameters for final output projection"""
    comptime W_layout = Layout.row_major(ModelParams.d_model, ModelParams.vocab_size)
    comptime b_layout = Layout.row_major(ModelParams.vocab_size)
    
    var W: LayoutTensor[ftype, Self.W_layout, MutAnyOrigin]
    var b: LayoutTensor[ftype, Self.b_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    @staticmethod
    fn initRandom(out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02): ...


struct TransformerBlockWeights(Copyable):
    """All learnable parameters for one transformer block"""
    var ln_attn: LayerNormWeights
    var attn: AttentionWeights
    var ln_ffn: LayerNormWeights
    var ffn: FFNWeights
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    @staticmethod
    fn initRandom(out self: Self, mut arena: BumpArenaAllocator, std: Float64 = 0.02): ...


struct LLMWeights:
    """All learnable parameters for the entire model - can be shared across batches"""
    var arena: BumpArenaAllocator
    var embedding: EmbeddingWeights
    var blocks: List[TransformerBlockWeights]
    var ln_final: LayerNormWeights
    var output: OutputWeights
    
    fn __init__(out self): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    fn saveToFile(self, path: String): ...
    fn loadFromFile(out self, path: String): ...


# ============================================================================
# ACTIVATIONS - Forward pass intermediate values (one per batch element)
# ============================================================================

struct EmbeddingActivations:
    """Forward pass buffers for embeddings"""
    var embedded_X: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


struct AttentionActivations:
    """Forward pass buffers for attention - what you need to compute attention"""
    comptime X_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    comptime Q_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_k)
    comptime K_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_k)
    comptime V_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_v)
    comptime attn_scores_layout = Layout.row_major(ModelParams.seq_len, ModelParams.seq_len)
    comptime attn_out_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    
    var X_post_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]
    var Q: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var K: LayoutTensor[ftype, Self.K_layout, MutAnyOrigin]
    var V: LayoutTensor[ftype, Self.V_layout, MutAnyOrigin]
    var attn_scores: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var attn_probs: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var attn_out_pre_residual: LayoutTensor[ftype, Self.attn_out_layout, MutAnyOrigin]
    var attn_out_post_residual: LayoutTensor[ftype, Self.attn_out_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


struct FFNActivations:
    """Forward pass buffers for feed-forward network"""
    comptime input_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    comptime hidden_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_ff)
    comptime output_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    
    var ffn_input: LayoutTensor[ftype, Self.input_layout, MutAnyOrigin]
    var ffn_hidden: LayoutTensor[ftype, Self.hidden_layout, MutAnyOrigin]
    var ffn_out: LayoutTensor[ftype, Self.output_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


struct TransformerBlockActivations:
    """All forward pass buffers for one transformer block"""
    comptime X_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    
    var X_pre_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]
    var attn: AttentionActivations
    var ffn: FFNActivations
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


struct LLMActivations:
    """All forward pass buffers for one sequence - allocate one per batch element"""
    var arena: BumpArenaAllocator
    var input_token_ids: LayoutTensor[token_itype, Layout.row_major(ModelParams.seq_len), MutAnyOrigin]
    var embedding: EmbeddingActivations
    var blocks: List[TransformerBlockActivations]
    var final_ln_output: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin]
    var logits: LayoutTensor[ftype, Layout.row_major(ModelParams.seq_len, ModelParams.vocab_size), MutAnyOrigin]
    
    fn __init__(out self): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


# ============================================================================
# GRADIENTS - Backward pass gradient accumulators (mirrors Weights structure)
# ============================================================================

struct EmbeddingGradients:
    """Gradient accumulators for embeddings"""
    comptime token_embeddings_layout = Layout.row_major(ModelParams.vocab_size, ModelParams.d_model)
    comptime position_embeddings_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    
    var token_embeddings_grad: LayoutTensor[ftype, Self.token_embeddings_layout, MutAnyOrigin]
    var position_embeddings_grad: LayoutTensor[ftype, Self.position_embeddings_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    fn zero(mut self): ...


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
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    fn zero(mut self): ...


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
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    fn zero(mut self): ...


struct LayerNormGradients:
    """Gradient accumulators for layer normalization"""
    comptime gamma_layout = Layout.row_major(ModelParams.d_model)
    comptime beta_layout = Layout.row_major(ModelParams.d_model)
    
    var gamma_grad: LayoutTensor[ftype, Self.gamma_layout, MutAnyOrigin]
    var beta_grad: LayoutTensor[ftype, Self.beta_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    fn zero(mut self): ...


struct OutputGradients:
    """Gradient accumulators for final output projection"""
    comptime W_layout = Layout.row_major(ModelParams.d_model, ModelParams.vocab_size)
    comptime b_layout = Layout.row_major(ModelParams.vocab_size)
    
    var W_grad: LayoutTensor[ftype, Self.W_layout, MutAnyOrigin]
    var b_grad: LayoutTensor[ftype, Self.b_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    fn zero(mut self): ...


struct TransformerBlockGradients:
    """All gradient accumulators for one transformer block"""
    var ln_attn: LayerNormGradients
    var attn: AttentionGradients
    var ln_ffn: LayerNormGradients
    var ffn: FFNGradients
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    fn zero(mut self): ...


struct LLMGradients:
    """All gradient accumulators for the entire model - accumulate over batch"""
    var arena: BumpArenaAllocator
    var embedding: EmbeddingGradients
    var blocks: List[TransformerBlockGradients]
    var ln_final: LayerNormGradients
    var output: OutputGradients
    
    fn __init__(out self): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...
    fn zero(mut self): ...  # Call before accumulating gradients for a new batch


# ============================================================================
# BACKWARD BUFFERS - Intermediate buffers needed during backprop
# These are like "activations" but for the backward pass
# ============================================================================

struct AttentionBackwardBuffers:
    """Buffers needed during attention backward pass"""
    comptime X_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    comptime Q_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_k)
    comptime attn_scores_layout = Layout.row_major(ModelParams.seq_len, ModelParams.seq_len)
    
    var d_Q: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var d_K: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var d_V: LayoutTensor[ftype, Self.Q_layout, MutAnyOrigin]
    var d_attn_scores: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var d_attn_probs: LayoutTensor[ftype, Self.attn_scores_layout, MutAnyOrigin]
    var d_X_post_ln_attn: LayoutTensor[ftype, Self.X_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


struct FFNBackwardBuffers:
    """Buffers needed during FFN backward pass"""
    comptime input_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_model)
    comptime hidden_layout = Layout.row_major(ModelParams.seq_len, ModelParams.d_ff)
    
    var d_ffn_input: LayoutTensor[ftype, Self.input_layout, MutAnyOrigin]
    var d_ffn_hidden: LayoutTensor[ftype, Self.hidden_layout, MutAnyOrigin]
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


struct TransformerBlockBackwardBuffers:
    """All backward buffers for one transformer block"""
    var attn: AttentionBackwardBuffers
    var ffn: FFNBackwardBuffers
    
    fn __init__(out self, mut arena: BumpArenaAllocator): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


struct LLMBackwardBuffers:
    """All backward pass buffers - allocate once for training"""
    var arena: BumpArenaAllocator
    var blocks: List[TransformerBlockBackwardBuffers]
    var d_final_ln_input: LayoutTensor[ftype, TransformerBlock.X_layout, MutAnyOrigin]
    
    fn __init__(out self): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...


# ============================================================================
# OPTIMIZER STATE - For algorithms like Adam that need extra buffers
# ============================================================================

struct AdamOptimizerState:
    """First and second moment estimates for Adam optimizer"""
    var arena: BumpArenaAllocator
    
    # Mirrors the structure of LLMGradients, but stores momentum
    var embedding_m: EmbeddingGradients  # first moment
    var embedding_v: EmbeddingGradients  # second moment
    var blocks_m: List[TransformerBlockGradients]
    var blocks_v: List[TransformerBlockGradients]
    var ln_final_m: LayerNormGradients
    var ln_final_v: LayerNormGradients
    var output_m: OutputGradients
    var output_v: OutputGradients
    
    var t: Int  # timestep for bias correction
    
    fn __init__(out self): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...  # 2x the size of LLMGradients
    
    fn step(
        mut self, 
        mut weights: LLMWeights, 
        gradients: LLMGradients, 
        lr: Float64, 
        beta1: Float64 = 0.9, 
        beta2: Float64 = 0.999, 
        eps: Float64 = 1e-8
    ): ...


struct SGDOptimizerState:
    """For SGD with momentum (optional)"""
    var arena: BumpArenaAllocator
    var velocity: LLMGradients  # Only need one copy for momentum
    
    fn __init__(out self): ...
    @staticmethod
    fn sizeInBytes() -> Int: ...  # 1x the size of LLMGradients
    
    fn step(mut self, mut weights: LLMWeights, gradients: LLMGradients, lr: Float64, momentum: Float64 = 0.9): ...


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
            
            var loss_grad = compute_loss_gradient(activations.logits, targets[i])
            backward(weights, activations, backward_buffers, loss_grad, gradients)
        
        # Update weights with accumulated gradients
        optimizer.step(weights, gradients, lr=0.001)


fn memory_efficient_inference():
    """For inference only, you don't need gradients or optimizer state"""
    var weights = LLMWeights()  # ~300MB for your model
    var activations = LLMActivations()  # ~10MB per sequence
    
    # Total memory: ~310MB instead of ~1GB if everything was bundled together
