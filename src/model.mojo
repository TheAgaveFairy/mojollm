from layout import Layout

from optimizer import Optimizer, SGD, ADAM
from llm import LLMWeights, LLMGradients, SGDOptimizerState
from attention import ftype, sftype, token_itype, ModelParams, display


struct TrainableModel:
    var arena: BumpArenaAllocator
    var weights: LLMWeights
    var grads: LLMGradients
    var optimizer: SGD
    # var activation_function: Some[ActivationFunction] # where do i want to track what act_fn I'm using?

    # there might be a better place to define this output shape?
    comptime output_layout = Layout.row_major(
        ModelParams.seq_len, ModelParams.vocab_size
    )

    def __init__(out self):
        var arena_size = (
            LLMWeights.sizeInBytes() + LLMGradients.sizeInBytes()
        )  # + optimizer state size
        self.arena = BumpArenaAllocator(arena_size)
        self.weights = LLMWeights(arena)
        self.grads = LLMGradients(arena)
        self.optimizer = SGD()

    def forward(
        mut self,
        token_ids: InlineArray[Int, ModelParams.seq_len],
        output: LayoutTensor[ftype, Self.output_layout, MutAnyOrigin],
    ):
        pass
