from random import seed
from helpers import systemInfo, randTensorHeap
from attention import ftype, LLM, TransformerBlock

fn main():
    seed(42)
    systemInfo[ftype]()

    var llm = LLM()
    var x = randTensorHeap[TransformerBlock.X_layout]()
    print(x)
    print(llm.forward(x))
