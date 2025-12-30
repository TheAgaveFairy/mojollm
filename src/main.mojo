from random import seed
from helpers import systemInfo, randTensorHeap
from attention import ftype, LLM, TransformerBlock

fn main():
    seed(42)
    systemInfo[ftype]()

    var llm = LLM()
    print("LLM()")
    var x = randTensorHeap[TransformerBlock.X_layout]()
    print("x.runtime_layout\n\t", x.runtime_layout)
    _ = llm.forward(x)
    print("forward results done\n\t")#, llm.forward(x))
