from random import seed
from helpers import systemInfo, randTensorHeap
from attention import ftype, LLM, TransformerBlock
from time import perf_counter_ns

fn main():
    seed(42)
    systemInfo[ftype]()

    var llm = LLM()
    print("LLM()")
    var x = randTensorHeap[TransformerBlock.X_layout]()
    
    print("x.runtime_layout\n\t", x.runtime_layout)
    comptime times = 1
    var start = perf_counter_ns()
    for i in range(times):
        var result = llm.forward(x)
        print(result[0,0])
    var end = perf_counter_ns()
    print("time", (end - start) // 1_000_000 , "ms for", times, "runs")
   # print("forward results done\n\t", result[0,0])
