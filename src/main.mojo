from random import seed
from sys import stderr
from time import perf_counter_ns
from pathlib import Path
from testing import assert_equal

from helpers import systemInfo, randTensorHeap
from attention import ftype, LLM, TransformerBlock, ModelParams
from tokenizer import ASCIITokenizer
from cliparser import TokenizerParser

fn printIntSpan[o: Origin](tokens: Span[Int, o]):
    var num_to_print = min(len(tokens), 100)
    for i in range(num_to_print):
        print(tokens[i], end = ", ")
    print()

fn main():
    seed(42)
    systemInfo[ftype]()

    var args = TokenizerParser()
    try:
        var tokenizer = ASCIITokenizer.load("./models/bpe_1024_shakespeare.tok")
        assert_equal(tokenizer.vocab_size, ModelParams.vocab_size)
        with open(args.input_filename, "r") as f:
            var text = f.read()
            var input_tokens = tokenizer.encode(text)
            var test_input = List[Int](input_tokens[:ModelParams.seq_len])
            printIntSpan(test_input)
            var llm = LLM()
            comptime times = 1
            var start = perf_counter_ns()
            for i in range(times):
                var result = llm.forward(test_input)
                var predicted_token = llm.getNextTokenGreedy()
                print("predicted token:", tokenier.decodeToken(predicted_token)
            var end = perf_counter_ns()
            print("time", (end - start) // 1_000_000 , "ms for", times, "runs")

    except e:
        print(e, file = stderr)
