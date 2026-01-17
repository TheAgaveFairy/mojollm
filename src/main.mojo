from random import seed
from sys import stderr
from time import perf_counter_ns
from pathlib import Path
from testing import assert_equal
from layout import Layout, LayoutTensor
from benchmark.compiler import keep

from helpers import systemInfo, randTensorHeap, ColorsEnum, coloredString
from attention import ftype, LLM, TransformerBlock, ModelParams
from tokenizer import ASCIITokenizer
from cliparser import TokenizerParser

fn prettyPrintBytes(bytes: Int) -> String:
    var fbytes = Float64(bytes)
    if fbytes < 1024:
        return "{}B".format(bytes)
    elif fbytes < 1024 * 1024:
        return "{}KB".format(round(fbytes / 1024, 4))
    elif fbytes < 1024 * 1024 * 1024:
        return "{}MB".format(round(fbytes / (1024 * 1024), 4))
    else:
        return "{}GB".format(round(fbytes / (1024 * 1024 * 1024), 4))

fn printIntSpan[o: Origin](tokens: Span[Int, o]):
    var num_to_print = min(len(tokens), 100)
    for i in range(num_to_print):
        print(tokens[i], end = ", ")
    print()

fn splitInput(toks: List[Int]) -> List[List[Int]]:
    var n = len(toks)
    comptime sl = ModelParams.seq_len
    var num_sequences = n // sl # floordiv
    var output = List[List[Int]](capacity = num_sequences)
    for i in range(0, num_sequences):
        var start = i * sl
        #var end = (i + 1) * sl
        var temp = List[Int](capacity = sl)
        for t in range(sl):
            temp.append(toks[start + t])
        if len(temp) == sl:
            output.append(temp^)
    return output^

fn tempPrintOutput[layout: Layout](output: LayoutTensor[ftype, layout, MutAnyOrigin]):
    comptime last_row = ModelParams.seq_len - 1
    comptime limit = 5
    comptime show_me = min(ModelParams.d_model, limit)
    print("Output, limit", limit, ":\n")
    for i in range(show_me):
        print(output[last_row, i], end = ", ")
    print()

fn main() raises:
    seed(420)
    systemInfo[ftype]()

    var args = TokenizerParser()
    try:
        with open(args.input_filename, "r") as f:
            var tokenizer = ASCIITokenizer.load("./models/bpe_1024_shakespeare.tok")
            assert_equal(tokenizer.vocab_size, ModelParams.vocab_size)
            var text = f.read()
            var input_tokens = tokenizer.encode(text)
            var test_input = splitInput(input_tokens)
            #printIntSpan(test_input)
            print(ModelParams.__str__())
            print("LLM Size In Bytes()", prettyPrintBytes(LLM.sizeInBytes()))
            var llm = LLM()
            comptime times = 1
            var start = perf_counter_ns()
            for i in range(times):
                for tok_list in test_input[:5]:
                    print(tok_list)
                    var output = llm.forward(tok_list)
                    tempPrintOutput(output)
                    var predicted_token = llm.getNextTokenGreedy()
                    print("predicted token:", coloredString(tokenizer.decodeToken(predicted_token)))
            var end = perf_counter_ns()
            print("time", (end - start) // 1_000 , "us for", times, "runs")
            #print("Capacity {} offset {}".format(llm.arena.capacity, llm.arena.offset))
            print("biases for QKV, ffn, consider looking up what W_o is, causal masking, comment things well, cross entropy loss, transformer grad buffers, optimizer trait")

    except e:
        print(e, file = stderr)
