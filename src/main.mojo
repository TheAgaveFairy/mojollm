from std.random import seed
from std.sys import stderr
from std.sys.info import has_accelerator
from std.time import perf_counter_ns
from std.pathlib import Path
from std.testing import assert_equal
from layout import Layout, LayoutTensor
from std.benchmark.compiler import keep

from helpers import (
    systemInfo,
    randTensorHeap,
    ColorsEnum,
    coloredString,
    showProgress,
)
from attention import (
    ftype,
    token_itype,
    LLM,
    TransformerBlock,
    ModelParams,
    display,
)
from tokenizer import Tokenizer
from cliparser import TokenizerParser
from ops import crossEntropyLoss
from mylogger import LLMInferenceRecord, LLMTrainRecord, CSVLogger, getDateTime


def prettyPrintBytes(bytes: Int) -> String:
    var fbytes = Float64(bytes)
    if fbytes < 1024:
        return "{}B".format(bytes)
    elif fbytes < 1024 * 1024:
        return "{}KB".format(round(fbytes / 1024, 4))
    elif fbytes < 1024 * 1024 * 1024:
        return "{}MB".format(round(fbytes / (1024 * 1024), 4))
    else:
        return "{}GB".format(round(fbytes / (1024 * 1024 * 1024), 4))


def printIntSpan[o: Origin](tokens: Span[Int, o]):
    var num_to_print = min(len(tokens), 100)
    for i in range(num_to_print):
        print(tokens[i], end=", ")
    print()


comptime toks_arr = InlineArray[Int, ModelParams.seq_len]


# TODO: move to tokenizer
def splitInput(toks: List[Int]) -> List[toks_arr]:
    """Left pads and splits into seq_len chunks! No special tokens handled here.
    """
    # TODO: handle special tokens?
    comptime PAD = 0

    var n = len(toks)
    comptime sl = ModelParams.seq_len
    var num_sequences = n // sl  # floordiv
    var output = List[toks_arr](capacity=num_sequences)
    for i in range(num_sequences):
        var start = i * sl
        # var end = (i + 1) * sl
        var temp = toks_arr(fill=PAD)  # fill with '<|PAD|>' signifier, aka 0
        for t in range(sl):
            temp[t] = toks[start + t]
        output.append(temp^)
    if n % sl:
        var left_pad_count = sl - (n % sl)
        var temp = toks_arr(fill=PAD)  # fill with '<|PAD|>' signifier, aka 0
        for t in range(sl - left_pad_count):
            temp[left_pad_count + t] = toks[num_sequences * sl + t]
        output.append(temp^)

    return output^


def tempPrintOutput[
    layout: Layout
](output: LayoutTensor[ftype, layout, MutAnyOrigin]):
    comptime last_row = ModelParams.seq_len - 1
    comptime limit = 5
    comptime show_me = min(ModelParams.d_model, limit)
    print("Output, limit", limit, ":\n")
    for i in range(show_me):
        print(output[last_row, i], end=", ")
    print()


def logFileName(mode: String = "test") -> Path:
    var temp_log_name = Path("./logs/")
    temp_log_name /= getDateTime() + "_" + mode + ".csv"
    return temp_log_name


def main() raises:
    seed(42)
    systemInfo[ftype]()

    var args = TokenizerParser()
    if args.had_error:
        return

    var logger = CSVLogger[LLMInferenceRecord](
        "./logs/main_log_test_delete.csv"
    )  # (logFileName())
    var device = "7600X 32GB"  # "RTX 3070" if has_accelerator() else "7600X"
    try:
        with open(args.input_filename, "r") as f:
            var tokenizer = Tokenizer.load("./models/bpe_8192_shakespeare.tok")
            assert_equal(tokenizer.vocab_size, ModelParams.vocab_size)
            var text = f.read()
            var input_tokens = tokenizer.encode(text)
            var test_input = splitInput(input_tokens)
            # printIntSpan(test_input)

            print(ModelParams())
            print("LLM Size In Bytes()", prettyPrintBytes(LLM.sizeInBytes()))
            var llm = LLM()

            comptime times = 1
            var start = perf_counter_ns()
            for i in range(times):
                var tlc = 0  # tok_list_count
                for tok_list in test_input[:20]:
                    var iter_start = perf_counter_ns()
                    comptime if display:
                        print("raw tokens:", tok_list)
                    else:
                        showProgress(tlc, len(test_input))
                    var temp = List[Int](capacity=len(tok_list))
                    for j in range(len(tok_list)):
                        temp[j] = tok_list[j]
                    var decoded_input = tokenizer.decode(temp^)
                    var output = llm.forward(tok_list)
                    comptime if display:
                        print("decoded:", decoded_input)
                        tempPrintOutput(output)

                    var predicted_token = llm.getNextTokenGreedy()
                    var predicted_tokstr = tokenizer.decodeToken(
                        predicted_token
                    )
                    comptime if display:
                        print(
                            "predicted token:",
                            coloredString(predicted_tokstr),
                        )
                    # var tokens_as_tensor = LayoutTensor[
                    # var loss = crossEntropyLoss(output, test_input)
                    var iter_end = perf_counter_ns()
                    var iter_elapsed = Int(iter_end - iter_start)
                    var log_record = LLMInferenceRecord(
                        device,
                        0.69,
                        iter_elapsed,
                        predicted_tokstr,
                        0.69,
                        69,
                        String(ftype),
                    )
                    logger.log(log_record)
                    tlc += 1
                    output.ptr.free()
            var end = perf_counter_ns()
            print("time", (end - start) // 1_000, "us for", times, "runs")
            # print("Capacity {} offset {}".format(llm.arena.capacity, llm.arena.offset))
            print(
                "TODO: backward, logger, better GEMM, consider params to"
                " disable backwards buffers, biases for QKV ?, biases ffn,"
                " consider looking up what W_o is, transformer grad buffers,"
                " optimizer trait (step)"
            )

    except e:
        print(e, file=stderr)
