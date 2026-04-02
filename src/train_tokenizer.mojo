from tokenizer import *
from cliparser import TokenizerParser


def main():
    var config = TokenizerParser()
    if config.had_error:
        return

    try:
        with open(config.input_filename, "r") as f:
            var text = f.read()

            var tokenizer = Tokenizer(
                config.vocab_size
            )  # ~280 is enough to display recursion
            tokenizer.train(text, regex = True, debug_display = True)
            # print(showExample(tokenizer, tokenizer.encode(text[:500])))
            tokenizer.save(config.save_name)

            # if you want to use the visualizer.html
            var vocab_filename = "models/vocab_{}.txt".format(config.vocab_size)
            tokenizer.exportVocab(vocab_filename)

    except e:
        print(e)

