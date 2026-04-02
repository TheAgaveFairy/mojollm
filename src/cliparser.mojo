from std.sys import stderr, argv
from std.os import abort
from std.pathlib import Path
from std.subprocess import run as subProcessRun

# from subprocess import run as subprocessRun


struct TokenizerParser(Copyable, ImplicitlyCopyable, Movable, Writable):
    comptime FLAG_HELP = "--help"
    comptime FLAG_HELP_SHORT = Self.FLAG_HELP[byte=1:3]  # -h
    comptime FLAG_VOCAB = "--vocab-size"
    comptime FLAG_VOCAB_SHORT = Self.FLAG_VOCAB[byte=1:3]  # -v
    comptime FLAG_SAVE_NAME = "--save-name"
    comptime FLAG_SAVE_NAME_SHORT = Self.FLAG_SAVE_NAME[byte=1:3]  # -s
    comptime FLAG_INPUT_FILENAME = "--input-filename"
    comptime FLAG_INPUT_FILENAME_SHORT = Self.FLAG_INPUT_FILENAME[byte=1:3]  # -i
    comptime FLAG_TAG = "--tag"
    comptime FLAG_TAG_SHORT = Self.FLAG_TAG[byte=1:3]

    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    comptime DUMMY = "DUMMY"

    var vocab_size: Int
    var save_name: String
    var input_filename: String
    var tag: String
    var had_error: Bool

    def __init__(out self):
        var args = argv()
        # defaults
        self.vocab_size = 5000
        self.input_filename = "./datasets/shakespeare.txt"
        self.save_name = Self.DUMMY
        try:
            var date = subProcessRun("TZ=\"America/New_York\" date +'%d %Y'")
            self.tag = date
        except e:
            self.tag = "none"  # TODO: today's date
        self.had_error = False

        var i = 1  # skip argv[0] / program name
        while i < len(args):
            arg = args[i]
            if (
                arg == materialize[Self.FLAG_HELP]()
                or arg == materialize[Self.FLAG_HELP_SHORT]()
            ):
                Self.printHelp()
                self.had_error = True  # don't run anything
                # abort("HELP")
                return
            elif (
                arg == materialize[Self.FLAG_VOCAB]()
                or arg == materialize[Self.FLAG_VOCAB_SHORT]()
            ):
                if i + 1 < len(args):
                    self._parseVocabSize(args[i + 1])
                    i += 2  # consume
                else:
                    self.printBoundsError("-v")
                    break
            elif (
                arg == materialize[Self.FLAG_INPUT_FILENAME]()
                or arg == materialize[Self.FLAG_INPUT_FILENAME_SHORT]()
            ):
                if i + 1 < len(args):
                    self._parseInputFilename(args[i + 1])
                    i += 2
                else:
                    self.printBoundsError("-i")
                    break
            elif (
                arg == materialize[Self.FLAG_SAVE_NAME]()
                or arg == materialize[Self.FLAG_SAVE_NAME_SHORT]()
            ):
                if i + 1 < len(args):
                    self._parseSaveName(args[i + 1])
                    i += 2
                else:
                    self.printBoundsError("-s")
                    break
            elif (
                arg == materialize[Self.FLAG_TAG]()
                or arg == materialize[Self.FLAG_TAG_SHORT]()
            ):
                if i + 1 < len(args):
                    self._parseTag(args[i + 1])
                    i += 2
                else:
                    self.printBoundsError("-t")
                    break
            else:
                print("unknown flag: " + arg, file=stderr)
                i += 1
                self.had_error = True
        # have to wait for this
        if self.save_name == Self.DUMMY:
            var filename = Path(self.input_filename).name().split(".")[0]
            self.save_name = "./models/bpe_{}_{}.tok".format(
                self.vocab_size, filename
            )

    def printBoundsError(mut self, text: String):
        print(
            "Error in args. Please try again; probably caused by a flag at the"
            " end with no following value."
        )
        self.had_error = True

    @staticmethod
    def printHelp():
        var help_str = (
            "Usage: ./tokenizer [OPTIONS]...\n"
            + "\t-i, --input-filename FILENAME\ta custom input file. default ="
            " ./datasets/input.txt\n"
            + "\t-s, --save-name FILENAME\twhat name we will save the tokenizer"
            " to. default = bpe_VOCAB_SIZE.tok.\n"
            + "\t-v, --vocab_size VOCAB_SIZE\tdefault = 5000.\n"
            + "\t-t, --tag TAG\tdefault = today's date.\n"
        )
        print(help_str)

    def _parseTag(mut self, tag: StringSlice):
        if tag[byte=0] == "-":
            print("please include the input filename", file=stderr)
            self.had_error = True
        else:
            self.tag = String(tag)

    def _parseInputFilename(mut self, filename: StringSlice):
        if filename[byte=0] == "-":
            print("please include the input filename", file=stderr)
            self.had_error = True
        else:
            self.input_filename = String(filename)

    def _parseSaveName(mut self, filename: StringSlice):
        if filename[byte=0] == "-":
            print("please include the save filename", file=stderr)
            self.had_error = True
        else:
            self.save_name = String(filename)

    def _parseVocabSize(mut self, vocab_size: StringSlice):
        if vocab_size[byte=0] == "-":
            print("please include the save filename", file=stderr)
            self.had_error = True
        else:
            try:
                self.vocab_size = Int(vocab_size)
            except e:
                print(e, file=stderr)
                self.had_error = True

    def __str__(self) -> String:
        return (
            "CLIParser:"
            + "\nVocab Size: "
            + String(self.vocab_size)
            + "\nSave Name: "
            + self.save_name
            + "\nInput Filename: "
            + String(self.input_filename)
            + "\nHad Error: "
            + String(self.had_error)
        )

    def __repr__(self) -> String:
        return self.__str__()

    def write_to(self, mut writer: Some[Writer]):
        writer.write_string(self.__str__())


def main():
    var parser = TokenizerParser()
    print(parser)
