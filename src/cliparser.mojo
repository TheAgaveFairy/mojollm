from sys import stderr, argv
#from subprocess import run as subprocessRun

struct TokenizerParser(Copyable, Movable, ImplicitlyCopyable, Writable, Representable):
    comptime FLAG_HELP = "--help"
    comptime FLAG_HELP_SHORT = Self.FLAG_HELP[1:3] # -h
    comptime FLAG_VOCAB = "--vocab-size"
    comptime FLAG_VOCAB_SHORT = Self.FLAG_VOCAB[1:3] # -v
    comptime FLAG_SAVE_NAME = "--save-name"
    comptime FLAG_SAVE_NAME_SHORT = Self.FLAG_SAVE_NAME[1:3] # -s
    comptime FLAG_INPUT_FILENAME = "--input-filename"
    comptime FLAG_INPUT_FILENAME_SHORT = Self.FLAG_INPUT_FILENAME[1:3] # -i

    comptime __copyinit__is_trivial = True
    comptime __moveinit__is_trivial = True

    comptime DUMMY = "DUMMY"

    var vocab_size: Int
    var save_name: String
    var input_filename: String
    var had_error: Bool

    fn __init__(out self):
        var args = argv()
        # defaults
        self.vocab_size = 5000
        self.input_filename = "./datasets/input.txt"
        self.save_name = Self.DUMMY
        self.had_error = False

        var i = 1 # skip argv[0] / program name
        while i < len(args):
            arg = args[i]
            if arg == materialize[Self.FLAG_HELP]() or arg == materialize[Self.FLAG_HELP_SHORT]():
                Self.printHelp()
                self.had_error = True # don't run anything
                break
            elif arg == materialize[Self.FLAG_VOCAB]() or arg == materialize[Self.FLAG_VOCAB_SHORT]():
                if i + 1 < len(args):
                    self._parseVocabSize(args[i + 1])
                    i += 2 # consume
                else:
                    self.printBoundsError("-v")
                    break
            elif arg == materialize[Self.FLAG_INPUT_FILENAME]() or arg == materialize[Self.FLAG_INPUT_FILENAME_SHORT]():
                if i + 1 < len(args):
                    self._parseInputFilename(args[i + 1])
                    i += 2
                else:
                    self.printBoundsError("-i")
                    break
            elif arg == materialize[Self.FLAG_SAVE_NAME]() or arg == materialize[Self.FLAG_SAVE_NAME_SHORT]():
                if i + 1 < len(args):
                    self._parseSaveName(args[i + 1])
                    i += 2
                else:
                    self.printBoundsError("-s")
                    break
            else:
                print("unknown flag: " + arg, file = stderr)
                i += 1
                self.had_error = True
        # have to wait for this
        if self.save_name == Self.DUMMY:
            self.save_name = "./models/bpe_{}.tok".format(self.vocab_size)

    fn printBoundsError(mut self, text: String):
        print("Error in args. Please try again; probably caused by a flag at the end with no following value.")
        self.had_error = True

    @staticmethod
    fn printHelp():
        var help_str = "Usage: ./tokenizer [OPTIONS]...\n" + \
                "\t-i, --input-filename FILENAME\ta custom input file. default = ./datasets/input.txt\n" + \
                "\t-s, --save-name FILENAME\twhat name we will save the tokenizer to. default = bpe_VOCAB_SIZE.tok.\n" + \
                "\t-v, --vocab_size VOCAB_SIZE\tdefault = 5000.\n"
        print(help_str)

    fn _parseInputFilename(mut self, filename: StringSlice):
        if filename[:1] == "-":
            print("please include the input filename", file = stderr)
            self.had_error = True
        else:
            self.input_filename = String(filename)

    fn _parseSaveName(mut self, filename: StringSlice):
        if filename[:1] == "-":
            print("please include the save filename", file = stderr)
            self.had_error = True
        else:
            self.save_name = String(filename)

    fn _parseVocabSize(mut self, vocab_size: StringSlice):
        if vocab_size[:1] == "-":
            print("please include the save filename", file = stderr)
            self.had_error = True
        else:
            try:
                self.vocab_size = Int(vocab_size)
            except e:
                print(e, file = stderr)
                self.had_error = True

    fn __str__(self) -> String:
        return "CLIParser:" + \
                "\nVocab Size: " + String(self.vocab_size) + \
                "\nSave Name: " + self.save_name + \
                "\nInput Filename: " + String(self.input_filename) + \
                "\nHad Error: " + String(self.had_error)
    fn __repr__(self) -> String:
        return self.__str__()
    fn write_to(self, mut writer: Some[Writer]):
        writer.write_bytes(self.__str__().as_bytes())

fn main():
    var parser = TokenizerParser()
    print(parser)
