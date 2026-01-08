from sys import stderr, argv
#from subprocess import run as subprocessRun

struct TokenizerParser(Copyable, Movable, ImplicitlyCopyable, Writable, Representable):
    comptime FLAG_HELP = "--help"
    comptime FLAG_HELP_SHORT = Self.FLAG_HELP[1:3] # -h
    comptime FLAG_MODE = "--vocab-size"
    comptime FLAG_MODE_SHORT = Self.FLAG_MODE[1:3] # -v
    comptime FLAG_SAVE_NAME = "--save-name"
    comptime FLAG_SAVE_NAME_SHORT = Self.FLAG_SAVE_NAME[1:3]
    comptime FLAG_FILENAME = "--filename"
    comptime FLAG_FILENAME_SHORT = Self.FLAG_FILENAME[1:3] # -f

    var vocab_size = 256
    var save_name = "bpe.tok"
    var filename = "./datasets/input.txt" #: Optional[String] # for a file without the standard name
    var had_error: Bool

    fn __init__(out self):
        var args = argv()
        # defaults
        self.filename = type_of(self.filename)(None)
        self.runs = 1
        self.had_error = False
        var i = 1 # skip argv[0] / program name
        while i < len(args):
            arg = args[i]
            if arg == materialize[Self.FLAG_HELP]() or arg == materialize[Self.FLAG_HELP_SHORT]():
                Self.printHelp()
                self.had_error = True # don't run anything
                break
            elif arg == materialize[Self.FLAG_MODE]() or arg == materialize[Self.FLAG_MODE_SHORT]():
                self._parseMode(args[i + 1])
                i += 2 # consume
            elif arg == materialize[Self.FLAG_FILENAME]() or arg == materialize[Self.FLAG_FILENAME_SHORT]():
                self._parseFilename(args[i + 1])
                i += 2
            elif arg == materialize[Self.FLAG_DAYS]() or arg == materialize[Self.FLAG_DAYS_SHORT]():
                self._parseDays(args[i + 1])
                i += 2
            elif arg == materialize[Self.FLAG_RUNS]() or arg == materialize[Self.FLAG_RUNS_SHORT]():
                self._parseRuns(args[i + 1])
                i += 2
            else:
                print("unknown flag: " + arg, file = stderr)
                i += 1
                self.had_error = True

    @staticmethod
    fn printHelp():
        var help_str = "Usage: main [OPTIONS]...\n" + \
                "\t-m, --mode MODE\tMODE is 'full', 'test', or 'both'. case-insensitive. default = 'full'\n" + \
                "\t-f, --filename FILENAME\ta custom input file for use with a single day. default = None and uses 'mode' implication\n" + \
                "\t-d, --days DAYS\twhere DAYS is csv. e.g. '1', '1,2,3', '7,6,9'. defaults to AoC schedule using 'date' during event or all days otherwise\n" + \
                "\t-r, --runs RUNS\tthe number of runs to run for benchmarking. default = 1.\n" + \
                "\n\tEXAMPLE: 'main -m both -d 7,8 -r 100' would run both test and full inputs for days 7 and 8 100 times each (400 total runs)"
        print(help_str)

    fn _parseFilename(mut self, filename: StringSlice):
        if filename[:2] == "--":
            print("please include the filename", file = stderr)
            self.had_error = True
        else:
            self.filename = String(filename)

    fn __copyinit__(out self, other: Self):
        self.days = other.days.copy()
        self.mode = other.mode
        self.filename = other.filename
        self.runs = other.runs
        self.had_error = other.had_error

    fn __moveinit__(out self, deinit existing: Self):
        self.days = existing.days^
        self.mode = existing.mode^
        self.filename = existing.filename^
        self.runs = existing.runs
        self.had_error = existing.had_error

    fn __str__(self) -> String:
        return "CLIParser:" + \
                "\nDays: " + String(self.days) + \
                "\nMode: " + self.mode + \
                "\nFilename: " + String(self.filename) + \
                "\nRuns: " + String(self.runs) + \
                "\nHad Error: " + String(self.had_error)
    fn __repr__(self) -> String:
        return self.__str__()
    fn write_to(self, mut writer: Some[Writer]):
        writer.write_bytes(self.__str__().as_bytes())

fn main():
    var parser = CLIParser()
    print(parser)
