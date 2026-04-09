from std.sys.info import has_accelerator
from std.sys import stderr
from std.utils import Variant
from std.pathlib import Path
from std.reflection import (
    struct_field_count,
    struct_field_names,
    struct_field_types,
)  # , __struct_field_ref
from std.reflection.struct_fields import is_struct_type
from std.testing import assert_equal, assert_true, TestSuite
from std.subprocess import run as subprocessRun

# deprecated until Mojo 1.0
#import emberjson  # 3rd party dep, talks that it will be adopted by stdlib

from attention import ModelParams, ftype, sftype


def getDateTime() -> String:
    try:
        # var datetime = subprocessRun("TZ=\"America/New_York\" date +'%d %Y'")
        var datetime = subprocessRun("date")
        var parts = datetime.split()

        var day_of_week = parts[0]
        var month = parts[1]
        var day_num = parts[2]
        var time = parts[3]
        var ampm = parts[4]
        var timezone = parts[5]
        var year = parts[6]

        _ = day_of_week, ampm, timezone

        # we want "Day-Month-Year-time"
        return "{}-{}-{}-{}".format(day_num, month, year, time)

    except e:
        print(e, file=stderr)
        return "DATE_FAILURE"

def prependFilename(p: Path, prefix: String) -> Path:
    var parts = p.parts()
    var filename = parts[-1]
    var suffix = p.suffix()
    
    # stem is filename minus the suffix
    var stem = filename[byte=: len(filename) - len(suffix)]
    var new_filename = prefix + stem + suffix
    
    return Path("/".join(parts[:-1])) / new_filename

def listFields[T: AnyType]() -> List[String]:
    """Prints the field names of a struct as csv."""
    comptime names = struct_field_names[T]()
    var result = List[String](capacity = len(names))

    comptime for i in range(struct_field_count[T]()):
        #result += materialize[names[i]]() + ","
        result.append(materialize[names[i]]())
    return result^


def listFieldValues[T: AnyType](ref s: T) -> List[String]:
    """Takes in a reference to a live instance and prints its values as csv."""
    comptime N = struct_field_count[T]()
    comptime TTs = struct_field_types[T]()
    comptime names = struct_field_names[T]()

    var result = List[String](capacity = N)

    comptime for i in range(N):
        comptime TT = TTs[i]
        ref my_ref = __struct_field_ref(i, s)
        comptime if conforms_to(TT, Writable & ImplicitlyCopyable):
            var w = trait_downcast[Writable & ImplicitlyCopyable](my_ref)
            result.append(String(w))
        elif is_struct_type[TT](): # hopefully this won't ever get too deep in a recursion
            #var inner_struct_fields = listFieldValues[TT](rebind[ref TT](my_ref))
            #var temp = ",".join(inner_struct_fields)
            #temp = "[" + temp + "]"
            result.append("(Nested Struct - TODO)") # TODO: determine if we need
        else:
            #comptime if not conforms_to(TT, Writable):
            #    print("{} IS NOT Writable".format(materialize[names[i]]()))
            #comptime if not conforms_to(TT, ImplicitlyCopyable):
            #    print("{} IS NOT ImplicitlyCopyable".format(materialize[names[i]]()))
            result.append("{}-FAILED".format(materialize[names[i]]()))
    return result^


trait LogRecord:
    @staticmethod
    def header() -> List[String]:
        ...

    def toCSV(self) -> String:
        ...


@fieldwise_init
struct LogFormat:
    var value: Int
    comptime CSV = Self(0)
    comptime JSON = Self(1)


@fieldwise_init
struct LLMTrainRecord(LogRecord):#, emberjson.JsonSerializable):
    var device: String
    var step: Int  # nani
    var loss: sftype
    var perplexity: Float64
    var tokens_per_sec: Float64
    var forward_ns: Int
    var backward_ns: Int
    var learning_rate: sftype
    var ftype: String #DType

    @staticmethod
    def header() -> List[String]:
        return listFields[Self]()

    def toCSV(self) -> String:
        """Returns a csv string of the values. Header handled separately."""
        return ",".join(listFieldValues(self))
    #
    # def write_json(self, mut writer: Some[emberjson.Serializer]):
    #     var field_names = listFields[Self]()
    #     var field_values = listFieldValues(self)
    #     #var field_names = field_names_csv.split(",")
    #     #var field_values = field_values_csv.split(",")
    #
    #     if len(field_names) != len(field_values):
    #         writer.write("FAILURE TO WRITE AS JSON, LLMTrainRecord")
    #         print(len(field_names), field_names, len(field_values), field_values)
    #         return
    #
    #     var n = len(field_names)
    #     if not len(field_names[n - 1]):  # empty field
    #         _ = field_names.pop()
    #         _ = field_values.pop()
    #         n -= 1
    #
    #     var result = "{"
    #     for i in range(n):
    #         if not len(field_names[i]):
    #             continue
    #         result += (
    #             '"' + String(field_names[i]) + '": ' + String(field_values[i])
    #         )
    #         if i != n - 1:
    #             result += ", "
    #     result += "}"
    #     writer.write(result)


@fieldwise_init
struct LLMInferenceRecord(LogRecord):
    var device: String
    var tokens_per_sec: Float64
    var forward_ns: Int
    var output: String
    var temp: Float64
    var top_k: Int  # TODO: implement
    var ftype: String # DType

    @staticmethod
    def header() -> List[String]:
        return listFields[Self]()

    def toCSV(self) -> String:
        """Returns a csv string of the values. Header handled separately."""
        return ",".join(listFieldValues(self))


struct CSVLogger[T: LogRecord]:
    var filename: Path  # or do i store a file and deinit it later
    var mp_filename: Path

    comptime mp_prefix = "mp_" # for ModelParams

    def __init__(out self, filename: Variant[Path, String]) raises:
        # could also overload constructor
        if filename.isa[Path]():
            self.filename = filename[Path]
            self.mp_filename = prependFilename(filename[Path], Self.mp_prefix)
        else:
            self.filename = Path(filename[String])
            self.mp_filename = prependFilename(self.filename, Self.mp_prefix)
            # if not self.filename.exists():

        with open(self.filename, "w") as f:
            f.write(",".join(self.T.header()) + "\n")

        with open(self.mp_filename, "w") as f:
            f.write(",".join(listFields[ModelParams]()) + "\n")
            f.write(",".join(listFieldValues(ModelParams())))

    def log(self, record: self.T) raises:
        with open(self.filename, "a") as f:
            f.write(record.toCSV() + "\n")


def main() raises:
    var suite = TestSuite()
    # suite.test[reflectionPrintTest]()
    # suite.test[testsNotWrittenYet]()
    suite.test[serializeTest]()
    suite^.run()


def testsNotWrittenYet() raises:
    assert_true(False)


def serializeTest() raises:
    var device = "xPU"
    var rec_train = LLMTrainRecord(
        device,
        1,
        3.14,
        4.20,
        1337.0,
        9001,
        60000,
        0.01,
        String(ftype),  # ModelParams() goes before ftype
    )
    
    _ = """
    var train_json = emberjson.serialize(rec_train)
    # var infer_json = emberjson.serialize(rec_infer)
    print("TRAIN json test...\n", train_json)
    # print("INFER\n", infer_json)

    print("try ModelParams() serialize tests...")
    var mp = ModelParams()
    var mp_s = emberjson.serialize(mp)
    print("serialize\n", mp_s)
    var mp_pp = emberjson.to_string[pretty=True](mp_s)  # prints {"key":123}
    print("true prettyprint\n", mp_pp)
    var mp_pp_f = emberjson.to_string[pretty=False](mp_s)  # prints {"key":123}
    print("false prettyprint\n", mp_pp_f)
    """
    assert_true(False)

def reflectionPrintTest() raises:
    _ = """
        return "{},{},{},{},{},{},{},{},".format(
            self.device,
            self.step,
            self.loss,
            self.perplexity,
            self.tokens_per_sec,
            self.learning_rate,
            "self.model_params",
            self.ftype,
        )
        """
    assert_true(False)
