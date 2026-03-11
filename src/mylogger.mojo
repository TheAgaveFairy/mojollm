from std.sys.info import has_accelerator
from std.sys import stderr
from std.utils import Variant
from std.pathlib import Path
from std.reflection import (
    struct_field_count,
    struct_field_names,
    struct_field_types,
)  # , __struct_field_ref
from std.testing import assert_equal, assert_true, TestSuite
from std.subprocess import run as subprocessRun

from attention import ModelParams, ftype, sftype


fn getDateTime() -> String:
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


fn printFieldsHeader[T: AnyType]() -> String:
    """Prints the field names of a struct as csv."""
    var result = ""
    comptime names = struct_field_names[T]()

    # @parameter
    comptime for i in range(struct_field_count[T]()):
        result += materialize[names[i]]() + ","
    return result


fn printAllFields[T: AnyType](ref s: T) -> String:
    """Takes in a reference to a live instance and prints its values as csv."""
    comptime TTs = struct_field_types[T]()
    comptime names = struct_field_names[T]()

    var result = ""

    # @parameter
    comptime for i in range(struct_field_count[T]()):
        comptime TT = TTs[i]
        var my_ref = __struct_field_ref(i, s)
        # @parameter
        comptime if conforms_to(TT, Writable & ImplicitlyCopyable):
            var w = trait_downcast[Writable & ImplicitlyCopyable](my_ref)
            result += "{},".format(w)
        else:
            result += "{}-FAILED,".format(materialize[names[i]]())
    return result


trait LogRecord:
    @staticmethod
    fn header() -> String:
        ...

    fn toCSV(self) -> String:
        ...


@fieldwise_init
struct LogFormat:
    var value: Int
    comptime CSV = Self(0)
    comptime JSON = Self(1)


@fieldwise_init
struct LLMTrainRecord(LogRecord):
    var device: String
    var step: Int  # nani
    var loss: sftype
    var perplexity: Float64
    var tokens_per_sec: Float64
    var forward_ns: Int
    var backward_ns: Int
    var learning_rate: sftype
    var model_params: ModelParams
    var ftype: DType

    @staticmethod
    fn header() -> String:
        return printFieldsHeader[Self]()

    fn toCSV(self) -> String:
        """Returns a csv string of the values. Header handled separately."""
        return printAllFields(self)


@fieldwise_init
struct LLMInferenceRecord(LogRecord):
    var device: String
    var tokens_per_sec: Float64
    var forward_ns: Int
    var output: String
    var temp: Float64
    var top_k: Int  # TODO: implement
    var model_params: ModelParams
    var ftype: DType

    @staticmethod
    fn header() -> String:
        return printFieldsHeader[Self]()

    fn toCSV(self) -> String:
        """Returns a csv string of the values. Header handled separately."""
        return printAllFields(self)


struct CSVLogger[T: LogRecord]:
    var filename: Path  # or do i store a file and deinit it later

    fn __init__(out self, filename: Variant[Path, String]) raises:
        # could also overload constructor
        if filename.isa[Path]():
            self.filename = filename[Path]
        else:
            self.filename = Path(filename[String])
            # if not self.filename.exists():

        with open(self.filename, "w") as f:
            f.write(self.T.header() + "\n")

    fn log(self, record: self.T) raises:
        with open(self.filename, "a") as f:
            f.write(record.toCSV() + "\n")


def main():
    var device = "gpu" if has_accelerator() else "cpu"
    var rec_train = LLMTrainRecord(
        device, 1, 3.14, 4.20, 1337.0, 9001, 60000, 0.01, ModelParams(), ftype
    )
    var rec_infer = LLMInferenceRecord(
        device, 5.0, 1337, "mojo", 0.05, 5, ModelParams(), ftype
    )
    # print(rec_train.header())
    # print(printAllFields(rec_train))
    # print(rec_infer.header())
    # print(printAllFields(rec_infer))

    try:
        var csv_train_logger = CSVLogger[LLMTrainRecord]("logs/train.csv")
        csv_train_logger.log(rec_train)
    except e:
        print(e, file=stderr)

    var suite = TestSuite()
    suite.test[reflectionPrintTest]()
    suite^.run()


def reflectionPrintTest():
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
