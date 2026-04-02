from std.sys import stderr, is_big_endian, argv
import std.os
from std.time import perf_counter_ns
from std.testing import TestSuite, assert_equal
from std.collections import Set

# from python import Python # for RegExp engines

from helpers import (
    showProgress,
    cleanFunctionName,
    compareBuffers,
    fillTensorRand,
)
from attention import ftype, sftype, nelts
from cliparser import TokenizerParser
from tokenizer.tokenchunks import TokenChunks
from tokenizer.tokenizer import *


comptime desired_vocab_size = 260 # train 4 new tokens
comptime train_filename = "./model.mojo" # train on self, why not
comptime temp_filename = "./temp.tok"

def main() raises:
    var suite = TestSuite()
    suite.test[saveLoadTest]()
    suite.test[decodeEncodeTest]()
    suite.test[pairReprTest]()
    suite^.run()


def saveLoadTest() raises:
    var tok = Tokenizer(desired_vocab_size)
    with open(train_filename, "r") as f:
        var text = f.read()
        tok.train(text)
        tok.save(temp_filename)
        var tok2 = Tokenizer.load(temp_filename)
        #assert_equal(tok, tok2)
        assert_equal(compareVocabsTest(tok, tok2), True)


def decodeEncodeTest() raises:
    var tok = Tokenizer(desired_vocab_size)
    with open(train_filename, "r") as f:
        var text = f.read()
        tok.train(text)
        assert_equal(text, tok.decode(tok.encode(text)))

def pairReprTest() raises:
    var p = Pair(7, 9)
    assert_equal(p, Pair.fromRepr(String(p)))
