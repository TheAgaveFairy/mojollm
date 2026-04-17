from std.algorithm.functional import vectorize, parallelize
from layout import Layout, LayoutTensor
from std.math import tanh, exp, sqrt, erf, log, pi, tau
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
)
from std.random import seed, randn

from attention import ftype, sftype, nelts
from helpers import compareBuffers, fillTensorRand  # testing


# comptime activation_fn = fn(sftype) -> sftype # if was scalar
trait ActivationFunction:
    @staticmethod
    @always_inline("nodebug")
    def forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        """Operates in-place."""
        ...

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """Takes the original forward input 'x',
        the upstream gradient, d_output,
        and calculates the d_z gradient as our output for pre-act.
        """
        ...


struct ReLU(ActivationFunction):
    @staticmethod
    @always_inline("nodebug")
    def forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        @parameter
        def vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            comptime zeros = SIMD[ftype, width](0)
            var mask = nums.gt(zeros)
            var relu = mask.select(nums, zeros)
            x.ptr.store[width=width](i, relu)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        SCALAR FORM is "return d_output if x > 0.0 else 0.0".
        SIMD enhanced.
        """

        def closure[width: Int](i: Int) unified {mut}:
            comptime zeros = SIMD[ftype, width](0.0)
            var vec = d_output.ptr.load[width](i)
            var mask = vec.gt(zeros)
            var res = mask.select(vec, zeros)
            d_z.ptr.store[width](i, res)

        vectorize[nelts](comptime (layout.size()), closure)


struct GELU(ActivationFunction):
    """
    Exact implementation.
    GELU(x) = x * CDF(x) = x * (1 + erf(x / sqrt(2))) / 2
    Can be approximated with a tanh version, or quick version (see GELU paper).
    """

    @staticmethod
    @always_inline("nodebug")
    def forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        comptime sqrt2 = sqrt(2.0)

        @parameter
        def vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            # var nums_cubed = nums * nums * nums
            # comptime scaling = SIMD[ftype, width](0.44715)
            # comptime term = sftype(sqrt(2 / pi))
            # var gelu = (
            #    nums / 2 * (1 + tanh(term * (nums + scaling * nums_cubed)))
            # )
            comptime sqrt2_vec = SIMD[ftype, width](sqrt2)
            comptime halves = SIMD[ftype, width](0.5)
            comptime ones = SIMD[ftype, width](1.0)
            var gelu = halves * nums * (ones + erf(nums / sqrt2_vec))
            x.ptr.store[width=width](i, gelu)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        (x * CDF(x))' = x'CDF(x) + xCDF'(x) .
                      = CDF(x) + xPDF(x)    .

        PyTorch:
        pdf_val = torch.distributions.Normal(0, 1).log_prob(data).exp() .
        return grad_output * (cdf + data * pdf_val)                     .

        Modifies d_input in-place, assuming it was already loaded with d_output.
        This approach is less explicit and more error prone, but a touch faster.
        """
        comptime sqrt2 = sqrt(2.0)
        comptime sqrttau = sqrt(tau)  # math.pi * 2.0

        @parameter
        def vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            comptime sqrt2_vec = SIMD[ftype, width](sqrt2)
            comptime sqrttau_vec = SIMD[ftype, width](sqrttau)
            comptime term = log(sqrttau_vec)
            comptime halves = SIMD[ftype, width](0.5)
            comptime neg_halves = SIMD[ftype, width](-0.5)
            comptime ones = SIMD[ftype, width](1.0)
            comptime inverse_sqrttau = SIMD[ftype, width](1 / sqrttau)
            var cdf = halves * (ones + erf(nums / sqrt2_vec))

            # var pdf = exp(
            #    neg_halves * nums * nums - term
            # )  # or exp(-0.5 * x**2) / sqrt(tau)
            var pdf = exp(neg_halves * nums * nums) * inverse_sqrttau

            var upstream = d_output.ptr.load[width=width](i)
            var answer = upstream * (cdf + nums * pdf)
            d_z.ptr.store[width=width](i, answer)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)


struct GELUTanh(ActivationFunction):
    """
    Tanh approximation.
    GELUTanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    @staticmethod
    @always_inline("nodebug")
    def forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        comptime term = sftype(sqrt(2.0 / pi))

        @parameter
        def vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            var nums_cubed = nums * nums * nums
            comptime scaling = SIMD[ftype, width](0.044715)
            comptime terms = SIMD[ftype, width](term)
            comptime halves = SIMD[ftype, width](0.5)
            comptime ones = SIMD[ftype, width](1.0)
            var gelu = (
                halves
                * nums
                * (ones + tanh(terms * (nums + scaling * nums_cubed)))
            )
            x.ptr.store[width=width](i, gelu)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        k = sqrt(2 / pi)
        c = 0.044715
        t = tanh(z) = tanh(k * (x + c * x^3))

        dy/dx = 0.5 * ((1 + t) + x * (1 - t^2) * k * (1 + 3 * c * x^2))
        """
        comptime k = sqrt(2.0 / pi)
        comptime c = 0.044715

        @parameter
        def vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            comptime ks = SIMD[ftype, width](k)
            comptime cs = SIMD[ftype, width](c)
            comptime threecs = SIMD[ftype, width](3.0 * c)
            comptime ones = SIMD[ftype, width](1.0)
            comptime halves = SIMD[ftype, width](0.5)

            var ts = tanh(ks * (nums + cs * (nums * nums * nums)))
            var deriv = halves * (
                (ones + ts)
                + nums
                * (ones - (ts * ts))
                * ks
                * (ones + threecs * nums * nums)
            )
            var upstream = d_output.ptr.load[width=width](i)
            var answer = upstream * deriv
            d_z.ptr.store[width=width](i, answer)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)


struct GELUFast(ActivationFunction):
    """
    FAST implementation. NOT exact - use GELU. Tanh is a TODO.
    GELUFast(x) = x * sigmoid(1.702 * x) https://arxiv.org/pdf/1606.08415
    """

    @staticmethod
    @always_inline("nodebug")
    def _sigmoid[
        stype: DType, width: Int
    ](x: SIMD[stype, width]) -> SIMD[stype, width]:
        """SIMD Sigmoid that accepts any floating point type, not just ftype."""
        comptime assert (
            stype.is_floating_point()
        ), "_sigmoid requires floating points"
        comptime ones = SIMD[stype, width](1.0)
        comptime neg_ones = SIMD[stype, width](-1.0)
        var input = x * neg_ones
        return ones / (ones + exp(input))

    @staticmethod
    @always_inline("nodebug")
    def forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        @parameter
        def vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            comptime scaling = SIMD[ftype, width](1.702)
            var gelu = nums * Self._sigmoid(scaling * nums)
            x.ptr.store[width=width](i, gelu)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    def backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_output: LayoutTensor[ftype, layout, _],
        d_z: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        f'(1.702x) = sigmoid(1.702x) + (x * 1.702 * sigmoid(1.702x) * (1 - sigmoid(1.702x)))
        Modifies d_input in-place, assuming it was already loaded with d_output.
        This approach is less explicit and more error prone, but a touch faster.
        """
        comptime alpha = sftype(1.702)

        @parameter
        def vectorize_closure[width: Int](i: Int) unified {mut}:
            comptime alphas = SIMD[ftype, width](alpha)
            comptime ones = SIMD[ftype, width](1.0)
            var nums = x.ptr.load[width=width](i)
            var s = Self._sigmoid(nums * alphas)
            var deriv = s + alphas * nums * s * (ones - s)
            var upstream = d_output.ptr.load[width=width](i)
            var answer = upstream * deriv
            d_z.ptr.store[width=width](i, answer)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)


def main() raises:
    var suite = TestSuite()
    suite.test[ReLUTest]()
    #suite.test[AnotherTest]()
    suite.test[testApproximations]()
    suite.test[GELUTorchTest]()
    suite^.run()


def ReLUTest() raises:
    """Ideally we'd test against torch or something, etc."""
    comptime layout = Layout.row_major(2, 3)
    comptime N = layout.size()

    var xs: List[sftype] = [-100.5, -2.0, -1.0, 0.0, 1.0, 2.0]
    var expected: List[sftype] = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]

    var x = LayoutTensor[ftype, layout, MutAnyOrigin].stack_allocation()
    var y = LayoutTensor[ftype, layout, MutAnyOrigin].stack_allocation()
    var n = sftype(N)
    comptime for i in range(N):
        # var val = sftype(i) - (n / 2)
        # x.ptr[i] = val
        # y.ptr[i] = val if val > 0.0 else 0.0
        x.ptr[i] = xs[i]
        y.ptr[i] = expected[i]
    ReLU.forward(x)

    assert_true(compareBuffers(x.ptr, y.ptr, N))


def AnotherTest() raises:
    comptime ScalarTensor = LayoutTensor[
        ftype, Layout.row_major(1), MutAnyOrigin
    ]

    def numerical_deriv(x: sftype, h: sftype = 1e-5) -> sftype:
        """LLM idea for testing."""
        var plus = ScalarTensor.stack_allocation().fill(x + h)
        var minus = ScalarTensor.stack_allocation().fill(x - h)
        GELU.forward(plus)
        GELU.forward(minus)
        return rebind[sftype](plus[0] - minus[0]) / rebind[sftype](2 * h)

    var x: sftype = 0.42
    var t = ScalarTensor.stack_allocation().fill(
        x
    )  # pretend we GELU this forward
    var d_output = ScalarTensor.stack_allocation().fill(1.0)
    var d_z = ScalarTensor.stack_allocation().fill(0.0)
    GELU.backward(t, d_output, d_z)

    var analytic_dx = d_z[0]
    var numeric_dx = numerical_deriv(x)
    assert_almost_equal(analytic_dx, numeric_dx, "almost equal", atol=1e-4)


def GELUTorchTest() raises:
    """Used torch to get some reference values, see utils."""
    comptime xs: List[sftype] = [
        -4.0,
        -2.5,
        -1.5,
        -1.0,
        -0.5,
        -0.1,
        0.0,
        0.1,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
    ]
    comptime N = len(xs)
    comptime TestTensor = LayoutTensor[ftype, Layout.row_major(N), MutAnyOrigin]

    comptime dys: List[sftype] = [1.0, 0.5, -1.0, 2.0, 0.0, 1.0, 1.0]

    comptime expected_forward: List[sftype] = [
        -0.0001267195,
        -0.0155241787,
        -0.1002108604,
        -0.1586552858,
        -0.1542687863,
        -0.0460172109,
        0.0000000000,
        0.0539827906,
        0.3457311988,
        0.8413447142,
        1.3997890949,
        1.9544999599,
        2.4844758511,
        2.9959502220,
        3.9998731613,
        4.9999985695,
    ]  #  torch.nn.functional.gelu(xs as a tensor, approx = 'none')


    comptime expected_backward: List[sftype] = [
        -0.0005036294, -0.0188055336, 0.1274691522, -0.1666308641,
        0.00000000, 0.4204768240, 0.500000, 0.5795231462,
        0.4337475300, -1.0833153725, 2.2549383640, 0.00000000,
        1.0376110077, 1.0119456053, 1.0005036592, 0.5000035763,
    ]

    var ref_fwd = TestTensor.stack_allocation()
    var my_fwd = TestTensor.stack_allocation()

    var x = TestTensor.stack_allocation()
    var d_output = TestTensor.stack_allocation()
    var d_z = TestTensor.stack_allocation().fill(0.0)
    var expected = TestTensor.stack_allocation()


    comptime for i in range(N):
        # fwd
        ref_fwd[i] = materialize[expected_forward[i]]()
        my_fwd[i] = materialize[xs[i]]()
        # back
        x[i] = materialize[xs[i]]()
        d_output[i] = materialize[dys[i]]()
        expected[i] = materialize[expected_backward[i]]()

    GELU.forward(my_fwd) # modifies in-place
    GELU.backward(x, d_output, d_z)
    # print(my_fwd)
    # print(ref_fwd)

    comptime for i in range(N):
        assert_almost_equal(my_fwd[i], ref_fwd[i], atol=1e-5, rtol=1e-5)
        assert_almost_equal(d_z[i], expected[i], atol=1e-5, rtol=1e-5)


def testApproximations() raises:
    comptime xs: List[sftype] = [
        -3.0,
        -2.0,
        -1.0,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        3.0,
    ]
    comptime N = len(xs)
    comptime TestTensor = LayoutTensor[ftype, Layout.row_major(N), MutAnyOrigin]

    var exact = TestTensor.stack_allocation()
    var tanh = TestTensor.stack_allocation()
    var sigmoid = TestTensor.stack_allocation()

    comptime for i in range(N):
        exact[i] = materialize[xs[i]]()
        tanh[i] = materialize[xs[i]]()
        sigmoid[i] = materialize[xs[i]]()

    GELU.forward(exact)
    GELUTanh.forward(tanh)
    GELUFast.forward(sigmoid)

    print("Forward exact, tanh, sigmoid:")
    comptime for i in range(N):
        print(exact[i], "\t", tanh[i], "\t", sigmoid[i])
    print(exact)
    print(tanh)
    print(sigmoid)


def benchmarkGELU():
    seed(9001)
    # fillTensorRand()
    comptime N = 1000
    comptime TestTensor = LayoutTensor[ftype, Layout.row_major(N), MutAnyOrigin]

    var exact = TestTensor.stack_allocation()
    var tanh = TestTensor.stack_allocation()
    var sigmoid = TestTensor.stack_allocation()

    fillTensorRand(exact, 1)
    fillTensorRand(tanh, 1)
    fillTensorRand(sigmoid, 1)

    GELU.forward(exact)
    GELUTanh.forward(tanh)
    GELUFast.forward(sigmoid)
