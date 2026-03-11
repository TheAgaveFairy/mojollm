from std.algorithm.functional import vectorize, parallelize
from layout import Layout, LayoutTensor
from std.math import tanh, exp, sqrt, erf, log, pi, tau
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
)

from attention import ftype, sftype, nelts
from helpers import compareBuffers


# comptime activation_fn = fn(sftype) -> sftype # if was scalar
trait ActivationFunction:
    @staticmethod
    @always_inline("nodebug")
    fn forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        """Operates in-place."""
        ...

    @staticmethod
    @always_inline("nodebug")
    fn backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_input: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """Operates in-place on pre-filled d_input."""
        ...


struct ReLU(ActivationFunction):
    @staticmethod
    @always_inline("nodebug")
    fn forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        @parameter
        fn vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            comptime zeros = SIMD[ftype, width](0)
            var mask = nums.gt(zeros)
            var relu = mask.select(nums, zeros)
            x.ptr.store[width=width](i, relu)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    fn backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_input: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        SCALAR FORM.
        return d_output if x > 0.0 else 0.0
        SIMD enhanced.
        """
        comptime size = layout.size()

        fn closure[width: Int](i: Int) unified {mut}:
            comptime zeros = SIMD[ftype, width](0.0)
            var vec = d_input.ptr.load[width](i)
            var mask = vec.gt(zeros)
            var res = mask.select(vec, zeros)
            d_input.ptr.store[width](i, res)

        vectorize[nelts](size, closure)


struct GELU(ActivationFunction):
    """
    Exact implementation.
    GELU(x) = x * CDF(x) = x * (1 + erf(x / sqrt(2))) / 2
    Can be approximated with a tanh version, or quick version (see GELU paper).
    """

    @staticmethod
    @always_inline("nodebug")
    fn forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        comptime sqrt2 = sqrt(2.0)

        @parameter
        fn vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            # var nums_cubed = nums * nums * nums
            # comptime scaling = SIMD[ftype, width](0.44715)
            # comptime term = sftype(sqrt(2 / pi))
            # var gelu = (
            #    nums / 2 * (1 + tanh(term * (nums + scaling * nums_cubed)))
            # )
            comptime sqrt2_vec = SIMD[ftype, width](sqrt2)
            var gelu = 0.5 * nums * (1 + erf(nums / sqrt2_vec))
            x.ptr.store[width=width](i, gelu)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    fn backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout, _],
        d_input: LayoutTensor[
            ftype, layout, MutAnyOrigin
        ],  # could split off "d_output _as_ upstream"
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
        fn vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            comptime sqrt2_vec = SIMD[ftype, width](sqrt2)
            comptime sqrttau_vec = SIMD[ftype, width](sqrttau)
            var cdf = 0.5 * (1 + erf(nums / sqrt2_vec))
            var pdf = exp(
                -0.5 * nums * nums - log(sqrttau_vec)
            )  # or exp(-0.5 * x**2) / sqrt(tau)

            var upstream = d_input.ptr.load[width=width](i)
            var answer = upstream * (cdf + nums * pdf)
            d_input.ptr.store[width=width](i, answer)

        vectorize[nelts](comptime (layout.size()), vectorize_closure)


def main() raises:
    var suite = TestSuite()
    suite.test[ReLUTest]()
    # suite.test[AnotherTest]()
    suite.test[GELUTorchForwardTest]()
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

    fn numerical_deriv(x: sftype, h: sftype = 1e-5) -> sftype:
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
    var grad_t = ScalarTensor.stack_allocation().fill(1.0)
    GELU.backward(t, grad_t)

    var analytic_dx = grad_t[0]
    var numeric_dx = numerical_deriv(x)
    assert_almost_equal(analytic_dx, numeric_dx, "almost equal", atol=1e-4)


def GELUTorchForwardTest() raises:
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

    var ref_fwd = TestTensor.stack_allocation()
    var my_fwd = TestTensor.stack_allocation()

    comptime for i in range(N):
        ref_fwd[i] = materialize[expected_forward[i]]()
        my_fwd[i] = materialize[xs[i]]()

    GELU.forward(my_fwd)
    # print(my_fwd)
    # print(ref_fwd)

    comptime for i in range(N):
        assert_almost_equal(my_fwd[i], ref_fwd[i], atol=1e-15, rtol=1e-5)
