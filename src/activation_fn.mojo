from algorithm.functional import vectorize, parallelize
from layout import Layout, LayoutTensor
from math import tanh, sqrt, pi

from attention import ftype, sftype, nelts


# comptime activation_fn = fn(sftype) -> sftype
trait ActivationFunction:  # for a 2D LayoutTensor
    @staticmethod
    @always_inline("nodebug")
    fn forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        ...

    @staticmethod
    @always_inline("nodebug")
    fn backward[
        layout: Layout
    ](
        x: LayoutTensor[ftype, layout],
        d_output: LayoutTensor[ftype, layout],
        d_input: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
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
        x: LayoutTensor[ftype, layout],
        d_output: LayoutTensor[ftype, layout],
        d_input: LayoutTensor[ftype, layout, MutAnyOrigin],
    ):
        """
        SCALAR FORM.
        return d_output if x > 0.0 else 0.0
        """
        comptime size = layout.size()
        _ = """
        for i in range(size):
            if x.ptr[i] > 0:
                d_input.ptr[i] = d_output.ptr[i]
            else:
                d_input.ptr[i] = 0.0
        """

        # Vectorized implementation
        fn closure[width: Int](i: Int) unified {mut}:
            comptime zeros = SIMD[ftype, width](0.0)
            var vec = d_input.ptr.load[width](i)
            var mask = vec.gt(zeros)
            var res = mask.select(vec, zeros)
            d_input.ptr.store[width](i, res)

        vectorize[nelts](size, closure)


struct GELU(ActivationFunction):
    @staticmethod
    @always_inline("nodebug")
    fn forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        @parameter
        fn vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width=width](i)
            var nums_cubed = nums * nums * nums
            comptime scaling = SIMD[ftype, width](0.44715)
            comptime term = sqrt(2 / pi)
            var gelu = (
                nums / 2 * (1 + tanh(term * (nums + scaling * nums_cubed)))
            )
            x.ptr.store[width=width](i, gelu)

        vectorize[nelts](layout.size(), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    fn backward(x: sftype, grad_output: sftype) -> sftype:
        """
        SCALAR FORM.
        """
        # TODO: update to accept LayoutTensor[]s
        return grad_output if x > 0.0 else 0.0
