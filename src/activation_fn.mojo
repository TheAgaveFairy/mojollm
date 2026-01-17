from algorithm.functional import vectorize, parallelize
from layout import Layout, LayoutTensor
from math import tanh, sqrt, pi

from attention import ftype, sftype, nelts

#comptime activation_fn = fn(sftype) -> sftype
trait ActivationFunction: # for a 2D LayoutTensor

    @staticmethod
    @always_inline("nodebug")
    fn forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        ...
    @staticmethod
    @always_inline("nodebug")
    fn backward(x: sftype, grad_output: sftype) -> sftype:
        ...

struct ReLU(ActivationFunction):
    @staticmethod
    @always_inline("nodebug")
    fn forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        @parameter
        fn vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width = width](i)
            comptime zeros = SIMD[ftype, width](0)
            var mask = nums.gt(zeros)
            var relu = mask.select(nums, zeros)
            x.ptr.store[width = width](i, relu)
        vectorize[nelts](layout.size(), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    fn backward(x: sftype, grad_output: sftype) -> sftype:
        """
        SCALAR FORM.
        """
        # TODO: update to accept LayoutTensor[]s
        return grad_output if x > 0.0 else 0.0


struct GELU(ActivationFunction):
    @staticmethod
    @always_inline("nodebug")
    fn forward[layout: Layout](x: LayoutTensor[ftype, layout, MutAnyOrigin]):
        @parameter
        fn vectorize_closure[width: Int](i: Int) unified {mut}:
            var nums = x.ptr.load[width = width](i)
            var nums_cubed = nums * nums * nums
            comptime scaling = SIMD[ftype, width](0.44715)
            comptime term = sqrt(2 / pi)
            var gelu = nums / 2 * (1 + tanh(term * (nums + scaling * nums_cubed)))
            x.ptr.store[width = width](i, gelu)
        vectorize[nelts](layout.size(), vectorize_closure)


    @staticmethod
    @always_inline("nodebug")
    fn backward(x: sftype, grad_output: sftype) -> sftype:
        """
        SCALAR FORM.
        """
        # TODO: update to accept LayoutTensor[]s
        return grad_output if x > 0.0 else 0.0
