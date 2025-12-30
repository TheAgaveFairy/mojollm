from algorithm.functional import vectorize, parallelize
from layout import Layout, LayoutTensor

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
            x.ptr.store[width = width](relu)
        vectorize[nelts](layout.size(), vectorize_closure)

    @staticmethod
    @always_inline("nodebug")
    fn backward(x: sftype, grad_output: sftype) -> sftype:
        """
        SCALAR FORM.
        """
        # TODO: update to accept LayoutTensor[]s
        return grad_output if x > 0.0 else 0.0


