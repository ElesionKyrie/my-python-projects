""" Activations

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Hacked together by Ross Wightman
"""


import paddle
import torch.jit
from paddle import nn as nn
import paddle.fluid as fluid
from paddle.nn import functional as F


_USE_MEM_EFFICIENT_ISH = True
if _USE_MEM_EFFICIENT_ISH:
    # This version reduces memory overhead of Swish during training by
    # recomputing torch.sigmoid(x) in backward instead of saving it.
    @torch.jit.script
    #@paddle.jit.to_static
    def swish_jit_fwd(x):
        return fluid.layers.mul(x, fluid.layers.sigmoid(x))                    #修改

    @torch.jit.script
    # @paddle.jit.to_static
    def swish_jit_bwd(x, grad_output):
        x_sigmoid = fluid.layers.sigmoid(x)
        return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


    class SwishJitAutoFn(paddle.autograd.PyLayer):
        """ torch.jit.script optimised Swish
        Inspired by conversation btw Jeremy Howard & Adam Pazske
        https://twitter.com/jeremyphoward/status/1188251041835315200
        """

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return swish_jit_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return swish_jit_bwd(x, grad_output)


    def swish(x, _inplace=False):
        return SwishJitAutoFn.apply(x)


    @torch.jit.script
    @paddle.jit.to_static
    def mish_jit_fwd(x):
        return x.mul(paddle.tanh(F.softplus(x)))


    @torch.jit.script
    @paddle.jit.to_static
    def mish_jit_bwd(x, grad_output):
        x_sigmoid = paddle.nn.functional.sigmoid(x)
        x_tanh_sp = F.softplus(x).tanh()
        return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


    class MishJitAutoFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return mish_jit_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return mish_jit_bwd(x, grad_output)

    def mish(x, _inplace=False):
        return MishJitAutoFn.apply(x)

else:
    def swish(x, inplace: bool = False):
        """Swish - Described in: https://arxiv.org/abs/1710.05941
        """
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


    def mish(x, _inplace: bool = False):
        """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
        """
        return x.mul(F.softplus(x).tanh())


class Swish(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class Mish(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x, self.inplace)


def sigmoid(x, inplace: bool = False):
    return fluid.layers.sigmoid.sigmoid_(x) if inplace else fluid.layers.sigmoid.sigmoid(x)


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x, inplace: bool = False):
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argmument interface
class Tanh(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x,):
    inner = F.relu6(x + 3.).div_(6.)
    return fluid.layers.mul(x, inner)


class HardSwish(nn.Layer):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return hard_swish(x)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class HardSigmoid(nn.Layer):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)

