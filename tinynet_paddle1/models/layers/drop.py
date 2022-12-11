""" DropBlock, DropPath

PyTorch implementations of DropBlock and DropPath (Stochastic Depth) regularization layers.

Papers:
DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)

Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)

Code:
DropBlock impl inspired by two Tensorflow impl that I liked:
 - https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
 - https://github.com/clovaai/assembled-cnn/blob/master/nets/blocks.py

Hacked together by Ross Wightman
"""
import paddle
import paddle.nn as nn
from paddle import fluid
import paddle.nn.functional as F


def drop_block_2d(
        x, drop_prob: float = 0.1, training: bool = False, block_size: int = 7,
        gamma_scale: float = 1.0, drop_with_noise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.

    """
    if drop_prob == 0. or not training:
        return x
    _, _, height, width = x.shape
    total_size = width * height
    clipped_block_size = min(block_size, min(width, height))
    # seed_drop_rate, the gamma parameter
    seed_drop_rate = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (width - block_size + 1) *
            (height - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = paddle.meshgrid(paddle.arange(width).to(x.device), paddle.arange(height).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < width - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < height - (clipped_block_size - 1) // 2))
    valid_block = paddle.reshape(valid_block, (1, 1, height, width)).float()

    uniform_noise = fluid.layers.gaussian_random_batch_size_like(x, dtype="float32")
    block_mask = ((2 - seed_drop_rate - valid_block + uniform_noise) >= 1).float()
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size, ???
        stride=1,
        padding=clipped_block_size // 2)

    if drop_with_noise:
        normal_noise = fluid.layers.gaussian_random_batch_size_like(x)
        x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / (paddle.sum(block_mask) + 1e-7)
        x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Layer):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise

    def forward(self, x):
        return drop_block_2d(x, self.drop_prob, self.training, self.block_size, self.gamma_scale, self.with_noise)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + paddle.rand((x.size()[0], 1, 1, 1), dtype=x.dtype)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.LayerDict):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
