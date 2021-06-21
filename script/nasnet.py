from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import onnx
from onnx import shape_inference
from tensorflow import keras
from tensorflow.keras import backend, Model
from tensorflow.keras.layers import *
from enum import IntEnum, auto
from collections import namedtuple
from tf2onnx import convert
import tensorflow as tf
import onnxoptimizer


batch_size = 1
num_stem_filters = 32
num_stacked = 4

backend.set_image_data_format('channels_first')


class CellKind(IntEnum):
    NORMAL = auto()
    REDUCTION = auto()


class Architecture:
    def name(self) -> str:
        raise NotImplementedError()

    def input_shape(self) -> Tuple[int]:
        raise NotImplementedError()

    def num_classes(self) -> int:
        raise NotImplementedError()

    def num_init_filters(self) -> int:
        raise NotImplementedError()

    def stem_cell(self, x):
        raise NotImplementedError()

    def cells(self) -> List[CellKind]:
        raise NotImplementedError()


class Cifar100(Architecture):
    def name(self) -> str:
        return 'cifar100'

    def input_shape(self) -> Tuple[int]:
        return (3, 32, 32)

    def num_classes(self) -> int:
        return 100

    def num_init_filters(self) -> int:
        return 44  # 44 * 4 * 6 = 1056

    def stem_cell(self, x):
        x = Conv2D(num_stem_filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=1)(x)
        return x

    def cells(self) -> List[CellKind]:
        normal = [CellKind.NORMAL] * num_stacked
        reduction = [CellKind.REDUCTION]
        return normal + reduction + normal + reduction + normal


class ImageNet(Architecture):
    def name(self) -> str:
        return 'imagenet'

    def input_shape(self) -> Tuple[int]:
        return (3, 224, 224)

    def num_classes(self) -> int:
        return 1000

    def num_init_filters(self) -> int:
        return 11  # 11 * 16 * 6 = 1056

    def stem_cell(self, x):
        x = Conv2D(num_stem_filters, 3, strides=2,
                   use_bias=False)(x)
        x = BatchNormalization(axis=1)(x)
        return x

    def cells(self) -> List[CellKind]:
        normal = [CellKind.NORMAL] * num_stacked
        reduction = [CellKind.REDUCTION]
        return reduction * 2 + normal + reduction + normal + reduction + normal


Genotype = namedtuple(
    'Genotype', ['normal', 'normal_concat', 'reduction', 'reduction_concat'])


ops: Dict[str, Callable[[Any, int, int], Any]] = {
    'id': lambda x, f, s: x,
    'sep3x3': lambda x, f, s: _sep_conv(x, f, 3, s),
    'sep5x5': lambda x, f, s: _sep_conv(x, f, 5, s),
    'sep7x7': lambda x, f, s: _sep_conv(x, f, 7, s),
    'dil3x3': lambda x, f, s: _dil_conv(x, f, 3, s, 2),
    'dil5x5': lambda x, f, s: _dil_conv(x, f, 5, s, 2),
    'avg3x3': lambda x, f, s: AvgPool2D(pool_size=3, strides=s, padding='same')(x),
    'max3x3': lambda x, f, s: MaxPool2D(pool_size=3, strides=s, padding='same')(x),
}


class NasNetBase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.genotype: Optional[Genotype] = None

    def build(self, arch: Architecture) -> Model:
        # Stem from inut
        inp = Input(shape=arch.input_shape(), batch_size=batch_size)
        cur = arch.stem_cell(inp)

        # Build cells
        assert self.genotype is not None
        keras.applications.NASNetLarge
        prev = None
        cur_filters = arch.num_init_filters()
        for kind in arch.cells():
            if kind == CellKind.NORMAL:
                nxt = self._create_normal(prev, cur, cur_filters)
            else:
                nxt = self._create_reduction(prev, cur, cur_filters)
                cur_filters *= 2
            prev, cur = cur, nxt

        # Final layer
        x = ReLU()(cur)
        x = GlobalAvgPool2D()(x)
        x = Dense(arch.num_classes())(x)

        return Model(inputs=inp, outputs=x, name=f'{self.name}-{arch.name()}')

    def _create_normal(self, prev, cur, num_filters: int):
        cur = _squeeze(cur, num_filters)
        prev = _fit(prev, cur, num_filters)
        return self._create_cell(prev, cur, num_filters, self.genotype.normal,
                                 self.genotype.normal_concat, False)

    def _create_reduction(self, prev, cur, num_filters: int):
        prev = _fit(prev, cur, num_filters)
        cur = _squeeze(cur, num_filters)
        return self._create_cell(prev, cur, num_filters, self.genotype.reduction,
                                 self.genotype.reduction_concat, True)

    def _create_cell(self, prev, cur, num_filters: int, block_genos: List[List[Tuple[str, int]]],
                     concat: List[int], reduction: bool):
        blocks = [prev, cur]
        for i, block_geno in enumerate(block_genos):
            leftGeno, rightGeno = block_geno
            lhs = self._create_op(
                num_filters, blocks, leftGeno[0], leftGeno[1], reduction)
            rhs = self._create_op(
                num_filters, blocks, rightGeno[0], rightGeno[1], reduction)
            blocks.append(add([lhs, rhs]))
        concated = [blocks[idx] for idx in concat]
        return concatenate(concated, axis=1)

    def _create_op(self, num_filters: int, blocks: List[Any], name: str, arg: int,
                   reduction: bool):
        op = ops[name]
        strides = 2 if reduction and arg < 2 else 1
        return op(blocks[arg], num_filters, strides)


def _sep_conv(x, num_filters: int, kernel_size: int, strides: int):
    x = ReLU()(x)
    x = SeparableConv2D(num_filters, kernel_size,
                        strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)
    x = SeparableConv2D(num_filters, kernel_size,
                        padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    return x


def _dil_conv(x, num_filters: int, kernel_size: int, strides: int, dilation: int):
    x = ReLU()(x)
    x = SeparableConv2D(num_filters, kernel_size, padding='same',
                        dilation_rate=dilation, use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    return x


def _fit(src, tgt, num_filters: int):
    if src is None:
        return tgt
    if src.shape[2] == tgt.shape[2]:
        return _squeeze(src, num_filters)
    x = ReLU()(src)
    p1 = AvgPool2D(pool_size=1, strides=2)(x)
    p1 = Conv2D(num_filters // 2, 1, use_bias=False)(p1)
    p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
    p2 = AvgPool2D(pool_size=1, strides=2)(p2)
    p2 = Conv2D(num_filters // 2, 1, use_bias=False)(p2)
    x = concatenate([p1, p2], axis=1)
    x = BatchNormalization(axis=1)(x)
    return x


def _squeeze(x, num_filters: int):
    x = ReLU()(x)
    x = Conv2D(num_filters, 1, use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    return x


class NasNetA(NasNetBase):
    def __init__(self) -> None:
        super().__init__('nasnet_a')
        self.genotype = Genotype(
            normal=[
                [('sep5x5', 1), ('sep3x3', 0)],
                [('sep5x5', 0), ('sep3x3', 0)],
                [('avg3x3', 1), ('id', 0)],
                [('avg3x3', 0), ('avg3x3', 0)],
                [('sep3x3', 1), ('id', 1)],
            ],
            normal_concat=[0, 2, 3, 4, 5, 6],
            reduction=[
                [('sep5x5', 1), ('sep7x7', 0)],
                [('max3x3', 1), ('sep5x5', 0)],
                [('avg3x3', 1), ('sep5x5', 0)],
                [('id', 3), ('avg3x3', 2)],
                [('sep3x3', 2), ('max3x3', 1)]
            ],
            reduction_concat=[3, 4, 5, 6],
        )


def create_model(arch_ty: Type[Architecture], net_ty: Type[NasNetBase]):
    arch = arch_ty()
    net = net_ty().build(arch)
    input_spec = tf.TensorSpec((batch_size,) + arch.input_shape())
    model, _ = convert.from_keras(net, [input_spec], opset=10)
    model = onnxoptimizer.optimize(model, passes=['fuse_bn_into_conv'])
    model = shape_inference.infer_shapes(model, check_type=True)
    onnx.save_model(model, f'model/{net.name}.onnx')


create_model(Cifar100, NasNetA)