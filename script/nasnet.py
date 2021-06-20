from typing import List, Tuple
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.layers import *
from enum import IntEnum, auto


batch_size = 1
num_stem_filters = 32
num_stacked = 4

backend.set_image_data_format('channels_last')


class CellKind(IntEnum):
    NORMAL = auto()
    REDUCTION = auto()


class Architecture:
    def name() -> str:
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
    def name() -> str:
        return 'cifar100'

    def input_shape(self) -> Tuple[int]:
        return (32, 32, 3)

    def num_classes(self) -> int:
        return 100

    def num_init_filters(self) -> int:
        return 44  # 44 * 4 * 6 = 1056

    def stem_cell(self, x):
        x = Conv2D(num_stem_filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        return x

    def cells(self) -> List[CellKind]:
        normal = [CellKind.NORMAL] * num_stacked
        reduction = [CellKind.REDUCTION]
        return normal + reduction + normal + reduction + normal


class ImageNet(Architecture):
    def name() -> str:
        return 'imagenet'

    def input_shape(self) -> Tuple[int]:
        return (224, 224, 3)

    def num_classes(self) -> int:
        return 1000

    def num_init_filters(self) -> int:
        return 11  # 11 * 16 * 6 = 1056

    def stem_cell(self, x):
        x = Conv2D(num_stem_filters, 3, strides=(2, 2),
                   padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        return x

    def cells(self) -> List[CellKind]:
        normal = [CellKind.NORMAL] * num_stacked
        reduction = [CellKind.REDUCTION]
        return reduction * 2 + normal + reduction + normal + reduction + normal


class NasNetBase:
    def __init__(self, name: str) -> None:
        self.name = name

    def build(self, arch: Architecture) -> keras.Model:
        # Stem from inut
        inp = Input(shape=arch.input_shape(), batch_size=batch_size)
        cur = arch.stem_cell(inp)

        # Build cells
        prev = None
        cur_filters = arch.num_init_filters()
        num_normal, num_reduct = 0, 0
        for kind in arch.cells():
            if kind == CellKind.NORMAL:
                with backend.name_scope(f'normal_{num_normal}'):
                    nxt = self._create_normal(prev, cur, cur_filters)
                num_normal += 1
            else:
                cur_filters *= 2
                with backend.name_scope(f'reduction_{num_reduct}'):
                    nxt = self.reduction_cell(prev, cur, cur_filters)
                num_reduct += 1
            prev, cur = cur, nxt

        # Final layer
        with backend.name_scope('final_layer'):
            x = ReLU()(cur)
            x = GlobalAvgPool2D()(x)
            x = Dense(arch.num_classes())(x)

        return keras.Model(inputs=inp, outputs=x, name=f'{self.name}-{arch.name()}')

    def _create_normal(self, prev, cur, num_filters: int):
        cur = _squeeze(cur, num_filters)
        prev = _fit(prev, cur, num_filters)
        return self.normal_cell(prev, cur, num_filters)

    def _create_reduction(self, prev, cur, num_filters: int):
        prev = _fit(prev, cur, num_filters)
        cur = _squeeze(cur, num_filters)
        return self.reduction_cell(prev, cur, num_filters)

    def normal_cell(self, prev, cur, num_filters: int):
        raise NotImplementedError()

    def reduction_cell(self, prev, cur, num_filters: int):
        raise NotImplementedError()


def _sep_conv(x, num_filters: int, kernel_size: int, strides: int = 1):
    with backend.name_scope('sep_conv'):
        x = ReLU()(x)
        x = SeparableConv2D(num_filters, kernel_size,
                            strides=strides, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SeparableConv2D(num_filters, kernel_size,
                            padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        return x


def _fit(src, tgt, num_filters: int):
    if src is None:
        return tgt
    if src.shape[2] == tgt.shape[2]:
        return _squeeze(src, num_filters)
    with backend.name_scope('fit'):
        x = ReLU()(src)
        p1 = AvgPool2D(pool_size=(1, 1), strides=(2, 2))(x)
        p1 = Conv2D(num_filters // 2, 1, use_bias=False)(p1)
        p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
        p2 = AvgPool2D(pool_size=(1, 1), strides=(2, 2))(p2)
        p2 = Conv2D(num_filters // 2, 1, use_bias=False)(p2)
        x = concatenate([p1, p2])
        x = BatchNormalization()(x)
        return x


def _squeeze(x, num_filters: int):
    with backend.name_scope('squeeze'):
        x = ReLU()(x)
        x = Conv2D(num_filters, 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        return x
