from typing import List, NamedTuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, Model
from tensorflow.keras.layers import *
from tf2onnx import convert
from onnxoptimizer import optimize
import onnx

backend.set_image_data_format('channels_first')


class Vertex(NamedTuple):
    preds: List[int]
    succs: List[int]


def gen_random_graph(n: int, k: int, p: float) -> List[Vertex]:
    adj = np.eye(n, dtype=bool)

    for i in range(n):
        for j in range(i-k//2, i+k//2+1):
            real_j = j % n
            if real_j == i:
                continue
            adj[real_j, i] = adj[i, real_j] = True

    for i in range(n):
        for j in range(k // 2):
            current = (i + j + 1) % n
            if np.random.rand() < p:  # rewire
                unoccupied = [x for x in range(n) if not adj[i, x]]
                rewired = np.random.choice(unoccupied)
                adj[i, current] = adj[current, i] = False
                adj[i, rewired] = adj[rewired, i] = True

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append((i, j))
    edges.sort()

    # Find predecessors and successors of vertices
    vertices = [Vertex([], []) for _ in range(n)]
    for (u, v) in edges:
        vertices[u].succs.append(v)
        vertices[v].preds.append(u)

    return vertices


def build_op(inputs, filters: int, strides: int):
    if len(inputs) == 1:
        x = inputs[0]
    else:
        x = tf.stack(inputs, axis=-1)
        agg_w = tf.Variable(np.zeros(len(inputs,), dtype='float32'))
        x = tf.linalg.matvec(x, tf.sigmoid(agg_w))
    x = ReLU()(x)
    x = SeparableConv2D(filters, 3, strides=strides,
                        padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    return x


def build_cell(x, filters: int):
    # Generate random graph
    vertices = gen_random_graph(32, 4, 0.75)

    # Build each node
    node_out = [None] * len(vertices)
    cell_out = []
    for i, v in enumerate(vertices):
        if len(v.preds) == 0:
            y = build_op([x], filters, 2)
        else:
            y = build_op([node_out[j] for j in v.preds], filters, 1)
        node_out[i] = y
        if len(v.succs) == 0:
            cell_out.append(y)

    return tf.reduce_mean(tf.stack(cell_out), axis=0)


def build_net(channels: int):
    inp = Input(shape=(3, 224, 224), batch_size=1)
    x = Conv2D(channels // 2, 3, strides=2,
               padding='same', use_bias=False)(inp)
    x = BatchNormalization(axis=1)(x)
    x = ReLU()(x)
    x = Conv2D(channels, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = build_cell(x, channels)
    x = build_cell(x, channels * 2)
    x = build_cell(x, channels * 4)
    x = Conv2D(1280, 1, use_bias=False)(x)
    x = BatchNormalization(axis=1)(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(1000)(x)
    net = Model(inputs=inp, outputs=x)
    net.summary()
    return net


def create_onnx():
    net = build_net(78)
    input_spec = tf.TensorSpec((1, 3, 224, 224))
    model, _ = convert.from_keras(net, [input_spec])
    model = optimize(model, passes=['fuse_bn_into_conv'])
    model = onnx.shape_inference.infer_shapes(model, check_type=True)
    onnx.save_model(model, f'model/randwire.onnx')


create_onnx()
