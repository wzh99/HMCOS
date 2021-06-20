
import onnx
from onnx import shape_inference
import onnxoptimizer as opt

model_name = 'mobilebert'
prefix = "sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/"


def short_name(name: str):
    pos = name.find(prefix)
    return name if pos == -1 else name[pos + len(prefix):]


def process_value_names(graph: onnx.GraphProto):
    # Process inputs and outputs
    for val in graph.input:
        val.name = short_name(val.name)
    for val in graph.output:
        val.name = short_name(val.name)

    # Process nodes
    for node in graph.node:
        node.name = short_name(node.name)
        for i, name in enumerate(node.input):
            node.input[i] = short_name(name)
        for i, name in enumerate(node.output):
            node.output[i] = short_name(name)

    # Process parameters
    for param in graph.initializer:
        param.name = short_name(param.name)


def remove_cnn_preproc(graph: onnx.GraphProto):
    # Transpose input dims to NCHW
    inp = graph.input[0]
    dim = inp.type.tensor_type.shape.dim
    new_dim = [dim[0], dim[3], dim[1], dim[2]]
    del dim[:]
    dim.extend(new_dim)

    # Remove all preprocessing layers
    for i, node in enumerate(graph.node):
        if node.op_type == 'Transpose':
            break
    del graph.node[:i+1]
    graph.node[0].input[0] = inp.name
    pass


model = onnx.load(f'model/{model_name}.onnx')
process_value_names(model.graph)
# remove_cnn_preproc(model.graph)
# model = opt.optimize(model, passes=['fuse_bn_into_conv'])
# model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
onnx.save_model(model, f'model/{model_name}.opt.onnx')
