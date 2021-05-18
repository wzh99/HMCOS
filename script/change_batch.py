import onnx
from onnx import shape_inference

model_name = 'nasnet_mobile'
batch_size = 1


def change_batch_size(graph: onnx.GraphProto, batch_size: int):
    del graph.value_info[:]
    for val in graph.input:
        val.type.tensor_type.shape.dim[0].dim_value = batch_size
    for val in graph.output:
        val.type.tensor_type.shape.dim[0].dim_value = batch_size


model = onnx.load(f'model/{model_name}.onnx')
change_batch_size(model.graph, batch_size)
model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
onnx.save_model(model, f'model/{model_name}.onnx')