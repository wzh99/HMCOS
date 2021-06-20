import tensorflow as tf
from tensorflow.python import tf2
from tf2onnx import convert
import tensorflow_hub as hub

batch_size = 1
image_shape_nchw = (3, 224, 224)
image_shape_nhwc = (224, 224, 3)

def cnn_to_onnx(model: tf.keras.Model, input_shape, name: str):
    convert.from_keras(model, input_signature=[tf.TensorSpec(
        (batch_size,) + input_shape)], output_path=f'model/{name}.onnx', opset=10)


def cnn_hub_to_onnx(url: str, name: str):
    model = tf.keras.Sequential(layers=[hub.KerasLayer(url)])
    cnn_to_onnx(model, image_shape_nhwc, name)


# cnn_hub_to_onnx('https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_100_224/classification/5', 'mobilenet_v2')
# cnn_hub_to_onnx('https://hub.tensorflow.google.cn/google/imagenet/nasnet_mobile/classification/5', 'nasnet_mobile')
# cnn_hub_to_onnx('https://hub.tensorflow.google.cn/google/imagenet/inception_v3/classification/5', 'inception_v3')
