import tensorflow as tf
from tf2onnx import convert
import tensorflow_hub as hub

batch_size = 16
image_shape_nchw = (3, 224, 224)
image_shape_nhwc = (224, 224, 3)
seq_len = 128


def cnn_to_onnx(model: tf.keras.Model, input_shape, name: str):
    convert.from_keras(model, input_signature=[tf.TensorSpec(
        (batch_size,) + input_shape)], output_path=f'model/{name}.onnx', opset=10)


def cnn_hub_to_onnx(url: str, name: str):
    model = tf.keras.Sequential(layers=[hub.KerasLayer(url)])
    cnn_to_onnx(model, image_shape_nhwc, name)


# cnn_hub_to_onnx('https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_100_224/classification/5', 'mobilenet_v2')
# cnn_hub_to_onnx('https://hub.tensorflow.google.cn/google/imagenet/nasnet_mobile/classification/5', 'nasnet_mobile')
# cnn_hub_to_onnx('https://hub.tensorflow.google.cn/google/imagenet/inception_v3/classification/5', 'inception_v3')

def transformer_to_onnx(url: str, name: str):
    encoder = hub.KerasLayer(url)
    input_word_ids = tf.keras.layers.Input(
        shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)
    input_mask = tf.keras.layers.Input(
        shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)
    input_type_ids = tf.keras.layers.Input(
        shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)
    encoder_inputs = dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids,
    )
    out = encoder(encoder_inputs)
    model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids], outputs=out['sequence_output'])
    input_sig = [
        tf.TensorSpec((batch_size, seq_len), tf.int32),
        tf.TensorSpec((batch_size, seq_len), tf.int32),
        tf.TensorSpec((batch_size, seq_len), tf.int32),
    ]
    convert.from_keras(model, input_signature=input_sig,
                       output_path=f'model/{name}.onnx', opset=12)


# transformer_to_onnx('https://tfhub.dev/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1', 'mobilebert')
# transformer_to_onnx('https://hub.tensorflow.google.cn/tensorflow/albert_en_base/3', 'albert')