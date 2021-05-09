import tensorflow_hub as hub
import tensorflow as tf
import tf2onnx


def to_onnx(name: str, url: str):
    m = tf.keras.Sequential([hub.KerasLayer(url)])
    tf2onnx.convert.from_keras(m, input_signature=[tf.TensorSpec(
        (16, 224, 224, 3))], output_path=f'model/{name}.onnx')


# to_onnx('mobilenet_v2', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5')
# to_onnx('mobilenet_v3', 'https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5')
to_onnx('resnet_v1', "https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5")
# to_onnx('nasnet_mobile', 'https://tfhub.dev/google/imagenet/nasnet_mobile/classification/5')
# to_onnx('inception_v3', 'https://tfhub.dev/google/imagenet/inception_v3/classification/5')

# encoder = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1")
# encoder_inputs = dict(
#     input_word_ids=tf.keras.layers.Input(shape=(512,), batch_size=16, dtype=tf.int32),
#     input_mask=tf.keras.layers.Input(shape=(512,), batch_size=16, dtype=tf.int32),
#     input_type_ids=tf.keras.layers.Input(shape=(512,), batch_size=16, dtype=tf.int32),
# )
# out = encoder(encoder_inputs)
# model = tf.keras.Model(inputs=encoder_inputs, outputs=out)
# tf.saved_model.save(model, 'model/mobilebert')

