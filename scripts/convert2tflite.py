import tensorflow as tf
import os
import numpy as np
import cv2
from glob import glob

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def representative_dataset_gen():
    path = "/aidata/anders/data_collection/okay/demo_test/imgs"
    img_paths = list(glob(os.path.join(path, "*.jpg")))
    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (1280, 720))
        img = np.reshape(img, (-1, 720, 1280, 3))
        yield [img]


class PreProcessModel(tf.keras.Model):

    def __init__(self, size: tuple, *args, **kwargs):
        super(PreProcessModel, self).__init__(*args, **kwargs)
        self.size = size
        self.scale = 1 / 255
        self.resize = tf.image.resize

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        x = self.resize(images=x,
                        size=self.size,
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        x *= self.scale
        return x


input_height = 720
input_width = 1280
inputs = tf.constant(0,
                     shape=(1, input_height, input_width, 3),
                     dtype=tf.uint8)
input_shape = (input_height, input_width, 3)
image_inputs = tf.keras.Input(shape=input_shape,
                              name='image_inputs',
                              dtype=tf.uint8)
preprocess = PreProcessModel((320, 320))
x = preprocess(image_inputs)

cp_dir = "/aidata/anders/data_collection/okay/total/archives/whole/VoVGSCSP/branch"
restored_model = tf.keras.models.load_model(cp_dir)
preds = restored_model(x, training=False)
finalModel = tf.keras.Model(image_inputs, preds)
_ = finalModel(inputs)
converter = tf.lite.TFLiteConverter.from_keras_model(finalModel)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset_gen
converter.inference_input_type = tf.uint8
converter.target_spec.supported_types = [tf.int8]

tflite_model = converter.convert()
model_path = os.path.join(cp_dir, "tflite/MTFD_INT8.tflite")
with open(model_path, 'wb') as f:
    f.write(tflite_model)
