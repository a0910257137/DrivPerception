import math
import tensorflow as tf
import tensorflow_addons as tfa
from pprint import pprint


class Scale(tf.keras.layers.Layer):

    def __init__(self, init_value, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.weight = tf.Variable(initial_value=1,
                                  name='scale_weight',
                                  trainable=True)
        super(Scale, self).build(input_shape)

    def cell(self, inputs, training=False):
        return self.weight * inputs
