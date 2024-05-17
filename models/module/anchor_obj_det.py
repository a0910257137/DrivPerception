import tensorflow as tf
from pprint import pprint
from keras_flops import get_flops


class AnchorObjDet(tf.keras.Model):

    def __init__(self, config, backbone, neck, head, **kwargs):
        super(AnchorObjDet, self).__init__(**kwargs)
        self.config = config
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @tf.function
    def call(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        preds = self.head(x)
        return preds
