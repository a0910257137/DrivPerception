import tensorflow as tf
from pprint import pprint


class AnchorFreeObjDet(tf.keras.Model):

    def __init__(self, config, backbone, neck, head, **kwargs):
        super(AnchorFreeObjDet, self).__init__(**kwargs)
        self.config = config
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @tf.function
    def call(self, x):
        x = self.backbone(x)
        if self.neck is None:
            pass
        else:
            x = self.neck(x)
        preds = self.head(x)
        return preds
