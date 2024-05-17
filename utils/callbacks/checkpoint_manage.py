import tensorflow as tf
from pathlib import Path
import os
from monitor import logger
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from pprint import pprint
import gc


class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    """
    Callback wraping `tf.train.CheckpointManager`.
    Restores previous checkpoint `on_train_begin`
    Example usage:
    ```python
    model = get_model(...)
    model.compile(optimizer=optimizer, ...)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, '/tmp/my_model', max_to_keep=5)
    callback = CheckpointManagerCallback(checkpoint, manager, period=1)
    model.fit(..., callbacks=[callbacks])
    ```
    """

    def __init__(self,
                 checkpoint,
                 manager,
                 model,
                 directory,
                 period=1,
                 save_on_train_end=True):
        self._manager = manager
        self._checkpoint = checkpoint
        self._period = period
        self._save_on_train_end = save_on_train_end
        self._restored = False
        self._epoch_count = None
        self._last_save = None
        self.directory = directory
        self.model = model

    # def on_batch_begin(self, epoch, logs=None):
    # self.model.epochs += 1
    # imgs = tf.constant(0., shape=(1, 384, 640, 3))
    # self.model.model(imgs, training=False)
    # model_dir = "/aidata/anders/autosys/archives/sunplus"
    # tf.keras.models.save_model(self.model.model, model_dir)
    # testing_model = tf.keras.models.load_model(model_dir)
    # results = testing_model(imgs, training=False)
    # exit(1)

    def on_epoch_end(self, epoch, logs=None):
        epochs_finished = epoch + 1
        self.model.epochs = epochs_finished
        self._epoch_count = epochs_finished
        if epochs_finished % self._period == 0:
            self._save()

    def on_train_end(self, logs=None):
        if self._save_on_train_end:
            self._save()

    def _save(self):
        if self._epoch_count is None:
            return
        if self._last_save != self._epoch_count:
            # save per epoch
            if self._epoch_count % 10 == 0:
                tf.keras.models.save_model(self.model.model, self.directory)
            self._last_save = self._epoch_count
