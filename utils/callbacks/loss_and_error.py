import tensorflow as tf

from monitor import logger
import time
import datetime


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

    def __init__(self, writers):
        super(LossAndErrorPrintingCallback, self).__init__()
        self.writers = writers
        self.train_seen = 0
        self.eval_seen = 0
        self.epoch = 0
        self._epoch_count = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count = epoch + 1

    def on_train_batch_begin(self, batch, logs=None):
        self.start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.train_seen += 1
        now = datetime.datetime.now()
        self.end_time = time.time()
        train_info = '\n[TRAIN_LOSSES] %s: %d, %s: %d, %s: %.4fs, %s: %s; ' % (
            'epoch', self.epoch, 'step', batch, 'duration',
            self.end_time - self.start_time, 'Date', now)
        for key in logs.keys():
            train_info += '%s: %.6f, ' % (key, logs[key])
        logger.info(train_info)

    def on_test_batch_begin(self, batch, logs=None):
        self.start_time = time.time()

    def on_test_batch_end(self, batch, logs=None):
        self.eval_seen += 1
        self.end_time = time.time()
        now = datetime.datetime.now()
        eval_info = '\n[EVAL_LOSSES] %s: %d, %s: %d, %s: %.4fs,%s: %s; ' % (
            'epoch', self.epoch, 'step', batch, 'duration',
            self.end_time - self.start_time, 'Date', now)
        for key in logs.keys():
            eval_info += ' %s: %.6f, ' % (key, logs[key])
        logger.info(eval_info)
