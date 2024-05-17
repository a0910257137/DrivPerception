import tensorflow as tf
import gc
import resource


class ClearMemory(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print('Memory usage: ',
              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
