from tensorflow.python.client import device_lib
import tensorflow as tf
import os


def count_variables():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return gpu_names


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device='/gpu:0'):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    PS_OPS = [
        'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
        'MutableHashTableOfTensors', 'MutableDenseHashTable'
    ]

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign


def set_gpu(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    # gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # tf.compat.v1.keras.backend.set_session(
    #     tf.compat.v1.Session(config=gpu_config))
    for i in range(len(gpus)):
        tf.config.experimental.set_memory_growth(gpus[i], True)
