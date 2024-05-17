"""
Misc Utility functions
"""
import os
import logging
import datetime
import commentjson
import tensorflow as tf
from .callbacks import EmbeddingMap, LossAndErrorPrintingCallback, CheckpointManagerCallback, Histogram, ClearMemory, WarmUpCosineDecayScheduler
from monitor import logger


def get_callbacks(config, model, optimizer, train_datasets, test_datasets,
                  training_length, testing_length):
    callbacks = []
    train_writer = tf.summary.create_file_writer(
        os.path.join(config.summary.log_dir, 'train'))
    eval_writer = tf.summary.create_file_writer(
        os.path.join(config.summary.log_dir, 'validation'))
    writers = {'train': train_writer, 'validation': eval_writer}
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_dir = config.model_path

    clear_memory = ClearMemory()
    manager = tf.train.CheckpointManager(checkpoint,
                                         checkpoint_dir,
                                         max_to_keep=5)

    saver_callback = CheckpointManagerCallback(checkpoint,
                                               manager,
                                               model,
                                               directory=checkpoint_dir,
                                               period=1)

    loss_callback = LossAndErrorPrintingCallback(writers)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=config.summary.log_dir,
        write_graph=False,
        write_images=False,
        update_freq='epoch',
        profile_batch='500, 520')
    # histogram = Histogram(config=config, writers=writers, update_freq=1000)
    # embedding_map = EmbeddingMap(config=config,
    #                              writers=writers,
    #                              train_datasets=train_datasets,
    #                              test_datasets=test_datasets,
    #                              update_freq=1000)
    cosine_decay_scheduler = WarmUpCosineDecayScheduler(
        config.learn_rate, config.epochs, training_length)
    callbacks.append([
        saver_callback, cosine_decay_scheduler, tensorboard_callback,
        loss_callback, clear_memory
    ])
    return callbacks


def get_logger():
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    return logger


def load_configger(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError('File %s does not exist.' % config_path)
    with open(config_path, 'r') as fp:
        config = commentjson.loads(fp.read())
    config = AttrDict(config)
    config = set_data_config(config)
    config = set_model_config(config)
    return config


def set_data_config(config):
    config.data_reader.epochs = config.epochs
    config.data_reader.batch_size = config.batch_size
    config.data_reader.model_name = config.models.model_name
    return config


def set_model_config(config):
    config.models.loss.batch_size = config.batch_size
    config.models.loss.resize_size = config.data_reader.resize_size
    config.models.tasks = config.data_reader.tasks
    config.models.max_obj_num = config.data_reader.max_obj_num
    config.models.resize_size = config.data_reader.resize_size
    config.models.batch_size = config.batch_size
    config.models.lr = config.learn_rate
    config.models.loss.max_obj_num = config.data_reader.max_obj_num
    return config


class AttrDict(dict):
    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if self.__dict__[AttrDict.IMMUTABLE]:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(key, value))

        if isinstance(value, dict):
            value = AttrDict(value)

        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def set_immutable(self, is_immutable=True):
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable

        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.set_immutable(is_immutable)

        for v in self.values():
            if isinstance(v, AttrDict):
                v.set_immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]
