import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import argparse
from monitor import logger
from dataset.general import GeneralDataset
from system.restore import Restore
from system.initial import Initialize
from utils.sys_tools import set_gpu
from models.model_factory import ModelFactory
from utils.tools import *
from utils.io import *


def train(config, initial, is_restore, excluded_layers):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    general_dataset = GeneralDataset(config.data_reader, mirrored_strategy)
    datasets = general_dataset.get_datasets()
    train_datasets, test_datasets = datasets["train"], datasets["test"]
    training_length, testing_length = (datasets["training_length"],
                                       datasets["testing_length"])
    with mirrored_strategy.scope():
        model, optimizer = ModelFactory(config.models,
                                        config.learn_rate).build_model()
        if initial:
            model = Initialize(config.models)(model)
        if is_restore:
            model = Restore(config.model_path,
                            config.data_reader.resize_size).build_restoration(
                                model, excluded_layers)
    callbacks = get_callbacks(config, model, optimizer, train_datasets,
                              test_datasets, training_length, testing_length)
    model.fit(
        train_datasets,
        validation_data=test_datasets,
        use_multiprocessing=True,
        epochs=config.epochs,
        callbacks=callbacks,
    )


def argparser():
    parser = argparse.ArgumentParser(description="Facial landmarks and bboxes")
    parser.add_argument("--config", type=str, help="Configuration file to use")
    parser.add_argument("--gpus", help="Use gpus for training or not")
    parser.add_argument("--initial",
                        action="store_true",
                        help="Initial weights for head weights ans bias")
    parser.add_argument("--restore",
                        action="store_true",
                        help="Restore pre-trained model in config")
    parser.add_argument(
        "--excluded_layers",
        nargs="+",
        help="Exclude layers with input keywords when restore",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    set_gpu(args.gpus)
    logger.info(f"Use config: {args.config} to train behavior")
    config = load_configger(args.config)
    train(config, args.initial, args.restore, args.excluded_layers)
