# Reference from
# https://www.tensorflow.org/tutorials/images/transfer_learning
"""
This script is used for training image classification models using TensorFlow 2.
It includes functionalities for data preparation, model training, evaluation, and conversion to TensorFlow Lite format.
Classes:
    WORKFOLDER: Manages the project workspace, including creating and deleting directories.
    TRAIN: Handles the training process, including data loading, model creation, training, evaluation, and conversion to TensorFlow Lite.
"""
import os
import shutil
import stat
import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np

# from tqdm import tqdm, trange
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

import data_prepare


class WORKFOLDER:
    """
    A class to manage project directories and logs for a machine learning project.
    Attributes:
    -----------
    proj_path : str
        The path to the project directory.
    """
    def __init__(self, proj_name):
        self.proj_path = os.path.join(os.getcwd(), "workspace", proj_name)

    def create_dirs(self):
        """
        Creates necessary directories for the project if they do not already exist.
        """
        if not os.path.exists(self.proj_path):
            os.makedirs(self.proj_path)
            os.makedirs(os.path.join(self.proj_path, "result_plots"))
            os.makedirs(os.path.join(self.proj_path, "tflite_model"))
            os.makedirs(os.path.join(self.proj_path, "checkpoint"))

        if not os.path.exists(os.path.join(self.proj_path, "logs")):
            os.makedirs(os.path.join(self.proj_path, "logs"))

        return self.proj_path

    def delete_logs_fdr(self):
        """
        Deletes the 'logs' directory within the project path if it exists.
        """

        def rm_dir_readonly(func, path, _):
            "Clear the readonly bit and reattempt the removal"
            os.chmod(path, stat.S_IWRITE)
            func(path)

        if os.path.exists(os.path.join(self.proj_path, "logs")):
            try:
                shutil.rmtree(os.path.join(self.proj_path, "logs"), onerror=rm_dir_readonly)
                print("Clear the previous log!")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")


class TRAIN:
    '''
    A class used to represent the training process for an image classification model using TensorFlow.
    Attributes
    ----------
    proj_path : str
        The project path where the model and related files will be saved.
    base_model : tf.keras.Sequential
        The base model used for transfer learning.
    custom_model : tf.keras.Sequential
        The custom model built on top of the base model.
    output_tflite_location : str
        The location where the TFLite model will be saved.
    flops : int
        The number of floating point operations (FLOPs) of the model.
    total_para : int
        The total number of parameters in the model.
    int8_tflite_size : int
        The size of the int8 quantized TFLite model.
    tf_callback : tf.keras.callbacks.TensorBoard
        The TensorBoard callback for logging.
    '''

    def __init__(self, proj_pth):
        self.proj_path = proj_pth
        self.base_model = tf.keras.Sequential([])
        self.custom_model = tf.keras.Sequential([])
        self.output_tflite_location = os.path.join(self.proj_path, "tflite_model")
        self.flops = 0
        self.total_para = 0
        self.int8_tflite_size = 0

        self.tf_callback = None

    def prepare_setting(self, args):
        """
        Prepares and returns a dictionary of settings based on the provided arguments.
        Returns:
            dict: A dictionary containing the settings.
        """

        return {
            "data_exist": args.data_exist,
            "url": args.url,
            "zip_name": args.zip_name,
            "dataset_name": args.dataset_name,
            "proj_name": args.proj_name,
            "BATCH_SIZE": args.BATCH_SIZE,
            "IMG_SIZE": args.IMG_SIZE,
            "VAL_PCT": args.VAL_PCT,
            "MODEL_NAME": args.MODEL_NAME,
            "TEST_PCT": args.TEST_PCT,
            "DATA_AUGM": args.DATA_AUGM,
            "EPOCHS": args.EPOCHS,
            "LEARNING_RATE": args.LEARNING_RATE,
            "FINE_TUNE_LAYER": args.FINE_TUNE_LAYER,
            "STEPS_PER_EPOCH": args.STEPS_PER_EPOCH,
            "switch_mode": args.switch_mode,
            "TFLITE_F": args.switch_mode,
            "TFLITE_TEST_BATCH_N": args.TFLITE_TEST_BATCH_N,
            "IMAGENET_MODEL_EN": args.IMAGENET_MODEL_EN,
            "ALPHA_WIDTH": args.ALPHA_WIDTH,
        }

    def data_pre_load(self, dir_list, d_style, info_dict):
        """
        Pre-loads and processes image data for training, validation, and testing.
        Args:
            dir_list (list): List of directories containing the image data. 
            d_style (int): Dataset style. 
                           1 - Separate directories for training and validation.
                           2 - Single directory to be split into training and validation.
            info_dict (dict): Dictionary containing various configuration parameters:
        Returns:
            tuple: A tuple containing three TensorFlow datasets:
                   - train_dataset: Dataset for training.
                   - validation_dataset: Dataset for validation.
                   - test_dataset: Dataset for testing.
        """
        train_dataset = None
        validation_dataset = None
        if d_style == 1:
            if len(dir_list) == 2:
                train_dataset = tf.keras.utils.image_dataset_from_directory(
                    dir_list[0],
                    shuffle=True,
                    color_mode="grayscale" if info_dict["switch_mode"] == 5 else "rgb",
                    batch_size=info_dict["BATCH_SIZE"],
                    image_size=(info_dict["IMG_SIZE"], info_dict["IMG_SIZE"]),
                )
                validation_dataset = tf.keras.utils.image_dataset_from_directory(
                    dir_list[1],
                    shuffle=True,
                    color_mode="grayscale" if info_dict["switch_mode"] == 5 else "rgb",
                    batch_size=info_dict["BATCH_SIZE"],
                    image_size=(info_dict["IMG_SIZE"], info_dict["IMG_SIZE"]),
                )
            else:
                print(f"Please use train & validation as dir_list !! The length of dir_list is: {len(dir_list)}")
        elif d_style == 2:
            # split the raw dataset into train and val
            train_dataset = tf.keras.utils.image_dataset_from_directory(
                dir_list[0],
                color_mode="grayscale" if info_dict["switch_mode"] == 5 else "rgb",
                validation_split=info_dict["VAL_PCT"],
                subset="training",
                seed=123,
                batch_size=info_dict["BATCH_SIZE"],
                image_size=(info_dict["IMG_SIZE"], info_dict["IMG_SIZE"]),
            )
            validation_dataset = tf.keras.utils.image_dataset_from_directory(
                dir_list[0],
                color_mode="grayscale" if info_dict["switch_mode"] == 5 else "rgb",
                validation_split=info_dict["VAL_PCT"],
                subset="validation",
                seed=123,
                batch_size=info_dict["BATCH_SIZE"],
                image_size=(info_dict["IMG_SIZE"], info_dict["IMG_SIZE"]),
            )
        else:
            raise ValueError('The d_style must be 1 or 2, please check the dataset style.' + str(d_style))

        # Create a labels.txt to record the classes label
        cls_names = train_dataset.class_names
        txt_path = os.path.join(dir_list[0].split("train")[0], "labels.txt")
        with open(txt_path, "w", encoding='utf-8') as f:
            for lab in cls_names:
                f.write(lab + "\n")

        # take some validation percent for test data
        val_batches = tf.data.experimental.cardinality(validation_dataset)
        denominator = (int)(1 / info_dict["TEST_PCT"])
        testdata_batch_num = (val_batches // denominator).numpy()  # How many batches for test data

        # If the tfrecord files didn't exist or test dataset percent didn't match the args
        if not self.check_tfr_exist(dir_list[0].split("train")[0], testdata_batch_num, info_dict):
            # Save the val & test dataset as tfrecord, in this way we can record which test data is w/o training.
            self.create_tfrecord(validation_dataset, testdata_batch_num, dir_list[0].split("train")[0])

        validation_dataset = self.read_tfrecord("val_images.tfrecords", dir_list[0].split("train")[0], info_dict["switch_mode"])
        # Shuffle, batch, and prefetch the data for training or inference
        validation_dataset = validation_dataset.shuffle(buffer_size=10000)
        validation_dataset = validation_dataset.batch(batch_size=info_dict["BATCH_SIZE"])

        test_dataset = self.read_tfrecord("test_images.tfrecords", dir_list[0].split("train")[0], info_dict["switch_mode"])
        # Shuffle, batch, and prefetch the data for training or inference
        test_dataset = test_dataset.shuffle(buffer_size=10000)
        test_dataset = test_dataset.batch(batch_size=info_dict["BATCH_SIZE"])

        if info_dict["switch_mode"] == 1:
            print("validation dataset example:")
            val_n_batch = 0
            for images_val, labels_val in validation_dataset.take(1):
                plt.figure(figsize=(15, 15))
                val_n_batch = val_n_batch + 1
                img_idx = 0
                for im_show, l_show in zip(images_val, labels_val):
                    if img_idx > 31:
                        break
                    _ = plt.subplot(8, 4, img_idx + 1)
                    img_idx = img_idx + 1

                    plt.imshow(im_show.numpy().astype("uint8"))
                    plt.title(cls_names[l_show])
                    plt.axis("off")
                plt.show()

            print("test dataset example:")
            test_n_batch = 0
            for images_test, labels_test in test_dataset.take(1):
                plt.figure(figsize=(15, 15))
                test_n_batch = test_n_batch + 1
                img_idx = 0
                for im_show, l_show in zip(images_test, labels_test):
                    if img_idx > 31:
                        break
                    _ = plt.subplot(8, 4, img_idx + 1)
                    img_idx = img_idx + 1

                    plt.imshow(im_show.numpy().astype("uint8"))
                    plt.title(cls_names[l_show])
                    plt.axis("off")
                plt.show()

        # Loop the val/test batches
        val_n_batch = 0
        for _, _ in validation_dataset:
            val_n_batch = val_n_batch + 1

        test_n_batch = 0
        for _, _ in test_dataset:
            test_n_batch = test_n_batch + 1

        # Save the tfrecord dataset info for next time usage
        self.save_dataset_info_txt(dir_list[0].split("train")[0], info_dict, testdata_batch_num)

        print(f"Class names: {cls_names}")
        print(f"Number of train batches: {tf.data.experimental.cardinality(train_dataset)}")
        print(f"Number of validation batches: {val_n_batch}")
        print(f"Number of test batches: {test_n_batch}")
        print("\n")

        return train_dataset, validation_dataset, test_dataset

    def check_tfr_exist(self, save_path, testdata_batch_num, info_dict):
        """
        Checks if the TensorFlow Record (TFR) files and dataset information exist and match the provided parameters.
        Args:
            save_path (str): The directory path where the dataset information and TFR files are saved.
            testdata_batch_num (int): The expected number of test data batches.
            info_dict (dict): A dictionary containing dataset information with keys:
        Returns:
            bool: True if the dataset information and TFR files exist and match the provided parameters.
        """

        txt_path = os.path.join(save_path, "datasetInfo.txt")
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding='utf-8') as f:
                linex = f.readline()
                while linex is not None and linex != "" and linex != "\n" and linex != " ":
                    linex = f.readline()
                    if "TEST_BATCH_NUM" in linex:
                        if not linex.split(" ")[1].count(str(testdata_batch_num)):
                            print(f"testdata_batch_num is not the same {testdata_batch_num}")
                            return False
                    elif "IMG_SIZE" in linex:
                        if not linex.split(" ")[1].count(str(info_dict["IMG_SIZE"])):
                            print(f'IMG_SIZE is not the same {info_dict["IMG_SIZE"]}')
                            return False
                    elif "VAL_PCT" in linex:
                        if not linex.split(" ")[1].count(str(info_dict["VAL_PCT"])):
                            print(f'VAL_PCT is not the same {info_dict["VAL_PCT"]}')
                            return False
                    elif "TEST_PCT" in linex:
                        if not linex.split(" ")[1].count(str(info_dict["TEST_PCT"])):
                            print(f'TEST_PCT is not the same {info_dict["TEST_PCT"]}')
                            return False
                    elif "COLOR_MODE" in linex:
                        if not linex.split(" ")[1].count("grayscale" if info_dict["switch_mode"] == 5 else "rgb"):
                            print(f'COLOR_MODE is not the same {"grayscale" if info_dict["switch_mode"] == 5 else "rgb"}')
                            return False
        else:
            return False

        if os.path.exists(os.path.join(save_path, "val_images.tfrecords")) and os.path.exists(os.path.join(save_path, "test_images.tfrecords")):
            print("The tfrecords is the same! Skip creating.")
            return True

        return False

    def save_dataset_info_txt(self, save_path, info_dict, testdata_batch_num):
        """
        Save dataset information to a text file.
        """

        if info_dict["switch_mode"] == 5:
            color_mode = "grayscale"
        else:
            color_mode = "rgb"

        lines = [
            f"TEST_BATCH_NUM {testdata_batch_num}",
            f'IMG_SIZE {info_dict["IMG_SIZE"]}',
            f'VAL_PCT {info_dict["VAL_PCT"]}',
            f'TEST_PCT {info_dict["TEST_PCT"]}',
            f'COLOR_MODE {color_mode}',
        ]

        save_path = os.path.join(save_path, "datasetInfo.txt")
        with open(save_path, "w", encoding='utf-8') as f:
            for line in lines:
                f.write(line)
                f.write("\n")

    def read_tfrecord(self, record_file, save_path, infodict_switch_mode):
        """
        Reads a TFRecord file and parses its contents into a TensorFlow dataset.
        Args:
            record_file (str): The name of the TFRecord file to read.
            save_path (str): The directory path where the TFRecord file is located.
            infodict_switch_mode (int): A mode switch to determine the number of channels for the image.
                If `infodict_switch_mode` is 5, the image will be read with 1 channel (grayscale).
                Otherwise, the image will be read with 3 channels (RGB).
        Returns:
            tf.data.Dataset: A TensorFlow dataset containing the parsed images and labels.
        """

        tf_read_channel = 1 if infodict_switch_mode == 5 else 3

        # Define the feature description for your TFRecord file
        feature_description = {
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "depth": tf.io.FixedLenFeature([], tf.int64),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "image_raw": tf.io.FixedLenFeature([], tf.string),
        }

        # Define a function to parse each record in the TFRecord file
        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the feature description
            example = tf.io.parse_single_example(example_proto, feature_description)
            # Decode the JPEG-encoded image into a uint8 tensor
            image = tf.io.decode_jpeg(example["image_raw"], channels=tf_read_channel)
            # image = example['image_raw']
            image = tf.cast(image, tf.float32)
            # Convert the image tensor to float32 and normalize its values to [0, 1]
            # image = tf.image.convert_image_dtype(image, tf.float32)
            # Resize the image to your desired shape
            # image = tf.image.resize(image, [224, 224])
            # Map each label to a one-hot encoded vector
            # label = tf.one_hot(example['label'], depth=10)
            label = example["label"]
            return image, label

        # Define the file path(s) to your TFRecord file(s)
        save_path = [os.path.join(save_path, record_file)]

        # Create a `tf.data.TFRecordDataset` object to read the TFRecord file(s)
        dataset = tf.data.TFRecordDataset(save_path)

        # Map the `_parse_function` over each example in the dataset
        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    def create_tfrecord(self, validation_dataset, testdata_batch_num, save_path):
        """
        Creates TFRecord files for validation and test datasets.
        Args:
            validation_dataset (tf.data.Dataset): The dataset containing validation images and labels.
            testdata_batch_num (int): The number of batches to be used for the test dataset.
            save_path (str): The directory path where the TFRecord files will be saved.
        Returns:
            None
        """

        def _image_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def image_example(image_tensor, label):
            image_shape = image_tensor.shape
            image_tensor_raw = image_tensor.numpy()
            # image_tensor_raw = image_tensor.numpy().tobytes()
            # image_tensor_raw = image_tensor.numpy().astype("uint8")

            feature = {
                "height": _int64_feature(image_shape[0]),
                "width": _int64_feature(image_shape[1]),
                "depth": _int64_feature(image_shape[2]),
                "label": _int64_feature(label),
                "image_raw": _image_feature(image_tensor_raw),
            }

            return tf.train.Example(features=tf.train.Features(feature=feature))

        val_record_file = "val_images.tfrecords"
        val_record_file = os.path.join(save_path, val_record_file)
        test_record_file = "test_images.tfrecords"
        test_record_file = os.path.join(save_path, test_record_file)
        n_batches = 0
        with tf.io.TFRecordWriter(val_record_file) as writer_val:
            with tf.io.TFRecordWriter(test_record_file) as writer_test:
                for images_val, labels_val in validation_dataset:

                    n_batches = n_batches + 1
                    if n_batches <= testdata_batch_num:
                        for im_i, l_i in zip(images_val, labels_val):
                            tf_example = image_example(im_i, l_i)
                            writer_test.write(tf_example.SerializeToString())
                    else:
                        for im_i, l_i in zip(images_val, labels_val):
                            tf_example = image_example(im_i, l_i)
                            writer_val.write(tf_example.SerializeToString())
                        plt.show()

        print(f"Finish the Val  TFrecord Creating: {val_record_file}.")
        print(f"Finish the Test TFrecord Creating: {test_record_file}.")

    def _cal_macs(model_func):
        def wrap_cal_macs(self, *args, **kargs):
            model_func(self, *args, **kargs)

            forward_pass = tf.function(self.custom_model.call, input_signature=[tf.TensorSpec(shape=(1,) + self.custom_model.input_shape[1:])])

            graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())

            # The //2 is necessary since `profile` counts multiply and accumulate
            # as two flops, here we report the total number of multiply accumulate ops
            self.flops = graph_info.total_float_ops // 2
            print("TensorFlow:", tf.__version__)
            print(f"The MACs of this model: {self.flops:,}")

        return wrap_cal_macs

    @_cal_macs
    def _model_chooser(self, info_dict, class_len, dropout_rate):
        """
        Chooses and constructs a model based on the provided configuration.
        Parameters:
        - info_dict (dict): A dictionary containing model configuration parameters.
        - class_len (int): The number of output classes for the classification task.
        - dropout_rate (float): The dropout rate to be applied to the model.
        Returns:
        - None: The chosen model is assigned to self.custom_model.
        """

        # Rescale pixel values,
        # expects pixel values in [-1, 1] from [0, 255], or use
        # rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
        # preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        if info_dict["IMAGENET_MODEL_EN"] == 0:
            # Create the base model from the pre-trained model MobileNet V2 without the top layer
            img_shape = (info_dict["IMG_SIZE"], info_dict["IMG_SIZE"]) + (3,)

            if info_dict["MODEL_NAME"] == "mobilenet_v1":
                self.base_model = tf.keras.applications.MobileNet(input_shape=img_shape, include_top=False, weights="imagenet", alpha=info_dict["ALPHA_WIDTH"])
            elif info_dict["MODEL_NAME"] == "mobilenet_v2":
                self.base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights="imagenet", alpha=info_dict["ALPHA_WIDTH"])
            elif info_dict["MODEL_NAME"] == "mobilenet_v3_mini":
                self.base_model = tf.keras.applications.MobileNetV3Small(
                    input_shape=img_shape, alpha=info_dict["ALPHA_WIDTH"], include_top=False, weights="imagenet", minimalistic=True, include_preprocessing=False
                )
            elif info_dict["MODEL_NAME"] == "mobilenet_v3":
                self.base_model = tf.keras.applications.MobileNetV3Small(
                    input_shape=img_shape, alpha=info_dict["ALPHA_WIDTH"], include_top=False, weights="imagenet", minimalistic=False, include_preprocessing=False
                )
            elif info_dict["MODEL_NAME"] == "efficientnetB0":
                self.base_model = tf.keras.applications.EfficientNetB0(input_shape=img_shape, include_top=False, weights="imagenet")
            elif info_dict["MODEL_NAME"] == "efficientnetv2B0":
                self.base_model = tf.keras.applications.EfficientNetV2B0(input_shape=img_shape, include_top=False, include_preprocessing=True, weights="imagenet")

            # Feature extraction
            self.base_model.trainable = False

            # calculate the final size of AveragePooling2D
            fin_pool_size = 0
            if (info_dict["IMG_SIZE"] % 32) == 0:
                fin_pool_size = info_dict["IMG_SIZE"] / 32
            else:
                raise ValueError("The pooling size of final AveragePooling2D doesn't match with previous layer! \
                                 The IMG_SIZE must be multiple of 32!")

            # create the custom model
            inputs = tf.keras.Input(shape=(info_dict["IMG_SIZE"], info_dict["IMG_SIZE"], 3))
            # x = data_augmentation(inputs)
            x = inputs
            x = self.base_model(x, training=False)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
            outputs = tf.keras.layers.Dense(class_len, activation="softmax", use_bias=True, name="Logits")(x)

            # x = tf.keras.layers.AveragePooling2D(pool_size=(fin_pool_size, fin_pool_size), strides=(1, 1), padding="valid")(x)
            # x = tf.keras.layers.Dropout(0.2)(x)
            # # outputs = prediction_layer(x)
            # x = tf.keras.layers.Conv2D(class_len, (1, 1), padding="same")(x)
            # x = tf.reshape(x, [-1, class_len])
            # outputs = tf.keras.layers.Softmax()(x)

            self.custom_model = tf.keras.Model(inputs, outputs)

        elif info_dict["IMAGENET_MODEL_EN"] == 1:  # download the pretrain model only
            img_shape = (info_dict["IMG_SIZE"], info_dict["IMG_SIZE"]) + (3,)
            if info_dict["MODEL_NAME"] == "mobilenet_v1":
                self.base_model = tf.keras.applications.MobileNet(input_shape=img_shape, include_top=True, weights="imagenet", alpha=info_dict["ALPHA_WIDTH"])
            elif info_dict["MODEL_NAME"] == "mobilenet_v2":
                self.base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=True, weights="imagenet", alpha=info_dict["ALPHA_WIDTH"])
            elif info_dict["MODEL_NAME"] == "mobilenet_v3_mini":
                self.base_model = tf.keras.applications.MobileNetV3Small(
                    input_shape=img_shape, alpha=info_dict["ALPHA_WIDTH"], include_top=True, weights="imagenet", minimalistic=True, include_preprocessing=False
                )
            elif info_dict["MODEL_NAME"] == "mobilenet_v3":
                self.base_model = tf.keras.applications.MobileNetV3Small(
                    input_shape=img_shape, alpha=info_dict["ALPHA_WIDTH"], include_top=True, weights="imagenet", minimalistic=False, include_preprocessing=False
                )
            elif info_dict["MODEL_NAME"] == "efficientnetB0":
                self.base_model = tf.keras.applications.EfficientNetB0(input_shape=img_shape, include_top=True, weights="imagenet")
            elif info_dict["MODEL_NAME"] == "efficientnetv2B0":
                self.base_model = tf.keras.applications.EfficientNetV2B0(input_shape=img_shape, include_top=True, include_preprocessing=True, weights="imagenet")
            self.base_model.trainable = False
            self.custom_model = self.base_model

        else:
            # Create the model without pre-train model
            img_shape = (info_dict["IMG_SIZE"], info_dict["IMG_SIZE"]) + (3,)
            if info_dict["MODEL_NAME"] == "mobilenet_v1":
                self.custom_model = tf.keras.applications.MobileNet(input_shape=img_shape, include_top=True, weights=None, classes=class_len, alpha=info_dict["ALPHA_WIDTH"])
            if info_dict["MODEL_NAME"] == "mobilenet_v2":
                self.custom_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=True, weights=None, classes=class_len, alpha=info_dict["ALPHA_WIDTH"])
            # elif info_dict['MODEL_NAME'] == 'mobilenet_v3_mini':
            #    self.base_model = tf.keras.applications.MobileNetV3Small(input_shape = img_shape, alpha=info_dict['ALPHA_WIDTH'],
            #                                                           include_top=False, weights=None, classes=class_len, pooling = 'avg',
            #                                                           minimalistic=True, include_preprocessing=False)
            elif info_dict["MODEL_NAME"] == "mobilenet_v3_mini":
                self.custom_model = tf.keras.applications.MobileNetV3Small(
                    input_shape=img_shape, alpha=info_dict["ALPHA_WIDTH"], include_top=True, weights=None, classes=class_len, minimalistic=True, include_preprocessing=False
                )
            elif info_dict["MODEL_NAME"] == "mobilenet_v3":
                self.custom_model = tf.keras.applications.MobileNetV3Small(
                    input_shape=img_shape, alpha=info_dict["ALPHA_WIDTH"], include_top=True, weights=None, classes=class_len, minimalistic=False, include_preprocessing=False
                )
            if info_dict["MODEL_NAME"] == "efficientnetB0":
                self.custom_model = tf.keras.applications.EfficientNetB0(input_shape=img_shape, include_top=True, weights=None, classes=class_len)
            if info_dict["MODEL_NAME"] == "efficientnetv2B0":
                self.custom_model = tf.keras.applications.EfficientNetV2B0(input_shape=img_shape, include_preprocessing=True, include_top=True, weights=None, classes=class_len)

    def predict_topn(self, custom_model, dataset, topn=5):
        """
        Predict the top-N classes for each image in the dataset and calculate the top-N accuracy.
        Args:
            custom_model (tf.keras.Model): The trained model to use for predictions.
            dataset (tf.data.Dataset): The dataset to evaluate, containing batches of images and labels.
            topn (int, optional): The number of top predictions to consider for accuracy calculation. Defaults to 5.
        Returns:
            tuple: A tuple containing:
                - topn (int): The number of top predictions considered.
                - top_n_accuracy (float): The top-N accuracy as a fraction.
        """

        # calculate how many batches for tqdm
        num_batch = 0
        for x_batch, y_batch in dataset:
            num_batch += 1
        progress = tqdm(total=num_batch)

        # Evaluate the model on the test dataset and decode the top-N predictions
        num_correct = 0
        num_x_total = 0
        for x_batch, y_batch in dataset:

            preds = custom_model(x_batch, training=False).numpy()

            # Get the top 5 predictions for each image
            top_num = tf.math.top_k(preds, k=topn)
            # Get the indices of the top 5 predictions
            top_n_indices = top_num.indices.numpy()

            # Calculate the top-5 accuracy
            num_x_total += len(x_batch)
            for i in range(len(x_batch)):
                if y_batch[i] in top_n_indices[i]:
                    num_correct += 1
            progress.update(1)

        top_n_accuracy = num_correct / num_x_total
        print(f"Top-{topn} Accuracy: {(top_n_accuracy * 100):.2f}%")

        return topn, top_n_accuracy

    @tf.autograph.experimental.do_not_convert
    def preprocess_data_augmentation(self, dataset):
        """
        Applies data augmentation to the input dataset.
        This method uses TensorFlow's data augmentation layers to randomly flip and rotate images in the dataset.
        The augmented images are then prefetched for improved performance during training.
        Args:
            dataset (tf.data.Dataset): The input dataset containing image-label pairs.
        Returns:
            tf.data.Dataset: The augmented and prefetched dataset.
        """
        autotune = tf.data.AUTOTUNE

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", 127),
                tf.keras.layers.RandomRotation(0.2),
            ]
        )

        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=autotune)

        # Use buffered prefetching on all datasets.
        return dataset.prefetch(buffer_size=autotune)

    @tf.autograph.experimental.do_not_convert
    def normalization_mobilenetv2(self, dataset):
        """
        Applies normalization to the dataset for MobileNetV2.
        This function normalizes the pixel values of the images in the dataset
        from the range [0, 255] to the range [-1, 1], which is expected by the
        MobileNetV2 model.
        Args:
            dataset (tf.data.Dataset)
        Returns:
            tf.data.Dataset
        """
        # normalization for the data, expects pixel values in [-1, 1] from [0, 255]
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)
        out_dataset = dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return out_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def train(self, info_dict, train_dataset, validation_dataset, test_dataset):
        """
        Train the model using the provided datasets and configuration.
        Returns:
        None
        """

        # Get the class length
        class_len = len(train_dataset.class_names)

        # Configure the dataset for performance
        if info_dict["STEPS_PER_EPOCH"] == 0:  # Normal mode, 1 epoch = all batches in ds
            autotune = tf.data.AUTOTUNE
            train_dataset = train_dataset.prefetch(buffer_size=autotune)
            validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
            test_dataset = test_dataset.prefetch(buffer_size=autotune)
        else:  # Use 'STEPS_PER_EPOCH' mode, user can use cache(). Be careful the OOM if 'STEPS_PER_EPOCH' is too large.
            autotune = tf.data.AUTOTUNE
            train_dataset = train_dataset.cache().prefetch(buffer_size=autotune)
            validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)
            test_dataset = test_dataset.prefetch(buffer_size=autotune)

        # Use data augmentation
        train_dataset = self.preprocess_data_augmentation(train_dataset)  # val & test data no need.

        # normalization for the data, efficientnet has normalization layers in model.
        if not info_dict["MODEL_NAME"].count("efficientnet"):
            print("The dataset no need normalization bcs it is in model layer!")
            train_dataset = self.normalization_mobilenetv2(train_dataset)
            validation_dataset = self.normalization_mobilenetv2(validation_dataset)
        if info_dict["switch_mode"] == 1:
            image_batch, _ = next(iter(train_dataset))
            first_image = image_batch[0]
            print("The range of normalization:")
            print(np.min(first_image), np.max(first_image))

        # create the base pre-train model
        print("----Start to create model----")

        self._model_chooser(info_dict, class_len, 0.2)

        learning_rate_list = list(map(float, info_dict["LEARNING_RATE"].split(",")))
        self.custom_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_list[0]),
            # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.custom_model.summary()
        self.total_para = self.custom_model.count_params()

        # TF Board callback create
        self.tf_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.proj_path, "logs"))
        # check point callback create
        callbacks_chpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=(os.path.join(self.proj_path, "checkpoint", "{val_accuracy:.3f}_best_val.ckpt")),
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_freq="epoch",
        )
        callbacks_reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=10, min_delta=0.005, mode="max", cooldown=3)

        print(f"The trainable layers number: {len(self.custom_model.trainable_variables)}")
        print("The pretrain model result is as below (use validation dataset):")
        loss0, accuracy0 = self.custom_model.evaluate(validation_dataset)
        print(f"initial loss: {loss0:.2f}")
        print(f"initial accuracy: {accuracy0:.2f}")

        # Calculate the TopN result, normally is using Top5, in here is Top3
        if class_len > 10:
            self.predict_topn(self.custom_model, validation_dataset)

        # Train the custom_model with freezen weights
        if (info_dict["switch_mode"] >= 2) and ((info_dict["IMAGENET_MODEL_EN"] == 0) or (info_dict["IMAGENET_MODEL_EN"] == 2)):
            print("----Start to training----")

            epochs_list = list(map(int, info_dict["EPOCHS"].split(",")))

            if info_dict["STEPS_PER_EPOCH"] == 0:  # Normal mode, 1 epoch = all batches in ds
                history = self.custom_model.fit(train_dataset, verbose=1, epochs=epochs_list[0], validation_data=validation_dataset, callbacks=[self.tf_callback, callbacks_chpt, callbacks_reducelr])
            else:  # Use 'STEPS_PER_EPOCH' mode. Be careful the OOM if 'STEPS_PER_EPOCH' is too large.
                train_dataset = train_dataset.repeat()  # repeat the dataset, all training is steps_per_epoch * epochs.
                history = self.custom_model.fit(
                    train_dataset,
                    verbose=1,
                    epochs=epochs_list[0],
                    steps_per_epoch=info_dict["STEPS_PER_EPOCH"],
                    validation_data=validation_dataset,
                    callbacks=[self.tf_callback, callbacks_chpt, callbacks_reducelr],
                )

            # Show the train result
            acc = history.history["accuracy"]
            val_acc = history.history["val_accuracy"]
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            loc_dt = datetime.datetime.today()
            loc_dt_format = loc_dt.strftime("%Y_%m_%d_%H%M")

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label="Training Accuracy")
            plt.plot(val_acc, label="Validation Accuracy")
            plt.legend(loc="lower right")
            plt.ylabel("Accuracy")
            plt.ylim([0, 1])
            plt.title(f"Training and Validation Accuracy {loc_dt_format}")

            plt.subplot(2, 1, 2)
            plt.plot(loss, label="Training Loss")
            plt.plot(val_loss, label="Validation Loss")
            plt.legend(loc="upper right")
            plt.ylabel("Cross Entropy")
            plt.ylim([0, max(plt.ylim())])
            plt.title(f"Training and Validation Loss {loc_dt_format}")
            plt.xlabel("epoch")
            plt.savefig(os.path.join(self.proj_path, "result_plots", f"train_vali_{loc_dt_format}.png"))
            plt.show()

        # Fine tunning training
        if (info_dict["switch_mode"] == 3) and (info_dict["IMAGENET_MODEL_EN"] == 0):
            print("\n")
            print("----Start to fine tunning training----")
            self.fine_tunning(learning_rate_list, info_dict["FINE_TUNE_LAYER"])

            total_epochs = epochs_list[0] + epochs_list[1]

            if info_dict["STEPS_PER_EPOCH"] == 0:  # Normal mode, 1 epoch = all batches in ds
                history_fine = self.custom_model.fit(
                    train_dataset, verbose=1, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset, callbacks=[self.tf_callback, callbacks_chpt, callbacks_reducelr]
                )
            else:  # Use 'STEPS_PER_EPOCH' mode. Be careful the OOM if 'STEPS_PER_EPOCH' is too large.
                history_fine = self.custom_model.fit(
                    train_dataset,
                    verbose=1,
                    epochs=total_epochs,
                    initial_epoch=history.epoch[-1],
                    steps_per_epoch=info_dict["STEPS_PER_EPOCH"],
                    validation_data=validation_dataset,
                    callbacks=[self.tf_callback, callbacks_chpt, callbacks_reducelr],
                )
            # Show the train result
            acc += history_fine.history["accuracy"]
            val_acc += history_fine.history["val_accuracy"]
            loss += history_fine.history["loss"]
            val_loss += history_fine.history["val_loss"]

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label="Training Accuracy")
            plt.plot(val_acc, label="Validation Accuracy")
            plt.ylim([0, 1])
            plt.plot([epochs_list[0] - 1, epochs_list[0] - 1], plt.ylim(), label="Start Fine Tuning")
            plt.legend(loc="lower right")
            plt.title(f"Fine Training and Validation Accuracy {loc_dt_format}")

            plt.subplot(2, 1, 2)
            plt.plot(loss, label="Training Loss")
            plt.plot(val_loss, label="Validation Loss")
            plt.ylim([0, max(plt.ylim())])
            plt.plot([epochs_list[0] - 1, epochs_list[0] - 1], plt.ylim(), label="Start Fine Tuning")
            plt.legend(loc="upper right")
            plt.title(f"Fine Training and Validation Loss {loc_dt_format}")
            plt.xlabel("epoch")
            plt.savefig(os.path.join(self.proj_path, "result_plots", f"fine_train_vali_{loc_dt_format}.png"))  # fine tune plot
            plt.show()

        if (info_dict["switch_mode"] >= 2) and ((info_dict["IMAGENET_MODEL_EN"] == 0) or (info_dict["IMAGENET_MODEL_EN"] == 2)):
            # Test the model
            print("\n")
            print("----Start to Test the model----")
            if not info_dict["MODEL_NAME"].count("efficientnet"):
                test_dataset = self.normalization_mobilenetv2(test_dataset)
            loss, accuracy = self.custom_model.evaluate(test_dataset)
            print("Test accuracy :", accuracy)

            # Calculate the TopN result, normally is using Top5.
            top_n = 0
            top_n_accuracy = 0
            if class_len > 10:
                top_n, top_n_accuracy = self.predict_topn(self.custom_model, validation_dataset)

            # Save the test result
            test_txt_path = os.path.join(self.proj_path, "result_plots", f"fine_train_testAcc_{loc_dt_format}.txt")
            with open(test_txt_path, "w", encoding='utf-8') as f:
                f.write(f"Test accuracy: {accuracy}" + "\n")
                f.write(f"Top-{top_n} accuracy: {top_n_accuracy * 100}" + "\n")
                f.write(f"MACs: {self.flops}" + "\n")
                f.write(f"Total Parameters: {self.total_para}" + "\n")

            # Save as tflite
            print("\n")
            print("----Save as tflites----")
            self.convert2tflite(info_dict, train_dataset)

        if info_dict["IMAGENET_MODEL_EN"] == 1:
            print("\n")
            print("----No testing on ImageNet pretrain model----")
            print("----Save as tflites----")
            self.convert2tflite(info_dict, train_dataset)

            # Add a tflite_size into txt
            with open(test_txt_path, "a", encoding='utf-8') as f:
                f.write(f"The int8 QUANT tflite size: {self.int8_tflite_size}" + "\n")

    def fine_tunning(self, learning_rate_list, fine_tune_layer):
        """
        Fine-tunes the model by unfreezing the layers from a specified layer and re-compiling the model.
        Args:
            learning_rate_list (list): A list of learning rates to be used for different stages of training.
            fine_tune_layer (int): The index of the layer from which to start fine-tuning. All layers before this layer will be frozen.
        Returns:
            None
        """

        # Set the base_model as trainable
        self.base_model.trainable = True
        print("Number of layers in the base model: ", len(self.base_model.layers))

        fine_tune_at = fine_tune_layer
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # compile the fine tunning model
        self.custom_model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate_list[1]), metrics=["accuracy"])
        self.custom_model.summary()
        print(f"The trainable layers number: {len(self.custom_model.trainable_variables)}")

    def convert2tflite(self, info_dict, train_dataset):
        """
        Converts the trained Keras model to various TensorFlow Lite formats and saves them to the specified output location.
        """

        def representative_dataset():
            take_batch_num = 3
            idx = 0
            for img_train, _ in train_dataset.take(take_batch_num):
                idx = 0
                for i in range(info_dict["BATCH_SIZE"]):
                    idx = idx + 1
                    image = tf.expand_dims(img_train[i], axis=0)
                    # image = tf.dtypes.cast(image, tf.float32)
                    yield [image]  # total loop is take_batch_num * args.BATCH_SIZE

        # normal tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.custom_model)
        tflite_model = converter.convert()
        output_location = os.path.join(self.output_tflite_location, (info_dict["proj_name"] + r".tflite"))
        with open(output_location, "wb") as f:
            f.write(tflite_model)
        print(f"The tflite output location: {output_location}")

        # dynamic tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.custom_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        output_location = os.path.join(self.output_tflite_location, (info_dict["proj_name"] + r"_dyquant.tflite"))
        with open(output_location, "wb") as f:
            f.write(tflite_model)
        print(f"The tflite output location: {output_location}")

        # int8 Full tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.custom_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.representative_dataset = representative_dataset
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_model = converter.convert()
        output_location = os.path.join(self.output_tflite_location, (info_dict["proj_name"] + r"_int8quant.tflite"))
        with open(output_location, "wb") as f:
            f.write(tflite_model)
        print(f"The tflite output location: {output_location}")
        self.int8_tflite_size = os.path.getsize(output_location)

        # f16 tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.custom_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        # converter.representative_dataset = representative_dataset
        tflite_model = converter.convert()
        output_location = os.path.join(self.output_tflite_location, (info_dict["proj_name"] + r"_f16quant.tflite"))
        with open(output_location, "wb") as f:
            f.write(tflite_model)
        print(f"The tflite output location: {output_location}")

    def tflite_inference(self, input_dataset, tflite_path, info_dict, batch_n):
        """Call forwards pass of TFLite file and returns the result.

        Args:
            input_data: Input data to use on forward pass.
            tflite_path: Path to TFLite file to run.

        Returns:
            Output from inference.
        """

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_dtype = input_details[0]["dtype"]

        # Check if the input/output type is quantized,
        # set scale and zero-point accordingly
        if input_dtype == np.int8:
            input_scale, input_zero_point = input_details[0]["quantization"]

            def fun_cal(x, y):
                # return tf.math.round((x) / input_scale + input_zero_point), y
                return tf.math.round(x - 128), y

            input_dataset = input_dataset.map(fun_cal, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            input_scale, input_zero_point = 1, 0

            input_scale, input_zero_point = input_details[0]["quantization"]

            def fun_cal(x, y):
                return x / input_scale + input_zero_point, y

            input_dataset = input_dataset.map(fun_cal, num_parallel_calls=tf.data.AUTOTUNE)

        if input_dtype == np.int8:
            output_scale, output_zero_point = output_details[0]["quantization"]
        else:
            output_scale, output_zero_point = 1, 0

        progress = tqdm(total=info_dict["BATCH_SIZE"] * batch_n)
        acc_test_num = 0
        correct_test_num = 0
        for images_input, labels_input in input_dataset.take(batch_n):
            for im_input, l_input in zip(images_input, labels_input):

                input_data = tf.expand_dims(im_input, axis=0)

                interpreter.set_tensor(input_details[0]["index"], tf.cast(input_data, input_dtype))
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]["index"])
                output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

                acc_test_num += 1
                if tf.argmax(output_data, axis=1).numpy() == l_input.numpy():
                    correct_test_num += 1
                progress.update(1)

        print(f"Test accuracy of {acc_test_num} data : {(correct_test_num / acc_test_num) * 100}")
        return correct_test_num / acc_test_num


if __name__ == "__main__":

    def str2bool(v):
        """
        Convert a string representation of truth to boolean.
        Args:
            v (str or bool)
        Returns:
            bool: The boolean representation of the input string.
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_exist", type=str2bool, nargs="?", const=True, default=True, help="If data exist, skip the download")
    parser.add_argument("--url", type=str, default="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip", help="The download link")
    parser.add_argument("--zip_name", type=str, default="cats_and_dogs.zip", help="The name of download zip file")
    parser.add_argument("--dataset_name", type=str, default="cats_and_dogs_filtered", help="The name of dataset")

    parser.add_argument("--proj_name", type=str, default="catsdogs", help="The name of project which is used as workfolders name")
    parser.add_argument("--BATCH_SIZE", type=int, default=32, help="The batch size")
    parser.add_argument("--IMG_SIZE", type=int, default=160, help="The image size: (IMG_SIZE, IMG_SIZE)")
    parser.add_argument("--ALPHA_WIDTH", type=float, default=1.00, help="The test percentage from validation set, 0~0.5")
    parser.add_argument("--VAL_PCT", type=float, default=0.2, help="The validation percentage from all dataset, 0.1~0.5. This value will be skip if the dataset has already split into train & val.")

    parser.add_argument("--MODEL_NAME", type=str, default="mobilenet_v2", help="Choose the using model")
    parser.add_argument("--TEST_PCT", type=float, default=0.2, help="The test percentage from validation set, 0~0.5")
    parser.add_argument("--DATA_AUGM", type=str2bool, nargs="?", const=True, default=True, help="Use data augmentation or not")
    parser.add_argument("--EPOCHS", type=str, default="2,2", help="The training epochs.")
    parser.add_argument("--LEARNING_RATE", type=str, default="0.0001,0.00001", help="[The base lerning rate, The fine-tuned learning rate]")
    parser.add_argument("--FINE_TUNE_LAYER", type=int, default=100, help="Freeze all the layers before the `FINE_TUNE_LAYER` layer ")
    parser.add_argument("--STEPS_PER_EPOCH", type=int, default=0, help="How many steps per epoch. This is an alternative way to set the training steps. Each step has 1 batch. This is hided in UI.")
    parser.add_argument(
        "--switch_mode",
        type=int,
        default=1,
        help="1: Show the train data and model, \
              2: Transfer training, \
              3: Transfer and fine tuning training, \
              4: Test tflite",
    )
    parser.add_argument("--TFLITE_F", type=str, default="mobilenet_v2_int8quant.tflite", help="The name of tflite")
    parser.add_argument("--TFLITE_TEST_BATCH_N", type=int, default=1, help="How many batch for tflite test")
    parser.add_argument("--IMAGENET_MODEL_EN", type=int, default=0, help="0: Pretrain ImageNet weights. 1: Load full ImageNet weights model and without training. 2: No pretrain model.")

    args_parser = parser.parse_args()

    # manage the workspace
    my_workfdr = WORKFOLDER(args_parser.proj_name)
    my_workfdr.delete_logs_fdr()  # delete the previous log folder
    proj_path = my_workfdr.create_dirs()

    print("----Start to load data----")
    # get the dataset's path
    data_loader = data_prepare.DataLoader(args_parser.data_exist, args_parser.url, args_parser.zip_name, args_parser.dataset_name)
    train_task = TRAIN(proj_path)
    prepare_set_dict = train_task.prepare_setting(args_parser)
    # The d_style will check the dataset's type for futher spliting of val and train
    train_dataset_preload, validation_dataset_preload, test_dataset_preload = \
    train_task.data_pre_load(data_loader.dir_list, data_loader.d_style, prepare_set_dict)
    class_names = train_dataset_preload.class_names

    if prepare_set_dict["switch_mode"] == 1:
        print("train dataset example:")
        for images, labels in train_dataset_preload.take(1):
            plt.figure(figsize=(15, 15))
            X_IDX = 0
            for im, l in zip(images, labels):
                if X_IDX > 31:
                    break
                ax = plt.subplot(8, 4, X_IDX + 1)
                X_IDX = X_IDX + 1
                plt.imshow(im.numpy().astype("uint8"))
                plt.title(class_names[l])
                plt.axis("off")
            plt.show()

    if prepare_set_dict["switch_mode"] == 4:
        # output_location = os.path.join(train_task.output_tflite_location, args.TFLITE_F)
        print(f"Test tflite: {args_parser.TFLITE_F}")
        train_task.tflite_inference(test_dataset_preload, args_parser.TFLITE_F, prepare_set_dict, prepare_set_dict["TFLITE_TEST_BATCH_N"])
    else:
        # Start to prepare training
        train_task.train(prepare_set_dict, train_dataset_preload, validation_dataset_preload, test_dataset_preload)
