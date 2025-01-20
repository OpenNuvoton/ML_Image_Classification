# Reference from
# https://www.tensorflow.org/tutorials/images/transfer_learning
'''
This script is designed for training and evaluating image classification models using TensorFlow 2 and tf2cv. 
It includes functionalities for data preparation, model training, evaluation, and conversion to TensorFlow Lite format.
Classes:
    WORKFOLDER:
        Manages the creation and deletion of project directories.
    TRAIN:
        Handles the training, evaluation, and conversion of models.
Main Execution:
    Parses command-line arguments and initiates the training or evaluation process based on the provided settings.
'''

# https://www.tensorflow.org/tutorials/images/transfer_learning
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
from tf2cv.model_provider import get_model as tf2cv_get_model

import data_prepare

class WORKFOLDER:
    """
    A class to manage project directories and logs for a machine learning project.
    """
    def __init__(self, proj_name):
        self.proj_path = os.path.join(os.getcwd(), "workspace", proj_name)

    def create_dirs(self):
        """
        Creates necessary directories for the project if they do not already exist.
        This method checks if the main project directory and its subdirectories 
        ("result_plots", "tflite_model", "checkpoint", and "logs") exist. If any 
        of these directories do not exist, it creates them.
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
        This method attempts to remove the 'logs' directory and its contents. If any files or directories
        within 'logs' are read-only, their permissions are changed to allow deletion.
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
    A class used to train a TensorFlow model for image classification. Also includes methods for data preparation,
    tfrecord creation, and model evaluation.
    '''
    def __init__(self, proj_path):
        self.proj_path = proj_path
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
        Pre-loads and processes image datasets for training, validation, and testing.
        Args:
            dir_list (list): List of directories containing the image datasets. 
            d_style (int): Dataset style. 
            info_dict (dict): Dictionary containing configuration parameters:
        Returns:
            tuple: A tuple containing:
                   - train_dataset (tf.data.Dataset): The training dataset.
                   - validation_dataset (tf.data.Dataset): The validation dataset.
                   - test_dataset (tf.data.Dataset): The test dataset.
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
            print("The d_style must be 1 or 2, please check the dataset style.")

        # Create a labels.txt to record the classes label
        class_names = train_dataset.class_names
        txt_path = os.path.join(dir_list[0].split("train")[0], "labels.txt")
        with open(txt_path, "w", encoding='utf-8') as f:
            for l in class_names:
                f.write(l + "\n")

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
            for images, labels in validation_dataset.take(1):
                plt.figure(figsize=(15, 15))
                val_n_batch = val_n_batch + 1
                x = 0
                for im, l in zip(images, labels):
                    if x > 31:
                        break
                    _ = plt.subplot(8, 4, x + 1)
                    x = x + 1
                    plt.imshow(im.numpy().astype("uint8"))
                    plt.title(class_names[l])
                    plt.axis("off")
                plt.show()

            print("test dataset example:")
            test_n_batch = 0
            for images, labels in test_dataset.take(1):
                plt.figure(figsize=(15, 15))
                test_n_batch = test_n_batch + 1
                x = 0
                for im, l in zip(images, labels):
                    if x > 31:
                        break
                    _ = plt.subplot(8, 4, x + 1)
                    x = x + 1
                    # print(im.numpy())
                    plt.imshow(im.numpy().astype("uint8"))
                    plt.title(class_names[l])
                    plt.axis("off")
                plt.show()

        # Loop the val/test batches
        val_n_batch = 0
        for images, labels in validation_dataset:
            val_n_batch = val_n_batch + 1

        test_n_batch = 0
        for images, labels in test_dataset:
            test_n_batch = test_n_batch + 1

        # Save the tfrecord dataset info for next time usage
        self.save_dataset_info_txt(dir_list[0].split("train")[0], info_dict, testdata_batch_num)

        print(f"Class names: {class_names}")
        print(f"Number of train batches: {tf.data.experimental.cardinality(train_dataset)}")
        print(f"Number of validation batches: {val_n_batch}")
        print(f"Number of test batches: {test_n_batch}")
        print("\n")

        return train_dataset, validation_dataset, test_dataset

    def check_tfr_exist(self, save_path, testdata_batch_num, info_dict):
        """
        Checks if the TensorFlow Record (TFR) files and dataset information exist and match the given parameters.
        Args:
            save_path (str): The path where the dataset information and TFR files are saved.
            testdata_batch_num (int): The number of test data batches.
            info_dict (dict): A dictionary containing dataset information with keys:
                - "IMG_SIZE" (int): The size of the images.
                - "VAL_PCT" (float): The percentage of validation data.
                - "TEST_PCT" (float): The percentage of test data.
                - "switch_mode" (int): The mode for color (5 for grayscale, otherwise RGB).
        Returns:
            bool: True if the dataset information and TFR files exist and match the given parameters, False otherwise.
        """

        txt_path = os.path.join(save_path, "datasetInfo.txt")
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding='utf-8') as f:
                x = f.readline()
                while x is not None and x != "" and x != "\n" and x != " ":
                    x = f.readline()
                    if "TEST_BATCH_NUM" in x:
                        if not x.split(" ")[1].count(str(testdata_batch_num)):
                            print(f"testdata_batch_num is not the same {testdata_batch_num}")
                            return False
                    elif "IMG_SIZE" in x:
                        if not x.split(" ")[1].count(str(info_dict["IMG_SIZE"])):
                            print(f'IMG_SIZE is not the same {info_dict["IMG_SIZE"]}')
                            return False
                    elif "VAL_PCT" in x:
                        if not x.split(" ")[1].count(str(info_dict["VAL_PCT"])):
                            print(f'VAL_PCT is not the same {info_dict["VAL_PCT"]}')
                            return False
                    elif "TEST_PCT" in x:
                        if not x.split(" ")[1].count(str(info_dict["TEST_PCT"])):
                            print(f'TEST_PCT is not the same {info_dict["TEST_PCT"]}')
                            return False
                    elif "COLOR_MODE" in x:
                        if not x.split(" ")[1].count("grayscale" if info_dict["switch_mode"] == 5 else "rgb"):
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
        Args:
            save_path (str): The directory path where the dataset information file will be saved.
            info_dict (dict): A dictionary containing dataset information with the following keys:
                - "switch_mode" (int): Mode to determine color mode (5 for grayscale, otherwise rgb).
                - "IMG_SIZE" (int): The size of the images in the dataset.
                - "VAL_PCT" (float): The percentage of the dataset used for validation.
                - "TEST_PCT" (float): The percentage of the dataset used for testing.
            testdata_batch_num (int): The number of test data batches.
        Returns:
            None
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
            f"COLOR_MODE {color_mode}",
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
                for images, labels in validation_dataset:

                    n_batches = n_batches + 1
                    if n_batches <= testdata_batch_num:
                        for im, l in zip(images, labels):
                            tf_example = image_example(im, l)
                            writer_test.write(tf_example.SerializeToString())

                    else:
                        for im, l in zip(images, labels):
                            tf_example = image_example(im, l)
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
        # Rescale pixel values,
        # expects pixel values in [-1, 1] from [0, 255], or use
        # rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

        def _unwrap_tf2cv_fdmobilenet(tf2cv_model):
            tf2cv_model.trainable = False
            # Be careful, this is basing on the structure of fdmobilenet from tf2cv
            print(f"The total layers number: {len(tf2cv_model.layers)}")
            base_model = tf2cv_model.layers[0]
            print(f"The total children layers number: {len(base_model.children)}")

            # Change the AveragePooling2D to the GlobalAveragePooling2D for sutiable for all kernal size
            base_model.children[5] = tf.keras.layers.GlobalAveragePooling2D()

            inp = tf.keras.Input(shape=(info_dict["IMG_SIZE"], info_dict["IMG_SIZE"], 3))
            x = inp
            training = None
            x = base_model(x, training=training)
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
            x = tf.keras.layers.Dense(class_len, activation="softmax", use_bias=True, name="Logits")(x)

            return tf.keras.Model(inp, x)

        def unwrap_tf2cv_shufflenet(tf2cv_model):
            tf2cv_model.trainable = False
            # Be careful, this is basing on the structure of shufflenet from tf2cv
            print(f"The total layers number: {len(tf2cv_model.layers)}")
            base_model = tf2cv_model.layers[0]
            print(f"The total children layers number: {len(base_model.children)}")
            print(f"The total children layers number of children 1: {len(base_model.children[1].children)}")
            print(f"The total children layers number of children 2: {len(base_model.children[2].children)}")
            print(f"The total children layers number of children 3: {len(base_model.children[3].children)}")

            # Change the AveragePooling2D to the GlobalAveragePooling2D for sutiable for all kernal size
            base_model.children[4] = tf.keras.layers.GlobalAveragePooling2D()

            inp = tf.keras.Input(shape=(info_dict["IMG_SIZE"], info_dict["IMG_SIZE"], 3))
            x = inp
            training = None
            x = base_model(x, training=training)
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
            x = tf.keras.layers.Dense(class_len, activation="softmax", use_bias=True, name="Logits")(x)
            # method 2
            # x = flatten(x, "channels_last")
            # output1 = nn.Dense(units=1000,input_dim=3,name="output1")
            # x = output1(x)
            # method 3
            # x = tf.keras.layers.Conv2D(1000, (1, 1), padding="same")(x)
            # x = tf.reshape(x,[-1,1000])
            # x = tf.keras.layers.Softmax()(x)
            return tf.keras.Model(inp, x)

        if info_dict["IMAGENET_MODEL_EN"] == 0:
            # Create the base model from the fdmobilenet without the top layer

            if info_dict["MODEL_NAME"] == "fdmobilenet_wd4":
                net = tf2cv_get_model("fdmobilenet_wd4", pretrained=True, data_format="channels_last")
                self.custom_model = _unwrap_tf2cv_fdmobilenet(net)
            if info_dict["MODEL_NAME"] == "fdmobilenet_wd2":
                net = tf2cv_get_model("fdmobilenet_wd2", pretrained=True, data_format="channels_last")
                self.custom_model = _unwrap_tf2cv_fdmobilenet(net)
            if info_dict["MODEL_NAME"] == "fdmobilenet_w1":
                net = tf2cv_get_model("fdmobilenet_w1", pretrained=True, data_format="channels_last")
                self.custom_model = _unwrap_tf2cv_fdmobilenet(net)

            if info_dict["MODEL_NAME"] == "shufflenet_g1_wd4":
                net = tf2cv_get_model("shufflenet_g1_wd4", pretrained=True, data_format="channels_last")
                self.custom_model = unwrap_tf2cv_shufflenet(net)
            if info_dict["MODEL_NAME"] == "shufflenet_g3_wd4":
                net = tf2cv_get_model("shufflenet_g3_wd4", pretrained=True, data_format="channels_last")
                self.custom_model = unwrap_tf2cv_shufflenet(net)
            if info_dict["MODEL_NAME"] == "shufflenet_g1_wd2":
                net = tf2cv_get_model("shufflenet_g1_wd2", pretrained=True, data_format="channels_last")
                self.custom_model = unwrap_tf2cv_shufflenet(net)
            if info_dict["MODEL_NAME"] == "shufflenet_g3_wd2":
                net = tf2cv_get_model("shufflenet_g3_wd2", pretrained=True, data_format="channels_last")
                self.custom_model = unwrap_tf2cv_shufflenet(net)

        else:  # download the pretrain model only
            print("This function is TODO!")

    def predict_top_n(self, custom_model, dataset, top_n=5):
        """
        Predict the top-N class probabilities for a given dataset using a custom model.
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
            top_num = tf.math.top_k(preds, k=top_n)
            # Get the indices of the top 5 predictions
            top_n_indices = top_num.indices.numpy()

            # Calculate the top-5 accuracy
            num_x_total += len(x_batch)
            for i in range(len(x_batch)):
                if y_batch[i] in top_n_indices[i]:
                    num_correct += 1
            progress.update(1)

        top_n_accuracy = num_correct / num_x_total
        print(f"Top-{top_n} Accuracy: {top_n_accuracy * 100:.2f}%")

        return top_n, top_n_accuracy

    @tf.autograph.experimental.do_not_convert
    def preprocess_data_augmentation(self, dataset):
        """
        Applies data augmentation to the given dataset using TensorFlow's data augmentation layers.
        Args:
            dataset (tf.data.Dataset): The input dataset containing image-label pairs.
        Returns:
            tf.data.Dataset
        """
        autotune = tf.data.AUTOTUNE

        myseed = 29
        data_augmentation = tf.keras.Sequential(
            [
                # tf.keras.layers.RandomFlip('horizontal', myseed),
                tf.keras.layers.RandomRotation(0.2, seed=myseed),
                tf.keras.layers.RandomContrast(0.3, myseed),
                tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=myseed),
                # tf.keras.layers.RandomCrop(128, 128, seed=myseed)
            ]
        )

        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=autotune)

        # Use buffered prefetching on all datasets.

        return dataset.prefetch(buffer_size=autotune)

    @tf.autograph.experimental.do_not_convert
    def normalization_mobilenetv2(self, dataset):
        """
        Normalize the dataset for MobileNetV2.
        This function normalizes the pixel values of the input dataset from the range [0, 255] to the range [-1, 1],
        which is the expected input range for the MobileNetV2 model.
        Args:
            dataset (tf.data.Dataset): The input dataset containing image and label pairs.
        Returns:
            tf.data.Dataset: The normalized dataset with pixel values in the range [-1, 1].
        """
        # normalization for the data, expects pixel values in [-1, 1] from [0, 255]
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)
        out_dataset = dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return out_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def train(self, info_dict, train_dataset, validation_dataset, test_dataset):
        """
        Train the model using the provided datasets and configuration.
        Args:
            info_dict (dict): Dictionary containing training configuration parameters.
            train_dataset (tf.data.Dataset): Dataset for training.
            validation_dataset (tf.data.Dataset): Dataset for validation.
            test_dataset (tf.data.Dataset): Dataset for testing.
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

        # normalization for the data
        train_dataset = self.normalization_mobilenetv2(train_dataset)
        validation_dataset = self.normalization_mobilenetv2(validation_dataset)
        if info_dict["switch_mode"] == 1:
            image_batch, _ = next(iter(train_dataset)) # image_batch, labels_batch
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
        # check point callback create, tf2cv can't use Checkpoint
        # callbacks_chpt = tf.keras.callbacks.ModelCheckpoint(
        #          filepath=(os.path.join(self.proj_path, 'checkpoint', '{val_accuracy:.3f}_best_val.ckpt')),
        #          #save_weights_only=True,
        #          monitor='val_accuracy',
        #          mode = 'max',
        #          save_best_only=True,
        #          save_freq='epoch')
        callbacks_reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=10, min_delta=0.005, mode="max", cooldown=3)

        print(f"The trainable layers number: {len(self.custom_model.trainable_variables)}")
        print("The pretrain model result is as below (use validation dataset):")
        loss0, accuracy0 = self.custom_model.evaluate(validation_dataset)
        print(f"initial loss: {loss0:.2f}")
        print(f"initial accuracy: {accuracy0:.2f}")

        # Calculate the TopN result, normally is using Top5, in here is Top3
        if class_len > 10:
            self.predict_top_n(self.custom_model, validation_dataset)

        # Train the custom_model with freezen weights
        if (info_dict["switch_mode"] >= 2) and (info_dict["IMAGENET_MODEL_EN"] == 0):
            print("----Start to training----")

            epochs_list = list(map(int, info_dict["EPOCHS"].split(",")))

            if info_dict["STEPS_PER_EPOCH"] == 0:  # Normal mode, 1 epoch = all batches in ds
                history = self.custom_model.fit(train_dataset, verbose=1, epochs=epochs_list[0], validation_data=validation_dataset, callbacks=[self.tf_callback, callbacks_reducelr])
            else:  # Use 'STEPS_PER_EPOCH' mode. Be careful the OOM if 'STEPS_PER_EPOCH' is too large.
                train_dataset = train_dataset.repeat()  # repeat the dataset, all training is steps_per_epoch * epochs.
                history = self.custom_model.fit(
                    train_dataset, verbose=1, epochs=epochs_list[0], steps_per_epoch=info_dict["STEPS_PER_EPOCH"], validation_data=validation_dataset, callbacks=[self.tf_callback, callbacks_reducelr]
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
            self.fine_tunning(learning_rate_list, info_dict["FINE_TUNE_LAYER"], info_dict["MODEL_NAME"])

            total_epochs = epochs_list[0] + epochs_list[1]

            if info_dict["STEPS_PER_EPOCH"] == 0:  # Normal mode, 1 epoch = all batches in ds
                history_fine = self.custom_model.fit(
                    train_dataset, verbose=1, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset, callbacks=[self.tf_callback, callbacks_reducelr]
                )
            else:  # Use 'STEPS_PER_EPOCH' mode. Be careful the OOM if 'STEPS_PER_EPOCH' is too large.
                history_fine = self.custom_model.fit(
                    train_dataset,
                    verbose=1,
                    epochs=total_epochs,
                    initial_epoch=history.epoch[-1],
                    steps_per_epoch=info_dict["STEPS_PER_EPOCH"],
                    validation_data=validation_dataset,
                    callbacks=[self.tf_callback, callbacks_reducelr],
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

        if (info_dict["switch_mode"] >= 2) and (info_dict["IMAGENET_MODEL_EN"] == 0):
            # Test the model
            print("\n")
            print("----Start to Test the model----")
            test_dataset = self.normalization_mobilenetv2(test_dataset)
            loss, accuracy = self.custom_model.evaluate(test_dataset)
            print("Test accuracy :", accuracy)

            # Calculate the TopN result, normally is using Top5.
            top_n = 0
            top_n_accuracy = 0
            if class_len > 10:
                top_n, top_n_accuracy = self.predict_top_n(self.custom_model, validation_dataset)

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

    def fine_tunning(self, learning_rate_list, fine_tune_layer, choose_model_name):
        """
        Fine-tunes the model by setting specific layers as trainable or non-trainable based on the chosen model name and fine-tune layer.
        Parameters:
        learning_rate_list (list): A list of learning rates to be used for compiling the model.
        fine_tune_layer (int): The layer number up to which layers should be frozen (non-trainable).
        choose_model_name (str): The name of the model to be fine-tuned. Supports "fdmobilenet" and "shufflenet".
        Returns:
        None
        """

        self.custom_model.trainable = True

        if choose_model_name.count("fdmobilenet"):
            # Set the freeze layers from the beginning

            if fine_tune_layer == 10:
                for tf2cv_layer in self.custom_model.layers[1].children[0:5]:  # The layers are 0~3. The first layer is input
                    tf2cv_layer.trainable = False
            elif fine_tune_layer > 4 and fine_tune_layer <= 9:
                for tf2cv_layer in self.custom_model.layers[1].children[0:4]:  # The layers are 0~3. The first layer is input
                    tf2cv_layer.trainable = False
                for block in self.custom_model.layers[1].children[4][0 : (fine_tune_layer - 5)]:  # The 5th child is the major parts which has 6 layers
                    block.trainable = False
            elif fine_tune_layer > 0 and fine_tune_layer <= 4:
                for tf2cv_layer in self.custom_model.layers[1].children[0 : (fine_tune_layer - 1)]:  # The layers are 0~3. The first layer is input
                    tf2cv_layer.trainable = False
            elif fine_tune_layer == 0:
                print("No freezing layers. The whole model is trainable!!")
            else:
                print("ERROR, the Freezing Layers of Fine-Tuning shold be in 0~17")

        elif choose_model_name.count("shufflenet"):
            # Set the freeze layers from the beginning
            if fine_tune_layer == 17:
                print("Only last layer is trainable!!")
                self.custom_model.layers[1].children[0].trainable = False  # 1 layer
                self.custom_model.layers[1].children[1].trainable = False  # 4 layers
                self.custom_model.layers[1].children[2].trainable = False  # 8 layers
                self.custom_model.layers[1].children[3].trainable = False  # 4 layers
            elif fine_tune_layer > 12 and fine_tune_layer <= 16:
                self.custom_model.layers[1].children[0].trainable = False
                self.custom_model.layers[1].children[1].trainable = False
                self.custom_model.layers[1].children[2].trainable = False
                for tf2cv_layer in self.custom_model.layers[1].children[3].children[0 : (fine_tune_layer - 13)]:  # The layers are 1~3. 4 8 4
                    tf2cv_layer.trainable = False
            elif fine_tune_layer > 4 and fine_tune_layer <= 12:
                self.custom_model.layers[1].children[0].trainable = False
                self.custom_model.layers[1].children[1].trainable = False
                for tf2cv_layer in self.custom_model.layers[1].children[2].children[0 : (fine_tune_layer - 5)]:  # The layers are 1~3. 4 8 4
                    tf2cv_layer.trainable = False
            elif fine_tune_layer > 0 and fine_tune_layer <= 4:
                self.custom_model.layers[1].children[0].trainable = False
                for tf2cv_layer in self.custom_model.layers[1].children[1].children[0 : (fine_tune_layer - 1)]:  # The layers are 1~3. 4 8 4
                    tf2cv_layer.trainable = False
            elif fine_tune_layer == 0:
                print("No freezing layers. The whole model is trainable!!")
            else:
                print("ERROR, the Freezing Layers of Fine-Tuning shold be in 0~17")

        # compile the fine tunning model
        self.custom_model.compile(
            loss="sparse_categorical_crossentropy",
            # optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_list[1]),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_list[1]),
            metrics=["accuracy"],
        )
        self.custom_model.summary()
        print(f"The trainable layers number: {len(self.custom_model.trainable_variables)}")

    def convert2tflite(self, info_dict, train_dataset):
        """
        Converts the Keras model to various TensorFlow Lite formats and saves them to the specified output location.
        The converted models are saved to the output location specified by `self.output_tflite_location` with appropriate filenames.
        """

        def representative_dataset():
            take_batch_num = 3
            idx = 0
            for images, _ in train_dataset.take(take_batch_num):
                idx = 0
                for i in range(info_dict["BATCH_SIZE"]):
                    idx = idx + 1
                    image = tf.expand_dims(images[i], axis=0)
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
        # output_dtype = output_details[0]["dtype"]

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
        for images, labels in input_dataset.take(batch_n):
            for im, l in zip(images, labels):

                input_data = tf.expand_dims(im, axis=0)

                interpreter.set_tensor(input_details[0]["index"], tf.cast(input_data, input_dtype))
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]["index"])
                output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

                acc_test_num += 1
                if tf.argmax(output_data, axis=1).numpy() == l.numpy():
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

    parser.add_argument("--MODEL_NAME", type=str, default="fdmobilenet_wd2", help="Choose the using model")
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
              3: (Not Work Now)Transfer and fine tuning training, \
              4: Test tflite",
    )
    parser.add_argument("--TFLITE_F", type=str, default="mobilenet_v2_int8quant.tflite", help="The name of tflite")
    parser.add_argument("--TFLITE_TEST_BATCH_N", type=int, default=1, help="How many batch for tflite test")
    parser.add_argument("--IMAGENET_MODEL_EN", type=int, default=0, help="Load full ImageNet weights model and without training.")

    args_parser = parser.parse_args()

    # manage the workspace
    my_workfdr = WORKFOLDER(args_parser.proj_name)
    my_workfdr.delete_logs_fdr()  # delete the previous log folder
    proj_path_workfolder = my_workfdr.create_dirs()

    print("----Start to load data----")
    # get the dataset's path
    data_loader = data_prepare.DataLoader(args_parser.data_exist, args_parser.url, args_parser.zip_name, args_parser.dataset_name)
    train_task = TRAIN(proj_path_workfolder)
    train_task_info_dict = train_task.prepare_setting(args_parser)

    # The d_style will check the dataset's type for futher spliting of val and train
    train_dataset_preload, validation_dataset_preload, test_dataset_preload = train_task.data_pre_load(data_loader.dir_list, data_loader.d_style, train_task_info_dict)
    train_class_names = train_dataset_preload.class_names

    if train_task_info_dict["switch_mode"] == 1:
        print("train dataset example:")
        for images_train, labels_train in train_dataset_preload.take(1):
            plt.figure(figsize=(15, 15))
            PLT_IDX = 0
            for image_exp, label_exp in zip(images_train, labels_train):
                if PLT_IDX > 31:
                    break
                _ = plt.subplot(8, 4, PLT_IDX + 1)
                PLT_IDX = PLT_IDX + 1
                plt.imshow(image_exp.numpy().astype("uint8"))
                plt.title(train_class_names[label_exp])
                plt.axis("off")
            plt.show()

    if train_task_info_dict["switch_mode"] == 4:
        # output_location = os.path.join(train_task.output_tflite_location, args.TFLITE_F)
        print(f"Test tflite: {args_parser.TFLITE_F}")
        train_task.tflite_inference(test_dataset_preload, args_parser.TFLITE_F, train_task_info_dict, train_task_info_dict["TFLITE_TEST_BATCH_N"])
    else:
        # Start to prepare training
        train_task.train(train_task_info_dict, train_dataset_preload, validation_dataset_preload, test_dataset_preload)
