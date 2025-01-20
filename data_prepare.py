"""
This module provides a DataLoader class for loading and preparing datasets for training.
Classes:
    DataLoader: A class to handle downloading, organizing, and preparing datasets.
"""

import os
import tensorflow as tf


class DataLoader(object):
    """Loads data and prepares for training."""

    def __init__(self, data_exist, url, zip_name, dataset_name):
        """
        self.dir_list: A list saving the path of train & val dataset.
        self.d_style: The download or user customized dataset style.
          self.d_style == 1:
                              |--dataset
                                  |
                                  |-- train
                                       |-- ...
                                       |-- Label-N
                                  |
                                  |-- validation
                                       |-- ...
                                       |-- Label-N
          self.d_style == 2:
                              |--dataset
                                  |
                                  |-- Label-A
                                  |-- ...
                                  |-- Label-N

        """

        # load the dataset, from download or exist #
        self.url = url
        self.zip_name = zip_name
        self.dataset_name = dataset_name
        self.dir_list = []
        self.d_style = 0

        if data_exist:
            # The default directory
            print("data exist!")
            dataset_path = os.path.join(os.getcwd(), "dataset", "datasets", self.dataset_name)
        else:
            # Download the dataset
            print("data downloading!")
            dataset_path = self.keras_url_download()

        # check dataset format, if == 1, we can directly use the dir path. if == 2, the dataset need be processed latter in train.py.
        self.d_style = self.check_dataset_style(dataset_path)
        if self.d_style == 1:
            # Get train/val datset path
            self.dir_list = self.get_exist_dataset_train_val(dataset_path)
        elif self.d_style == 2:
            self.dir_list.append(dataset_path)

    def keras_url_download(self):
        """
        Downloads and extracts a dataset from a specified URL using Keras utilities.
        Returns:
            str: The path to the extracted dataset directory.
        """
        _cache_dir = os.path.join(os.getcwd(), "dataset")
        path_to_dir = tf.keras.utils.get_file(self.zip_name, cache_dir=_cache_dir, origin=self.url, extract=True)
        dataset_path = os.path.join(os.path.dirname(path_to_dir), self.dataset_name)

        return dataset_path

    def get_exist_dataset_train_val(self, dataset_path):
        """
        Retrieves the paths for the training and validation datasets.
        """
        train_dir = os.path.join(dataset_path, "train")
        validation_dir = os.path.join(dataset_path, "validation")

        return [train_dir, validation_dir]

    def check_dataset_style(self, dataset_path):
        """
        Check the style of the dataset based on the presence of 'train' and 'validation' directories.
        """

        if os.path.exists(os.path.join(dataset_path, "train")) and os.path.exists(os.path.join(dataset_path, "validation")):
            print("The dataset is style 1 with train and validation.")
            return 1
        else:
            print("The dataset is style 2 without train and validation.")
            return 2
