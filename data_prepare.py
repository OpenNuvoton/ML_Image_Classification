import numpy as np
import os
import tensorflow as tf


class DataLoader(object):
  """Loads data and prepares for training."""

  def __init__(self, data_exist, _URL, zip_name, dataset_name):
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
      self._URL = _URL
      self.zip_name = zip_name
      self.dataset_name = dataset_name
      self.dir_list = []
      self.d_style = 0
      
      if data_exist:
          # The default directory
          print("data exist!")
          dataset_PATH = os.path.join(os.getcwd(), "dataset", "datasets", self.dataset_name)
      else:
          # Download the dataset
          print("data downloading!")
          dataset_PATH = self.keras_url_download()

      # check dataset format, if == 1, we can directly use the dir path. if == 2, the dataset need be processed latter in train.py.    
      self.d_style = self.check_dataset_style(dataset_PATH)
      if self.d_style == 1: 
          # Get train/val datset path
          self.dir_list = self.get_exist_dataset_train_val(dataset_PATH)
      elif self.d_style == 2:
          self.dir_list.append(dataset_PATH)         
  
  def keras_url_download(self): 
      _cache_dir = os.path.join(os.getcwd(), "dataset") 
      path_to_dir = tf.keras.utils.get_file(self.zip_name, cache_dir = _cache_dir, origin=self._URL, extract=True)
      dataset_PATH = os.path.join(os.path.dirname(path_to_dir), self.dataset_name)
      
      return dataset_PATH
            
  def get_exist_dataset_train_val(self, dataset_PATH):
      train_dir = os.path.join(dataset_PATH, 'train')
      validation_dir = os.path.join(dataset_PATH, 'validation')
      
      return [train_dir, validation_dir]

  def check_dataset_style(self, dataset_PATH):
      
      if os.path.exists(os.path.join(dataset_PATH, "train")) and os.path.exists(os.path.join(dataset_PATH, "validation")):
          print("The dataset is style 1 with train and validation.")
          return 1
      else:
          print("The dataset is style 2 without train and validation.")
          return 2
  