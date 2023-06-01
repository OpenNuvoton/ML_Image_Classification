import argparse
import tensorflow as tf
import pathlib
import os
import re
import numpy as np

import matplotlib.pyplot as plt

from collections import namedtuple
from tqdm.notebook import tqdm

from backbone import VWW

# Disable a lot of useless warnings
tf.get_logger().setLevel('ERROR')
ImageShape = namedtuple('ImageShape', 'height width channels')

parser = argparse.ArgumentParser(description="Train a model to predict whether an image contains a person")

parser.add_argument("--dataset", default="coco2017_vww", 
                    help="Name of dataset. Subdirectory of dataset/datasets")
parser.add_argument("--input-height", default=96, type=int, 
                    help="Height of input")
parser.add_argument("--input-width", default=96, type=int, 
                    help="Width of input")
parser.add_argument("--model-prefix", default="mobilenet",
                    help="Prefix to be used in naming the model")
parser.add_argument("--alpha", type=float, default=0.25,
                    help="Depth multiplier. The smaller it is, the smaller the resulting model.")
parser.add_argument("--epochs", type=int, default=10, 
                    help="Training procedure runs through the whole dataset once per epoch.")
parser.add_argument("--batch-size", type=int, default=256,
                    help="Number of examples to process concurrently")
parser.add_argument("--verbose", type=int, default=1,
                    help="Printing verbosity of Tensorflow model.fit()"
                    "Set --verbose=1 for per-batch progress bar, --verbose=2 for per-epoch")
parser.add_argument("--learning-rate", type=float, default=0.0001,
                    help="Initial learning rate of SGD training")
parser.add_argument("--decay-rate", type=float, default=0.98,
                    help="Number of steps to decay learning rate after")
parser.add_argument(
        '--switch_mode',
        type=int,
        default=1,
        help='1: Train model & Convert to tflite \
              2: Convert to tflite \
              3: Test tflite \
              ')
parser.add_argument('--COLOR_MODE',
        type=str,
        default='rgb',
        help='rgb, grayscale')
parser.add_argument(
        '--proj_name',
        type=str,
        default='vww_person',
        help='The name of project which is used as workfolders name')
parser.add_argument(
        '--logits',
        type=int,
        default=0,
        help='')

class train_vww():
  def __init__(self):
      self.backbone = tf.keras.Sequential([])
      self.model  = tf.keras.Sequential([]) 

  def _example_to_tensors(self, example, input_shape):
      """
      @brief: Read a serialized tf.train.Example and convert it to a (image, label) pair of tensors.
              TFRecords are created using src/create_coco_vww_tf_record.py 
      @author: Daniel Tan
      """
      example = tf.io.parse_example(
          example[tf.newaxis], {
              'image/encoded': tf.io.FixedLenFeature(shape = (), dtype=tf.string),
              'image/class': tf.io.FixedLenFeature(shape = (), dtype=tf.int64)
          })
      img_tensor =  tf.io.decode_jpeg(example['image/encoded'][0], channels=input_shape.channels)
      img_tensor = tf.image.resize(img_tensor, size=(input_shape.height, input_shape.width))
      #img_tensor.as_type('int8')
      label = example['image/class']
      return img_tensor, label
  
  def load_dataset(self, dataset_name, input_shape, color_mode, split="train"):
      """
      Parameters: 
          split: 'train' or 'val'
          dataset_name: A subdirector of data/vww_tfrecord
          input_shape: An ImageShape instance
          
      Return: 
          A dataset where each entry is a (image, label) tuple
      """
      
      datadir = pathlib.Path('dataset/datasets') / dataset_name
      filenames = [str(p) for p in datadir.glob(f"coco_{split}.record*")]
      tfrecords = tf.data.TFRecordDataset(filenames)
      def _map_fn(example):
          return self._example_to_tensors(example, input_shape)
      dataset = tfrecords.map(_map_fn)
      return dataset.filter(lambda x, y: tf.shape(x)[2] == input_shape.channels)
  
  def normalization(self, dataset, mode):
      """
      convert from float [0, 255]
      Parameters: 
          mode: 1 is mobilenetv1 or vww version => [0, 1]
                2 is mobilenetv2                => [-1, 1] 
      Return: 
          A dataset where each entry is a (image, label) tuple
      """
      # normalization for the data, expects pixel values in [-1, 1] from [0, 255]
      if mode == 1:
          normalization_layer =  tf.keras.layers.Rescaling(1./255.0)
          out_dataset = dataset.map(lambda x, y: ((x/255), y), num_parallel_calls=tf.data.AUTOTUNE)
         
      elif mode == 2:    
          normalization_layer =  tf.keras.layers.Rescaling(1./127.5, offset=-1)
          out_dataset = dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
          
      return out_dataset

  @tf.autograph.experimental.do_not_convert 
  def preprocess_data_augmentation(self, dataset):
       AUTOTUNE = tf.data.AUTOTUNE
       data_augmentation = tf.keras.Sequential([
       tf.keras.layers.RandomFlip('horizontal',127),
       tf.keras.layers.RandomRotation(0.2),
       tf.keras.layers.RandomZoom(0.2, 0.2)
       ])

       dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

       # Use buffered prefetching on all datasets.
       return dataset.prefetch(buffer_size=AUTOTUNE)    
      
  def build_model(self, input_shape, alpha, args):
      """
      Build a MobilenetV1 architecture with given input shape and alpha. 
      
      Parameters:
          input_shape: An ImageShape instance
          alpha: A float between 0 and 1. Model size scales with (alpha^2).
       
      Returns:
          A newly initialized model with the given architecture. 
      """
  
      input_shape = (input_shape.height, input_shape.width, input_shape.channels)
      
      if args.model_prefix == 'mobilenet':
          self.backbone = tf.keras.applications.MobileNet(
              input_shape = input_shape, alpha=alpha, include_top=False, weights=None, classes=2
          )
  
      elif args.model_prefix == 'mobilenetv2':
          if args.COLOR_MODE == 'rgb':
              self.backbone = tf.keras.applications.MobileNetV2(
              input_shape = input_shape, alpha=alpha, include_top=False, weights='imagenet', classes=2)
              self.backbone.trainable = False
          else:
              self.backbone = tf.keras.applications.MobileNetV2(
                  input_shape = input_shape, alpha=alpha, include_top=False, weights=None, classes=2)
      
      elif args.model_prefix == 'vww4':
          IMG_SHAPE = (args.input_height, args.input_width) + (1,) 
          vww_model = VWW(IMG_SHAPE)
          self.backbone = vww_model.vww_model   
      
      # Logits
      #classifier = tf.keras.Sequential(
      #    [tf.keras.layers.GlobalAveragePooling2D(),
      #     tf.keras.layers.Flatten(),
      #     tf.keras.layers.Dense(1, activation=None)]
      #)
  
      
      inputs = tf.keras.Input(input_shape)
      x = inputs
      if args.COLOR_MODE == 'rgb':
          x = self.backbone(x, training=False)
      else:    
          x = self.backbone(x)
      # Last layers
      x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
      x = tf.keras.layers.Dropout(0.2)(x)
      x = tf.keras.layers.Conv2D(2, (1, 1), padding="same")(x)
      x = tf.reshape(x,[-1, 2])
      outputs = tf.keras.layers.Softmax()(x)
      
      self.model = tf.keras.Model(inputs, outputs)
  
  def get_model_name(self, args):
      return f"vww_{args.model_prefix}_{args.alpha}_{args.input_height}_{args.input_width}"
  
  def get_checkpoint_dir(self, args):
      return f'workspace/{args.proj_name}/{self.get_model_name(args)}/best_val.ckpt'
  
  def get_model_dir(self, args):
      return f'workspace/{args.proj_name}/{self.get_model_name(args)}/saved_model'      
  
  def convert2tflite(self, args, train_dataset):
      def representative_dataset():
          take_batch_num = 3
          idx = 0
          for images, _ in train_dataset.take(take_batch_num):
              idx = 0
              for i in range(args.batch_size): 
                  idx = idx + 1
                  image = tf.expand_dims(images[i], axis=0)
                  #image = tf.dtypes.cast(image, tf.float32) 
                  yield [image] # total loop is take_batch_num * args.BATCH_SIZE
  
      loaded_model = self.find_best_ckpt(args, (pathlib.Path('workspace')/args.proj_name/self.get_model_name(args)), self.model)
      #loaded_model = tf.keras.models.load_model(get_model_dir(args))
      
      tflite_output = pathlib.Path('workspace')/ args.proj_name / 'tflite_model'
      
      # normal tflite
      converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
      tflite_model = converter.convert()
      output_location = pathlib.Path(tflite_output) / (self.get_model_name(args)+r'.tflite')
      with open(output_location, 'wb') as f:
            f.write(tflite_model)
      print("The tflite output location: {}".format(output_location)) 
      
      # dynamic tflite
      converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      tflite_model = converter.convert()
      output_location = pathlib.Path(tflite_output) / (self.get_model_name(args)+r'_dyquant.tflite')
      with open(output_location, 'wb') as f:
            f.write(tflite_model)
      print("The tflite output location: {}".format(output_location))
      
      # int8 Full tflite
      converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
      converter.representative_dataset = representative_dataset
      converter.inference_input_type = tf.int8  # or tf.uint8
      converter.inference_output_type = tf.int8  # or tf.uint8
      tflite_model = converter.convert()
      output_location = pathlib.Path(tflite_output) / (self.get_model_name(args)+r'_int8quant.tflite')
      with open(output_location, 'wb') as f:
            f.write(tflite_model)
      print("The tflite output location: {}".format(output_location)) 
      
      # f16 tflite
      converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.target_spec.supported_types = [tf.float16]
      #converter.representative_dataset = representative_dataset
      tflite_model = converter.convert()
      output_location = pathlib.Path(tflite_output) / (self.get_model_name(args)+r'_f16quant.tflite')
      with open(output_location, 'wb') as f:
            f.write(tflite_model)
      print("The tflite output location: {}".format(output_location))                     
  
  def find_best_ckpt(self, args, dir_path, model):
      
      # Define a regular expression pattern to match filenames with numbers
      pattern = re.compile(r"\d.\d+")
      
      # Initialize variables to keep track of the maximum number found
      max_number = None
      
      # Iterate over the files in the directory
      for file in dir_path.iterdir():
          # Extract the filename
          filename = file.name
  
          # Search for the number in the filename using the pattern
          match = pattern.search(filename)
  
          if match:
              # Get the matched number as a string
              number_str = match.group()
  
              # Convert the number to an integer
              number = float(number_str)
  
              # Update the maximum number if necessary
              if max_number is None or number > max_number:
                  max_number = number
      if max_number is None:
          print("There is no best ckpt in this work project.")  
      else:
          print("Find the best ckpt:{}".format(pathlib.Path(dir_path)/(number_str+f'_best_val.ckpt')))
          model.load_weights(str(pathlib.Path(dir_path)/(number_str+f'_best_val.ckpt')))      
      
      return model
  
  def tflite_test(self, args, val_dataset, tflite_path):
      
      # Create the truth & prediction metrix
      #expected_indices = np.concatenate([l for im, l in labels for x, labels in val_dataset])
      expected_indices = []
      predicted_indices = []
      ife_size = 0
      
      # Load the tflite model
      interpreter = tf.lite.Interpreter(model_path=tflite_path)
      interpreter.allocate_tensors()
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()
      input_dtype = input_details[0]["dtype"]
      output_dtype = output_details[0]["dtype"]
  
      # Check if the input/output type is quantized,
      # set scale and zero-point accordingly
      if input_dtype == np.int8:
          input_scale, input_zero_point = input_details[0]["quantization"]
          
          def fun_cal(x, y):
                return tf.math.round((x) / input_scale + input_zero_point), y
                #return tf.math.round(x - 128), y
          val_dataset = val_dataset.map(fun_cal, num_parallel_calls=tf.data.AUTOTUNE) 
      else:
          input_scale, input_zero_point = 1, 0
          
          def fun_cal(x, y):
                return (x) / input_scale + input_zero_point, y
          val_dataset = val_dataset.map(fun_cal, num_parallel_calls=tf.data.AUTOTUNE)
      
      if output_dtype == np.int8:
          output_scale, output_zero_point = output_details[0]["quantization"]
      else:
          output_scale, output_zero_point = 1, 0    
  
      
      print("Running val on test set...")
      for images, labels in tqdm(val_dataset):
              for input_im, l in zip(images, labels):
              
                  input_im = tf.expand_dims(input_im, axis=0)
                  
                  interpreter.set_tensor(input_details[0]['index'], tf.cast(input_im, input_dtype))
                  interpreter.invoke()
                  ife_size += 1
              
                  output_data = interpreter.get_tensor(output_details[0]['index'])
                  output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)
                  
                  # add the predict result to metrix
                  expected_indices.append(l.numpy()[0])
                  predicted_indices.append(np.squeeze(tf.argmax(output_data, axis=1).numpy()[0]))
      
      test_accuracy = self.calculate_accuracy(predicted_indices, expected_indices)
      confusion_matrix = tf.math.confusion_matrix(expected_indices, predicted_indices, num_classes=2)
  
      print(confusion_matrix.numpy())
      print(f'Test accuracy = {test_accuracy * 100:.2f}%'
            f'(N={ife_size})')
  
  def calculate_accuracy(self, predicted_indices, expected_indices):
      """Calculates and returns accuracy.
  
      Args:
          predicted_indices: List of predicted integer indices.
          expected_indices: List of expected integer indices.
  
      Returns:
          Accuracy value between 0 and 1.
      """
      correct_prediction = tf.equal(predicted_indices, expected_indices)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      return accuracy    

def main():

    my_train = train_vww()
   
    args = parser.parse_args()
    input_shape = ImageShape(height=args.input_height, width=args.input_width, channels= (1 if args.COLOR_MODE=='grayscale' else 3))
    CKPT_PATH = my_train.get_checkpoint_dir(args)
    
    my_train.build_model(input_shape, args.alpha, args)
    
    # If there is checkpt, load the weights.
    if (pathlib.Path('workspace')/args.proj_name/my_train.get_model_name(args)).exists():    
        print("Previous model folder found; loading saved weights")
        my_train.model = my_train.find_best_ckpt(args, (pathlib.Path('workspace')/args.proj_name/my_train.get_model_name(args)), my_train.model)
    else:
        print("No checkpoint found, create the workfolder.")
        (pathlib.Path('workspace')/args.proj_name/my_train.get_model_name(args)).mkdir(parents=True, exist_ok=False)
        (pathlib.Path('workspace')/args.proj_name/'tflite_model').mkdir(parents=False, exist_ok=False)
  
    # Load the data from tfrecord
    train_dataset = my_train.load_dataset(args.dataset, input_shape, args.COLOR_MODE, split="train").shuffle(1024).batch(args.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = my_train.load_dataset(args.dataset, input_shape, args.COLOR_MODE, split="val").shuffle(1024).batch(args.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Use data augmentation
    train_dataset = my_train.preprocess_data_augmentation(train_dataset) # val & test data no need.

    # normalization for the data
    train_dataset = my_train.normalization(train_dataset, 2).prefetch(buffer_size=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = my_train.normalization(val_dataset, 2).prefetch(buffer_size=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
    #    # Show the plts    
    #    for images, labels in train_dataset.take(1):
    #        plt.figure(figsize=(15, 15))
    #        x = 0
    #        for im, l in zip(images, labels):
    #            if x > 31:
    #                  break   
    #            ax = plt.subplot(8, 4, x + 1)
    #            x = x +1
    #            #print(im.numpy())
    #            plt.imshow(im.numpy().astype("uint8"), cmap='gray', vmin=0, vmax=255)
    #            #plt.title(class_names[l])
    #            plt.axis("off")      
    #        plt.show() 
    
    # train
    if args.switch_mode == 1:   
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_steps=100000,
            decay_rate=args.decay_rate,
            staircase=True)
        
        # Logits
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
        #     loss="sparse_categorical_crossentropy",
        #     #loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #     metrics=['accuracy'])
        
        my_train.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])
        
        callbacks_chpt = tf.keras.callbacks.ModelCheckpoint(
                filepath=(f'workspace/{args.proj_name}/{my_train.get_model_name(args)}/' + '{val_accuracy:.3f}_best_val.ckpt'),
                #filepath=CKPT_PATH, 
                save_weights_only=True, 
                monitor='val_accuracy',
                mode = 'max',
                save_best_only=True, 
                save_freq='epoch')    

        callbacks_tb = tf.keras.callbacks.TensorBoard(log_dir = f'workspace/{args.proj_name}/logs/')
        
        history = my_train.model.fit(x = train_dataset, validation_data = val_dataset, epochs=args.epochs, callbacks=[callbacks_chpt, callbacks_tb], verbose=args.verbose)

        if args.COLOR_MODE == 'rgb':
            # Set the base_model as trainable
            my_train.backbone.trainable = True
            print("Number of layers in the base model: ", len(my_train.backbone.layers))
            
            fine_tune_at = 100
            # Freeze all the layers before the `fine_tune_at` layer
            for layer in my_train.backbone.layers[:fine_tune_at]:
                layer.trainable = False     
            
            # compile the fine tunning model    
            my_train.model.compile(loss="sparse_categorical_crossentropy",
                      optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001),
                      metrics=['accuracy'])
            print("The trainable layers number: {}".format(len(my_train.model.trainable_variables)))

            fine_tune_epochs = 30
            total_epochs = args.epochs + fine_tune_epochs
            my_train.model.fit(train_dataset,
                        epochs=total_epochs,
                        initial_epoch=args.epochs,
                        validation_data=val_dataset,
                        callbacks=[callbacks_chpt, callbacks_tb]) 



    # Save the train model or the ckpt model    
    print(f"Model name: {my_train.get_model_name(args)}")
    my_train.model.save(my_train.get_model_dir(args))
    
    # convert to tflite
    if args.switch_mode <= 2:
        if (pathlib.Path('workspace')/args.proj_name/my_train.get_model_name(args)).exists():
            print("Load model to convert to tflite.")
            print(f"Model name: {my_train.get_model_name(args)}")
            my_train.convert2tflite(args, train_dataset)
        else:
            print("No model found! => {}".format((pathlib.Path('workspace')/args.proj_name/my_train.get_model_name(args))))
    
    # test tflite with val_dataset
    if args.switch_mode == 3:
        tflite_path = pathlib.Path('workspace')/ args.proj_name / 'tflite_model' / (my_train.get_model_name(args)+r'_int8quant.tflite')
        # tflite_path = pathlib.Path('workspace')/ args.proj_name / 'tflite_model' / 'vww2_50_50_INT8.tflite' 
        if pathlib.Path(tflite_path).exists():
            print("Test the tflite with validation dataset, {}".format(tflite_path))
            my_train.tflite_test(args, val_dataset, str(tflite_path))
        else:
           print("No tflite model found! => {}".format(tflite_path))            


    
if __name__ == "__main__":
    main()
    
    
    
        
