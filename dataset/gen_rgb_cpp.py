#  Copyright (c) 2021 Arm Limited. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Utility script to convert a set of RGB images in a given location into
corresponding cpp files and a single hpp file referencing the vectors
from the cpp files.
"""
import datetime
import glob
import math
import os
import numpy as np

from argparse import ArgumentParser
from PIL import Image, UnidentifiedImageError
from jinja2 import Environment, FileSystemLoader

import tensorflow as tf

parser = ArgumentParser()
parser.add_argument("--image_path", type=str, help="path to images folder or image file  to convert.")
parser.add_argument("--source_folder_path", type=str, help="path to source folder to be generated.")
parser.add_argument("--header_folder_path", type=str, help="path to header folder to be generated.")
parser.add_argument("--image_size", type=int, nargs=2, help="Size (width and height) of the converted images.")
parser.add_argument("--license_template", type=str, help="Header template file",
                    default="header_template.txt")
parser.add_argument("--mode", type=int, nargs=1, help="0: image file mode. 1: tfrecord mode.",
                   default=0)                    
parser.add_argument("--tfrecord", type=str, help="read which tfrecords",
                    default="test_images.tfrecords")
parser.add_argument("--tfrd_num_cho", type=int, nargs=1, help="Choose how many plots to be converted to cpp.",
                   default=5)                                        
args = parser.parse_args()

env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
                  trim_blocks=True,
                  lstrip_blocks=True)


def write_hpp_file(header_file_path, cc_file_path, header_template_file, num_images, image_filenames,
                   image_array_names, image_size ,args):
    print(f"++ Generating {header_file_path}")
    header_template = env.get_template(header_template_file)
    hdr = header_template.render(script_name=os.path.basename(__file__),
                                 gen_time=datetime.datetime.now(),
                                 year=datetime.datetime.now().year)
    if args.mode == 0:                             
        env.get_template('Images.hpp.template').stream(common_template_header=hdr,
                                                       imgs_count=num_images,
                                                       img_size=str(image_size[0] * image_size[1] * 3),
                                                       var_names=image_array_names) \
            .dump(str(header_file_path))
    else:
        env.get_template('Images.hpp.template').stream(common_template_header=hdr,
                                                       imgs_count=num_images,
                                                       img_size=str(image_size * image_size * 3),
                                                       var_names=image_array_names) \
            .dump(str(header_file_path))        

    env.get_template('Images.cc.template').stream(common_template_header=hdr,
                                                  var_names=image_array_names,
                                                  img_names=image_filenames) \
        .dump(str(cc_file_path))


def write_individual_img_cc_file(image_filename, cc_filename, header_template_file, original_image,
                                 image_size, array_name, args):
    print(f"++ Converting {image_filename} to {os.path.basename(cc_filename)}")

    header_template = env.get_template(header_template_file)
    hdr = header_template.render(script_name=os.path.basename(__file__),
                                 gen_time=datetime.datetime.now(),
                                 file_name=os.path.basename(image_filename),
                                 year=datetime.datetime.now().year)
    
    if args.mode == 0:
       original_image.thumbnail(image_size)
       delta_w = abs(image_size[0] - original_image.size[0])
       delta_h = abs(image_size[1] - original_image.size[1])
       resized_image = Image.new('RGB', args.image_size, (255, 255, 255, 0))
       resized_image.paste(original_image, (int(delta_w / 2), int(delta_h / 2)))
    else:
       resized_image = original_image
    
    #resized_image.save("test.png","png")

    # Convert the image and write it to the cc file
    rgb_data = np.array(resized_image, dtype=np.uint8).flatten()
    hex_line_generator = (', '.join(map(hex, sub_arr))
                          for sub_arr in np.array_split(rgb_data, math.ceil(len(rgb_data) / 20)))
    env.get_template('image.cc.template').stream(common_template_header=hdr,
                                                 var_name=array_name,
                                                 img_data=hex_line_generator) \
        .dump(str(cc_filename))

def main_read_tfrecord(args):
    # Define the feature description for your TFRecord file
      feature_description = {
          'height': tf.io.FixedLenFeature([], tf.int64),
          'width': tf.io.FixedLenFeature([], tf.int64),
          'depth': tf.io.FixedLenFeature([], tf.int64),
          'label': tf.io.FixedLenFeature([], tf.int64),
          'image_raw': tf.io.FixedLenFeature([], tf.string),   
      }
      
      tf_image_size = 0

      # Define a function to parse each record in the TFRecord file
      def _parse_function(example_proto):
          # Parse the input `tf.train.Example` proto using the feature description
          example = tf.io.parse_single_example(example_proto, feature_description)
          # Decode the JPEG-encoded image into a uint8 tensor
          image = tf.io.decode_jpeg(example['image_raw'], channels=3)
          #image = example['image_raw']
          image = tf.cast(image, tf.float32)
         
          label = example['label']
          
          return image, label

      def _get_image_size(example_proto):
          example = tf.io.parse_single_example(example_proto, feature_description)
          return example['width']

      tfrecord_file = os.path.join(args.image_path, args.tfrecord)
      #print(tfrecord_file)

      if os.path.isfile(tfrecord_file):
        filepaths = [args.image_path]
      else:
        raise OSError("File does not exist.")

      
      # Define the file path(s) to your TFRecord file(s)
      save_path = [tfrecord_file]
      
      # Create a `tf.data.TFRecordDataset` object to read the TFRecord file(s)
      dataset = tf.data.TFRecordDataset(save_path)

      # Get the image_size from width
      tf_image_size_dataset =dataset.map(_get_image_size)
      for s in tf_image_size_dataset:
        tf_image_size = s.numpy()
        break
      print(tf_image_size)
      
      # Map the `_parse_function` over each example in the dataset
      dataset = dataset.map(_parse_function)
      dataset = dataset.shuffle(buffer_size=10000)

      # Process the tfrecord dataset to C source files.

      # Keep the count of the images converted
      image_idx = 0
      image_filenames = []
      image_array_names = []

      for image, label in dataset:
          if image_idx >= args.tfrd_num_cho: # user choose number
            break

          filename = str(image_idx) + "_" + str(label.numpy())
          
          image_filenames.append(filename)
          original_image = image.numpy()
  
          # Save the cc file
          cc_filename = os.path.join(args.source_folder_path,
                                     filename + ".cc")
          array_name = "im" + str(image_idx)
          image_array_names.append(array_name)
          write_individual_img_cc_file(filename, cc_filename, args.license_template,
                                       original_image, tf_image_size, array_name, args)
  
          # Increment image index
          image_idx = image_idx + 1
  
      header_filename = "InputFiles.hpp"
      header_filepath = os.path.join(args.header_folder_path, header_filename)
      common_cc_filename = "InputFiles.cc"
      common_cc_filepath = os.path.join(args.source_folder_path, common_cc_filename)
  
      if len(image_filenames) > 0:
          write_hpp_file(header_filepath, common_cc_filepath, args.license_template,
                      image_idx, image_filenames, image_array_names, tf_image_size, args)
      else:
          raise FileNotFoundError("No valid images found.")

      

def main(args):
    # Keep the count of the images converted
    image_idx = 0
    image_filenames = []
    image_array_names = []


    if os.path.isdir(args.image_path):
        filepaths = sorted(glob.glob(os.path.join(args.image_path, '**/*.*'), recursive=True))
    elif os.path.isfile(args.image_path):
        filepaths = [args.image_path]
    else:
        raise OSError("Directory or file does not exist.")

    for filepath in filepaths:
        filename = os.path.basename(filepath)

        try:
            original_image = Image.open(filepath).convert("RGB")
        except UnidentifiedImageError:
            print(f"-- Skipping file {filepath} due to unsupported image format.")
            continue

        image_filenames.append(filename)

        # Save the cc file
        cc_filename = os.path.join(args.source_folder_path,
                                   (filename.rsplit(".")[0]).replace(" ", "_") + ".cc")
        array_name = "im" + str(image_idx)
        image_array_names.append(array_name)
        write_individual_img_cc_file(filename, cc_filename, args.license_template,
                                     original_image, args.image_size, array_name, args)

        # Increment image index
        image_idx = image_idx + 1

    header_filename = "InputFiles.hpp"
    header_filepath = os.path.join(args.header_folder_path, header_filename)
    common_cc_filename = "InputFiles.cc"
    common_cc_filepath = os.path.join(args.source_folder_path, common_cc_filename)

    if len(image_filenames) > 0:
        write_hpp_file(header_filepath, common_cc_filepath, args.license_template,
                    image_idx, image_filenames, image_array_names, args.image_size, args)
    else:
        raise FileNotFoundError("No valid images found.")


if __name__ == '__main__':
    if args.mode == 0:
        main(args)
    else:
       main_read_tfrecord(args)     
