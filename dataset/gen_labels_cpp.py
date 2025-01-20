#!env/bin/python3

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
Utility script to convert a given text file with labels (annotations for an
NN model output vector) into a vector list initialiser. The intention is for
this script to be called as part of the build framework to auto-generate the
cpp file with labels that can be used in the application without modification.
"""
import datetime
import os
from argparse import ArgumentParser
from jinja2 import Environment, FileSystemLoader

parser = ArgumentParser()

# Label file path
parser.add_argument("--labels_dataset_name", type=str, help="Name of the label's dataset", required=True)
# Output file to be generated
parser.add_argument("--source_folder_path", type=str, help="path to source folder to be generated.", required=True)
parser.add_argument("--header_folder_path", type=str, help="path to header folder to be generated.", required=True)
parser.add_argument("--output_file_name", type=str, help="Required output file name", default="Labels")
# Namespaces
parser.add_argument("--namespaces", action='append', default=[])
# License template
parser.add_argument("--license_template", type=str, help="Header template file",
                    default="header_template.txt")

args_paser = parser.parse_args()

env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
                  trim_blocks=True,
                  lstrip_blocks=True)


def main(args):
    """
    Main function to generate C++ header and source files for labels.
    Args:
        args: Command line arguments containing the following attributes:
            - labels_dataset_name (str): The name of the dataset containing the labels.
            - license_template (str): The path to the license template file.
            - header_folder_path (str): The path to the folder where the header file will be saved.
            - output_file_name (str): The base name for the output files (without extension).
            - source_folder_path (str): The path to the folder where the source file will be saved.
            - namespaces (list): A list of namespaces to be used in the generated files.
    """
    # Get the labels from text file
    txt_path = os.path.join('datasets', args.labels_dataset_name, 'labels.txt') # The name of labels.txt is fixed.
    with open(txt_path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()

    # No labels?
    if len(labels) == 0:
        raise ValueError(f"No labels found in {txt_path}")

    header_template = env.get_template(args.license_template)
    hdr = header_template.render(script_name=os.path.basename(__file__),
                                 gen_time=datetime.datetime.now(),
                                 file_name=os.path.basename(txt_path),
                                 year=datetime.datetime.now().year)

    hpp_filename = os.path.join(args.header_folder_path, args.output_file_name + ".hpp")
    env.get_template('Labels.hpp.template').stream(common_template_header=hdr,
                                                   filename=(args.output_file_name).upper(),
                                                   namespaces=args.namespaces) \
        .dump(str(hpp_filename))


    cc_filename = os.path.join(args.source_folder_path, args.output_file_name + ".cc")
    env.get_template('Labels.cc.template').stream(common_template_header=hdr,
                                                  labels=labels,
                                                  labelsSize=len(labels),
                                                  namespaces=args.namespaces) \
        .dump(str(cc_filename))


if __name__ == '__main__':
    main(args_paser)
