import subprocess
import os
from subprocess import Popen
import argparse

def update_bat_file(batch_file_path, var_update_list):
     
     # read the contents of the batch file
     with open(batch_file_path, 'r') as f:
         content = f.readlines()
     
     # loop through the lines of the batch file and update the variables
     for i in range(len(content)):
         if content[i].startswith('set MODEL_SRC_DIR='):
             content[i] = f'set MODEL_SRC_DIR={var_update_list[0]}\n'
         elif content[i].startswith('set MODEL_SRC_FILE='):
             content[i] = f'set MODEL_SRC_FILE={var_update_list[1]}\n'
         elif content[i].startswith('set MODEL_OPTIMISE_FILE='):
             content[i] = f'set MODEL_OPTIMISE_FILE={var_update_list[2]}\n'    
         elif content[i].startswith('set GEN_SRC_DIR='):
             content[i] = f'set GEN_SRC_DIR={var_update_list[3]}\n'    
     
     # write the updated content to the batch file
     with open(batch_file_path, 'w') as f:
         f.writelines(content)
     
     print('Batch file content updated successfully.')



if __name__ == "__main__":
  
  def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

  parser = argparse.ArgumentParser()
 
  parser.add_argument(
        '--SRC_DIR',
        type=str,
        default='..\\workspace\\catsdogs\\tflite_model',
        help='The model source dir')
  parser.add_argument(
        '--SRC_FILE',
        type=str,
        default='mobilenet_v2_int8quant.tflite',
        help='The model source file name')
  parser.add_argument(
        '--GEN_DIR',
        type=str,
        default='..\\workspace\\catsdogs\\tflite_model\\vela',
        help='The generated dir for *vela.tflite')            
  args = parser.parse_args()


  # Change to ../vela folder
  old_cwd = os.getcwd()
  batch_cwd = os.path.join(old_cwd, "vela")
  #print(batch_cwd)
  os.chdir(batch_cwd)
  
  # Get the MODEL_OPTIMISE_FILE
  if args.SRC_FILE.count(".tflite"):
      MODEL_OPTIMISE_FILE = args.SRC_FILE.split(".")[0] + f"_vela.tflite"
  else:
      raise OSError("Please input .tflite file!")    

  # Update the variables.bat
  batch_file_path = "variables.bat"
  var_update_list = []
  var_update_list.append(args.SRC_DIR)
  var_update_list.append(args.SRC_FILE)
  var_update_list.append(MODEL_OPTIMISE_FILE)
  var_update_list.append(args.GEN_DIR)
  update_bat_file(batch_file_path, var_update_list)

  
  # Execute the bat file
  print('Executing the {}.'.format(os.path.join(batch_cwd, "gen_model_cpp.bat")))
  print('Please wait...')
  p = Popen("gen_model_cpp.bat")
  stdout, stderr = p.communicate()
  #subprocess.call(["gen_model_cpp.bat"])
  
  os.chdir(old_cwd)
  vela_output_path = os.path.join(old_cwd, args.GEN_DIR.split("..\\")[1], MODEL_OPTIMISE_FILE) 
  print('Finish, the vela file is at: {}'.format(vela_output_path))