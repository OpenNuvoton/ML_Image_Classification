# ML_Image_Classification
- TensorFlow image classification with TFLite and Vela-TFLite, converting C/C++ source files.
## 1. First step
### 1. Install virtual env  
- If you haven't installed [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow these steps to install Python virtual environment and ***choose `NuEdgeWise_env`***.
- Skip if you have already done it.
### 2. Running
- The `classfication.ipynb notebook` will help you prepare data, train the model, and finally convert it to a TFLite and C++ file.

## 2. Work Flow
### 1. Data prepare
- Users can utilize `classfication.ipynb` to download easy datasets, prepare their custom datasets (or even download from other open-source platforms like Kaggle).
- `classfication.ipynb` will prepare the user's chosen dataset folder, supporting a general structure where the folder names correspond to class labels.

### 2. Training
- `classfication.ipynb` offers some attributes for training configuration.
- The strategy of this image classification training is [transfer learning & fine-tunning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- The output is tflite model.

### 3. Test
- Use `classfication.ipynb` to test the tflite model.

### 4. Deployment
- Utilize `classfication.ipynb` to convert the TFLite model to Vela and generate C source/header files.
- Also support Label source/header files converting.
- The `cmd.ipynb` notebook will demonstrate how to use the script located in `datasets\gen_rgb_cpp.py` to convert an image to a bytes source file.

## 3. Inference code
- MCU:
    - [M55M1](https://github.com/OpenNuvoton/M55M1BSP/tree/master/SampleCode/MachineLearning)
    - M460: Please contact Nuvoton
      
- MPU: [MA35D1](https://github.com/OpenNuvoton/MA35D1_Linux_Applications/tree/master/machine_learning)

