# ML_tf2_image_classfication_nu
- Tensorflow image classfication with tflite and vela C/C++ source file converting.
- Also including [VWW](#VWW) (visual-wake-words) data prepare & training jupyter notebooks.
## 1. First step
### 1. Install virtual env  
- If you havn't install [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow the steps to install python virtual env and ***choose `NuEdgeWise_env`***.
- Skip if you have done.
### 2. Running
- The `classfication.ipynb` will help you prepare data, train the model, and finally convert to tflite & c++ file.

## 2. Work Flow
### 1. data prepare
- User can use `classfication.ipynb` to download some easy dataset or prepare their custom dataset(or maybe download from other opensource, ex: Kaggle)
- `classfication.ipynb` will prepare the user chosen dataset folder which support general structure(ex: the names of folders are class label)

### 2. training
- `classfication.ipynb` offers some attributes for training configuration.
- The strategy of this image classification training is [transfer learning & fine-tunning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- The output is tflite model.

### 3. Test
- Use `classfication.ipynb` to test the tflite model.

### 4. Deployment
- Use `classfication.ipynb` to convert tflite model to vela and C source/header files.
- Also support Label source/header files converting.
- Image convert to bytes source file: `cmd.ipynb` will show you how to use the script in `datasets\gen_rgb_cpp.py`

## 3. Inference code
- MCU: [M55+NPU](https://github.com/chchen59/M55A1BSP)  (not yet released)
- MPU: [MA35D1](https://github.com/OpenNuvoton/MA35D1_Linux_Applications/tree/master/machine_learning)

# VWW
- What is [VWW](https://paperswithcode.com/dataset/visual-wake-words).
- So far we offer python scripts for COCO data prepare & training.
- Steps/usage will update latter.
