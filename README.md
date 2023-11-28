# Waterfowl_detector_trainer

This code serves as a demo of showing how to train waterfowl detectors(Yolov5, YoloNAS and FasterRCNN) on our waterfowl pipelines.

## System requirements
Support is available only for the Linux Ubuntu system, and it can be used within a CUDA environment. It has been tested in Ubuntu 18 with Python 3.8.

## Installation

### Clone the repository
You can use the command line to clone the repository:
```
git clone https://github.com/YangZhangMizzou/Waterfowl_detector_trainer.git
```

### Create virtual environment
Using a virtual environment is recommended for installing the package. Anaconda is recommended for this purpose. After installing Anaconda, refer to this guide to create your virtual environment. It's recommended to create the environment with Python 3.8:

```
conda create -n torch_py3 python==3.8
conda activate torch_py3
cd Waterfowl_detector_pipeline_final
```

### Install pytorch

We recommend installing PyTorch with CUDA to accelerate running speed:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
### Install basic dependency

```
pip install pandas
pip install numpy
pip install opencv-python
pip install tqdm
pip install scikit-learn
```

### Install dependency for FasterRCNN

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Install dependency for YOLOV5

Just run this command and all set for YOLOV5
```
pip install -r requirements.txt
```

### Install dependency for YoloNAS

```
pip install super-gradients==3.1.3
pip install setuptools==59.5.0
pip install tensorboardX
pip install progress
```
## Dataset preparation

The datasets arrangement for yolonas, yolov5 and fasterrcnn are different. we prepare a sample dataset for each models. They can be downloaded from [here](https://drive.google.com/file/d/1ihQHkPfZ-IhSa5bmvoVmeuVo9GtKdd1r/view?usp=sharing). Please unzip them into folder dataset. 
If you want to prepare your customized dataset for training, check github repositories [yolonas](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md), [yolov5](https://github.com/ultralytics/yolov5) and [fasterrcnn](https://github.com/jwyang/faster-rcnn.pytorch) for details.

## Pretrained Weight

For fast model converge, we also provide pretrained weight of yolo5, yolonas and fasterrcnn which are pretrained by our 15m waterfowl detection datasets. Pretrained weight can be downloaded [here](https://drive.google.com/file/d/1zRMwYgwUnp5q0dnFEH_b0oTpglP3AJJn/view?usp=sharing). Please unzip them into folder pretrained_weights. 

## Run the Scripts:
After you finish the data preparation and the installation of our software, you can run the following commands to start training for yolonas, yolov5 and fasterrcnn. Training result will be saved in folder result.

Training for yolonas
```
python train_yolonas.py
```
Training for yolov5
```
python train_yolov5.py
```
Training for fasterrcnn
```
python train_fasterrcnn.py
```






