import glob
import os
import shutil

all_images_list = glob.glob('/home/yangzhang/yolov5/data/datasets/drone_collection/images/train_all/*.JPG')
all_label_list = glob.glob('/home/yangzhang/yolov5/data/datasets/drone_collection/labels/train/*.txt')
for image_dir in all_images_list:
	if image_dir not in all_label_list:
		shutil.copy(image_dir,image_dir.replace('train_all','train_background'))