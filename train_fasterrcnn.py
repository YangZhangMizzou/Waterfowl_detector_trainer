import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
from PIL import Image  
import PIL
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances

I_train_dir = './dataset/BirdI_faster/train'
I_test_dir =  './dataset/BirdI_faster/test'
register_coco_instances("I_train", {}, I_train_dir+"/bird.json", I_train_dir)
register_coco_instances("I_test", {}, I_test_dir+"/bird.json", I_test_dir)

birds_metadata = MetadataCatalog.get("train")
birds_metadata.thing_classes = ['bird']


from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.data import build_detection_test_loader, build_detection_train_loader
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "D3")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        # if cfg.MODEL.DENSEPOSE_ON:
        #     evaluators.append(DensePoseCOCOEvaluator(dataset_name, True, output_folder))
        return DatasetEvaluators(evaluators)
    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

cfg = get_cfg()
cfg.merge_from_file("./data/faster.yaml")
cfg.MODEL.WEIGHTS = "./pretrained_weights/fasterrcnn/best.pth"
cfg.OUTPUT_DIR = './result/fasterrcnn/birdI'
cfg.DATASETS.TRAIN = ("I_train",)   # no metrics implemented for this dataset = ("decoy_summer_train",)   # no metrics implemented for this dataset
cfg.DATASETS.TEST = ()
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32],[64],[128]]
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
cfg.SOLVER.WARMUP_ITERS = 100
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.STEPS = (30000,45000)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.SOLVER.MAX_ITER = 60000  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()
