from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback


dataset_params = {
    'data_dir':'./dataset/BirdI_yolonas',
    'train_images_dir':'train',
    'train_labels_dir':'train_anno',
    'val_images_dir':'test',
    'val_labels_dir':'test_anno',
    'test_images_dir':'test',
    'test_labels_dir':'test_anno',
    'classes': ['Bird']
}

train_params = {
    # ENABLING SILENT MODE
    'silent_mode': True,
    "average_best_models":False,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-8,
    "lr_warmup_epochs": 1,
    "initial_lr": 5e-8,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.05,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": False,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 100,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.3,
            top_k_predictions=500,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=500,
                nms_threshold=0.3
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}


train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':4,
        'num_workers':1
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':8,
        'num_workers':1
    }
)

trainer = Trainer(experiment_name='birdI', ckpt_root_dir='./result/yolonas')
model = models.get('yolo_nas_m', num_classes=len(dataset_params['classes']), checkpoint_path='./pretrained_weights/yolonas/best.pth').cuda()
trainer.train(model=model, training_params=train_params, train_loader=train_data, valid_loader=val_data)