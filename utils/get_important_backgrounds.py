import glob
import shutil

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync

image_list = glob.glob('/home/yangzhang/yolov5/data/datasets/drone_collection/images/train_background/*.JPG')
model_dir = '/home/yangzhang/yolov5/runs/yolov5m/dr3/weights/best.pt'


def inference_mega_image_YOLO(image_list, model_dir):
    device = select_device('')
    model = DetectMultiBackend(model_dir, device=device, dnn=False, data='./configs/BirdA_all.yaml', fp16=False)
    stride, names, pt = model.stride, model.names, model.pt

    mega_imgae_id = 0
    bbox_id = 1
    all_annotations= []

    with tqdm(total = len(image_list)) as pbar:
        for idxs,image_dir in (enumerate(image_list)):
            pbar.update(1)
            start_time = time.time()
            bbox_list = []
            mega_imgae_id += 1
            mega_image  = cv2.imread(image_dir)
            ratio = 1.0
            im = np.expand_dims(mega_image, axis=0)
            im = np.transpose(im,(0,3,1,2))
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, 0.05, 0.2, None, False, max_det=1000)

            for i, det in enumerate(pred):
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        x1,y1,x2,y2 = xywh[0]-0.5*xywh[2], xywh[1]-0.5*xywh[3],xywh[0]+0.5*xywh[2], xywh[1]+0.5*xywh[3]  # (x1,y1, x2,y2)
                        if conf.cpu().numpy() > 0.5:
                            bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1, coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2,conf.cpu().numpy()])
            if bbox_list != []:
            	shutil.copy(image_dir,image_dir.replace('train_background','train'))

inference_mega_image_YOLO(image_list,model_dir)