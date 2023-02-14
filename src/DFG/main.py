import os

import detectron2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

setup_logger()
import torch
# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

# Our dataset is in coco format, so we can use the coco api to load it
register_coco_instances("my_dataset_train", {}, "../../../datasets/DFG/train.json", "../../../datasets/DFG/JPEGImages")
register_coco_instances("my_dataset_val", {}, "../../../datasets/DFG/test.json", "../../../datasets/DFG/JPEGImages")
# register_coco_instances("my_dataset_test", {}, "/content/test/_annotations.coco.json", "/content/test")

# visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

# import random
# from detectron2.utils.visualizer import Visualizer
#


def custom_config(num_classes, weight_path = None):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # Let training initialize from model zoo
    if weight_path is None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes


    cfg.SOLVER.IMS_PER_BATCH = 8 # Images per batch - basically on your Batch size
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.WARMUP_ITERS = 1000
    # cfg.SOLVER.STEPS = (100, 1000, 5000)
    # cfg.SOLVER.GAMMA = 0.05

    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.OUTPUT_DIR = './output/faster-rcnn/r50-fpn-3x/test2/'


    cfg.TEST.EVAL_PERIOD = 1000
    return cfg


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)



if __name__ == '__main__':
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # # trainer = DefaultTrainer(cfg)
    # # trainer.resume_or_load(resume=False)
    # # trainer.train()
    cfg = custom_config(200)
    # run_name = 'test1'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.build_evaluator(cfg, "my_dataset_val", output_folder=cfg.OUTPUT_DIR)
    trainer.resume_or_load(resume=False)
    trainer.train()
    # torch.save(trainer.model, cfg.OUTPUT_DIR  +'checkpoint.pth')
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)


    # Prediction
    # cfg = custom_config(200, weight_path=os.path.join('./output/', "model_final.pth"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    # my_dataset_test_metadata = MetadataCatalog.get("my_dataset_val")
    # dataset_dicts = DatasetCatalog.get("my_dataset_val")
    # predictor = DefaultPredictor(cfg)
    # for name in os.listdir('../../../datasets/DFG/model_testing'):
    #     im = cv2.imread(os.path.join('../../../datasets/DFG/model_testing', name))
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1],
    #                    metadata=my_dataset_test_metadata,
    #                    scale=0.5,
    #                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    #     )
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imwrite(os.path.join('./output/visualization/', name), v.get_image()[:, :, ::-1])
    # img = cv2.imread("../../../datasets/DFG/test_img.jpg")
    # outputs = predictor(img)
    # v = Visualizer(img[:, :, ::-1],
    #                metadata=my_dataset_train_metadata,
    #                scale=0.3,
    #                instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
    #                )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # print(outputs)
    # img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
    # cv2.imshow('test_img', out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR, 'visualization'), 'test_img' + '.png'), img)

