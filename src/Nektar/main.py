import os

import detectron2
import gaps_dataset.gaps
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

setup_logger()
import cv2
import random
import matplotlib.pyplot as plt
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
register_coco_instances("my_dataset_train", {},
                        "../../../datasets/Yellowstone ATS/Annotations/combined_annotations_train.json",
                        "../../../datasets/Yellowstone ATS/Images/train")
register_coco_instances("my_dataset_val", {},
                        "../../../datasets/Yellowstone ATS/Annotations/combined_annotations_test.json",
                        "../../../datasets/Yellowstone ATS/Images/test")

# register_coco_instances("my_dataset_test", {}, "/content/test/_annotations.coco.json", "/content/test")
print(os.getcwd())
# visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")


# import random
# from detectron2.utils.visualizer import Visualizer


def custom_config(num_classes, weight_path=None):
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

    cfg.SOLVER.IMS_PER_BATCH = 8  # Images per batch - basically on your Batch size
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 50000
    cfg.SOLVER.WARMUP_ITERS = 2000
    # cfg.SOLVER.STEPS = (100, 1000, 5000)
    # cfg.SOLVER.GAMMA = 0.05

    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.OUTPUT_DIR = './output/faster-rcnn/r50-fpn-3x/DSPS23/test1/'

    cfg.TEST.EVAL_PERIOD = 4000
    return cfg


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def train_model(config, change_default_output_dir=False, general_output_dir='./output/', output_dir=None,
                run_name=None):
    cfg = config
    if change_default_output_dir:
        cfg.OUTPUT_DIR = os.path.join(general_output_dir, output_dir)
    if run_name is not None:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, run_name)
    else:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'test' + str(os.listdir(cfg.OUTPUT_DIR).__len__()) + '/')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.build_evaluator(cfg, "my_dataset_val", output_folder=cfg.OUTPUT_DIR)
    trainer.resume_or_load(resume=False)
    trainer.train()


def prediction(num_classes=8, num_images=10, model_path=None, model_name=None, output_dir=None, show_flag=False):
    cfg = custom_config(num_classes, weight_path=os.path.join(model_path, model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    my_dataset_test_metadata = MetadataCatalog.get("my_dataset_val")
    dataset_dicts = DatasetCatalog.get("my_dataset_val")
    if output_dir is None:
        output_dir = os.path.join(model_path, 'images/')
    os.makedirs(output_dir, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    for i, d in enumerate(random.sample(dataset_dicts, num_images)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=my_dataset_test_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(20, 30))

        plt.imshow(v.get_image()[:, :, ::-1])
        plt.savefig(os.path.join(output_dir, f'prediction{i}.png'))
        if show_flag:
            plt.show()


def prediction_new_images(num_classes=8, num_images=10, model_path=None, data_path=None, model_name=None,
                          output_dir=None, show_flag=False):
    cfg = custom_config(num_classes, weight_path=os.path.join(model_path, model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    my_dataset_test_metadata = MetadataCatalog.get("my_dataset_val")
    if output_dir is None:
        output_dir = os.path.join(model_path, 'images/')
    os.makedirs(output_dir, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    for i,name in enumerate(os.listdir(data_path)):
        im = cv2.imread(os.path.join(data_path, name))
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=my_dataset_test_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(20, 30))

        plt.imshow(v.get_image()[:, :, ::-1])
        plt.savefig(os.path.join(output_dir, f'prediction{i}.png'))
        if show_flag:
            plt.show()


if __name__ == '__main__':
    config = custom_config(7)
    run_name = 'test1'
    # train_model(config, change_default_output_dir=True, general_output_dir='./output/',
    #             output_dir='faster-rcnn/r50-fpn-3x/Nektar/')

    # Prediction

    # prediction(num_classes=8, num_images=10, model_path='./output/faster-rcnn/r50-fpn-3x/Nektar/test2/',
    #            model_name="model_final.pth", output_dir='../GoPro/output/test1/images/', show_flag=False)

    prediction_new_images(num_classes=8, num_images=10, model_path='./output/faster-rcnn/r50-fpn-3x/Nektar/test2/',
                          data_path="../../../datasets/GoPro/GX010002",
                          model_name="model_final.pth", output_dir='../GoPro/output/test1/images/', show_flag=False)
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
