import tensorflow as tf
import numpy as np
from model import create_model
import logging
import argparse

from utils import load_coco_names, load_weights


IMG_H, IMG_W = 416, 416

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) 
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

parser = argparse.ArgumentParser(description="YOLO-V3 weight convert")

parser.add_argument("--class_names", type=str, default="coco.names",
                    help="File with class names.")    
parser.add_argument("--weights_file", type=str, default="yolov3.weights",
                    help="Binary file with detector weights.")    
parser.add_argument("--data_format", type=str, default="channels_last",
                    help="Data format: channels_first / channels_last.")    
#parser.add_argument("--tiny", type=bool, default=False,
#                    help="Use tiny version of YOLOv3.")    
parser.add_argument("--tf2_weights", type=str, default="./weights/yolov3.tf",
                    help="Tensorflow 2.0 Weights file.")    


def main(argv=None):

    classes = load_coco_names(args.class_names)
    
    model = create_model(IMG_H, yolo_anchors, yolo_anchor_masks, len(classes))
    load_ops = load_weights(model, args.weights_file)  

    """
    Saving Subclassed Models

    https://www.tensorflow.org/alpha/guide/keras/saving_and_serializing
    Sequential models and Functional models are datastructures that represent a DAG of layers. 
    As such, they can be safely serialized and deserialized.

    https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021
    """  
    model.save_weights(args.tf2_weights)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
