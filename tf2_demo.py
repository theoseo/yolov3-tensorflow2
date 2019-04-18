import time
import numpy as np
import tensorflow as tf
import argparse
import yolo_v3
from utils import load_coco_names, draw_boxes
from PIL import Image

IMG_H, IMG_W = 416, 416

parser = argparse.ArgumentParser(description="YOLO-V3 weight convert")

parser.add_argument("--class_names", type=str, default="coco.names",
                    help="File with class names.")
parser.add_argument("--input_img", type=str, default="./img_data/test.jpg",
                    help="Input image file name.")
parser.add_argument("--output_img", type=str, default="./img_data/out/test.jpg",
                    help="Output image file name.")                          
parser.add_argument("--data_format", type=str, default="channels_last",
                    help="Data format: channels_first (gpu only) / channels_last.")    
parser.add_argument("--weights", type=str, default="./tf2_weights/yolov3",
                    help="Tensorflow 2.0 Weights file.") 
parser.add_argument("--score_threshold", type=float, default=0.5,
                    help="Desired Score Theshold") 
parser.add_argument("--iou_threshold", type=float, default=0.5,
                    help="Desired IOU Theshold")                                                             

def main(argv=None):

    img = Image.open(args.input_img)
    img_resized = np.asarray(img.resize(size=(IMG_H, IMG_W)), dtype=np.float32)
    img_resized = img_resized/255.0

    classes = load_coco_names(args.class_names)
    model = yolo_v3.YoloV3(len(classes), data_format=args.data_format)
    inputs = tf.keras.Input(shape=(IMG_H, IMG_W, 3))
    output = model(inputs, training=False)
    print("=> loading weights ...")
    model.load_weights(args.weights)
    print("=> sucessfully loaded weights ")

    start = time.time()
    boxes, scores, labels = model.detector(img_resized[np.newaxis, ...], score_thresh=args.score_threshold, iou_thresh=args.iou_threshold)
    print("=> nms on the number of boxes= %d  time=%.2f ms" %(len(boxes), 1000*(time.time()-start)))

    image = draw_boxes(img, boxes, scores, labels, classes, [IMG_H, IMG_W], show=True)
    image.save(args.output_img)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)