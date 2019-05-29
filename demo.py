import time
import numpy as np
import tensorflow as tf
import argparse
from model import create_model
from utils import load_coco_names, draw_boxes3, draw_boxes2, draw_boxes
from PIL import Image
from absl import app, flags, logging
from absl.flags import FLAGS

IMG_H, IMG_W = 416, 416

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) 
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


flags.DEFINE_string("class_names", "coco.names", "File with class names.")
flags.DEFINE_string("input_img", "./img_data/test_4.jpg", "Input image file name.")
flags.DEFINE_string("output_img", "./img_data/out/test_4.jpg", "Output image file name.")                          
flags.DEFINE_string("data_format", "channels_last", "Data format: channels_first (gpu only) / channels_last.")    
flags.DEFINE_string("weights", "./weights/yolov3.tf", "Tensorflow 2.0 Weights file.") 
flags.DEFINE_float("score_threshold", 0.5, "Desired Score Threshold") 
flags.DEFINE_float("iou_threshold", 0.5, "Desired IOU Threshold")                                                             

def main(_argv):

    img = Image.open(FLAGS.input_img)
    img_resized = np.asarray(img.resize(size=(IMG_H, IMG_W)), dtype=np.float32)
    img_resized = img_resized/255.0

    classes = load_coco_names(FLAGS.class_names)
    model = create_model(IMG_H, yolo_anchors, yolo_anchor_masks, len(classes))
    print("=> loading weights ...")
    model.load_weights(FLAGS.weights)
    print("=> sucessfully loaded weights ")

    start = time.time()
    boxes, scores, labels, nums  = model(img_resized[np.newaxis, ...], training=False)
    boxes, scores, labels, nums = boxes.numpy(), scores.numpy(), labels.numpy(), nums.numpy()
    #boxes, scores, labels = model(img_resized[np.newaxis, ...])
    print("=> nms on the number of boxes= %d  time=%.2f ms" %(nums, 1000*(time.time()-start)))

    image = draw_boxes2(img, boxes[0], scores[0], labels[0], nums[0], classes, [IMG_H, IMG_W], show=True)
    #image = draw_boxes3(img, boxes, scores, labels, classes, [IMG_H, IMG_W], show=True)
    image.save(FLAGS.output_img)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass