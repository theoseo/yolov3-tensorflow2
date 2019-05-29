# YOLOv3 implementation in TensorFlow 2.0

This implementation of YOLOv3 object detector in TensorFlow 2.0 (Keras). Referenced great resources below. Traning is on going. Tested on Python 3.6, TensorFlow 2.0 alpha on Ubuntu 16.04.

## Reference
- [Darknet YOLO](https://pjreddie.com/darknet/yolo/)
    - models
    - weight converter
- [YangYun](https://github.com/YunYang1994/tensorflow-yolov3) 
    - weight onverter
- [How to implement a YOLO(v3) Object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
    - weight converter
- [Paweł Kapica](https://github.com/mystic123/tensorflow-yolo-v3)
    - models    
- [YoloV3 Implemented in TensorFlow 2.0](https://github.com/zzh8829/yolov3-tf2)
    - models for TensorFlow 2.0 
- [Densely Connected Convolutional Networks](https://github.com/tensorflow/examples/tree/master/tensorflow_examples/models/densenet)
    - model class for TensorFlow 2.0

## Todo list:
- [x] Darknet53 architecture and Test
- [x] YOLOv3 architecture
- [x] Basic working demo
- [x] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [ ] Training pipeline
- [ ] More backends

## How to run the demo:
To run demo type this in the command line:

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download and convert model weights:    
    1. Download binary file with desired weights: 
        1. Full weights: `wget https://pjreddie.com/media/files/yolov3.weights`
    2. Run `python ./convert_weights.py` 
3. Run `python ./demo.py --input_img <path-to-image> --output_img <name-of-output-image> `

####Optional Flags
1. convert_weights.py:
    1. `--class_names`
        1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file
    3. `--data_format`
        1.  `channels_first` or `channels_last`
    4. `--tf2_weights`
        1. Output weights file
2. demo.py
    1. `--class_names`
        1. Path to the class names file
    2. `--input_img`
        1. Path to the input image file
    3. `--output_img`
        1. Path to the output image file
    4. `--data_format`
        1.  `channels_first` or `channels_last`
    5. `--weights`
        1. TensorFlow 2.0 Weights file.
    6. `--score_threshold`
        1. Desired Score Threshold
    7. `--iou_threshold`
        1. Desired IOU Threshold
