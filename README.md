# YOLO v3 in Tensorflow 2.0

Implementation of YOLO v3 object detector in Tensorflow 2.0 (Keras). Tested on Python 3.6, Tensorflow 2.0 alpha on Ubuntu 16.04.

## Reference
1. [Darknet YOLO](https://pjreddie.com/darknet/yolo/)
2. [How to implement a YOLO(v3) Object detector from scratch in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
3. [Paweł Kapica
](https://github.com/mystic123/tensorflow-yolo-v3) and his post - [https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe)
4. [YangYun](https://github.com/YunYang1994/tensorflow-yolov3) 

## Todo list:
- [x] Darknet53 architecture and Test
- [x] YOLO v3 architecture
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
3. Run `python ./tf2_demo.py --input_img <path-to-image> --output_img <name-of-output-image> `

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
2. tf2_demo.py
    1. `--class_names`
        1. Path to the class names file
    2. `--weights`
        1. Path to the desired weights
    3. `--data_format`
        1.  `channels_first` or `channels_last`
    4. `--score_threshold`
        1. Desired score threshold
    5. `--iou_threshold`
        1. Desired iou threshold
