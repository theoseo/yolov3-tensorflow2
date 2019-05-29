# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import time
from absl import app, flags, logging

def create_model(size, yolo_anchors, yolo_anchor_masks, classes, training=False):

    inputs = tf.keras.Input(shape=(size, size, 3))
    outputs = YoloV3(size, yolo_anchors, yolo_anchor_masks, classes)(inputs, training=training)
    return tf.keras.Model(inputs, outputs, name='yolov3')


class DarknetConv(layers.Layer):
    
    def __init__(self, filters, size, strides=1, is_batch_norm=True):
        
        super(DarknetConv, self).__init__()
        self.filters = filters
        self.size = size
        self.is_batch_norm = is_batch_norm
        self.strides = strides
        
        self.zeropadding = layers.ZeroPadding2D(((1,0),(1,0)))
        self.conv2d = layers.Conv2D(filters, size, 
                        strides, padding = ('same' if strides == 1 else 'valid'), 
                        use_bias = not is_batch_norm, kernel_regularizer=l2(0.0005))
        self.batchnorm = layers.BatchNormalization(momentum = 0.9, epsilon = 1e-05)
        self.leakyrelu = layers.LeakyReLU(alpha=0.1)
        
    def call(self, x, training=True):
        
        if self.strides > 1:
            x = self.zeropadding(x)
            
        x = self.conv2d(x)
        
        if self.is_batch_norm :
            x = self.batchnorm(x)
            x = self.leakyrelu(x)
            
        return x

class DarknetResidual(layers.Layer):
    
    def __init__(self, filters):
        
        super(DarknetResidual, self).__init__()      
        self.filters = filters
        
        self.darknetconv1 = DarknetConv(filters, 1)
        self.darknetconv2 = DarknetConv(filters * 2, 3)        
        self.add = layers.Add()
        
    def call(self, x, training=True):
        
        shortcut = x
        x = self.darknetconv1(x, training=training)
        x = self.darknetconv2(x, training=training)        
        x = self.add([shortcut, x])
        
        return x

class DarknetBlock(layers.Layer):
    
    def __init__(self, filters, blocks):
        
        super(DarknetBlock, self).__init__()      
        self.filters = filters
        self.blocks = blocks
        
        self.darknetconv = DarknetConv(filters, 3, strides=2)
        self.darknetblocks = [DarknetResidual(filters//2) for _ in range(blocks)]
        
    def call(self, x, training=True):
        
        x = self.darknetconv(x, training=training)
        for i in range(self.blocks):
            x = self.darknetblocks[i](x, training=training)
            
        return x

class Darknet(tf.keras.Model):
    
    def __init__(self, name, **kwargs):
        super(Darknet, self).__init__(name=name, **kwargs)
        #self.name = name

        self.conv1 = DarknetConv(32, 3)
        self.block1 = DarknetBlock(64, 1)
        self.block2 = DarknetBlock(128, 2)
        self.block3 = DarknetBlock(256, 8)
        self.block4 = DarknetBlock(512, 8)
        self.block5 = DarknetBlock(1024, 4)
        
    def call(self, x, training=True):
        
        x = self.conv1(x, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = route_1 = self.block3(x, training=training)
        x = route_2 = self.block4(x, training=training)
        x = self.block5(x, training=training)        
        
        return route_1, route_2, x

class YoloConv(layers.Layer):
    
    def __init__(self, filters, is_first=True):
        super(YoloConv, self).__init__()    
        self.is_first = is_first
        
        if not self.is_first :
            
            self.conv1 = DarknetConv(filters, 1)
            self.upsampling = layers.UpSampling2D()
            self.concat = layers.Concatenate()
            

        self.conv2 = DarknetConv(filters, 1)
        self.conv3 = DarknetConv(filters * 2, 3)
        self.conv4 = DarknetConv(filters, 1)
        self.conv5 = DarknetConv(filters * 2, 3)
        self.conv6 = DarknetConv(filters, 1)        
            
    def call(self, x, training=True):
        
        if not self.is_first :
            x, x_skip = x
            
            x = self.conv1(x, training=training)
            x = self.upsampling(x)
            x = self.concat([x, x_skip])
        
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)        

        return x

class YoloOutput(layers.Layer):
    
    def __init__(self, filters, anchors, classes ):
        super(YoloOutput, self).__init__()        
        
        self.filters = filters
        self.anchors = anchors
        self.classes = classes
        
        self.darkconv = DarknetConv(filters*2, 3)
        self.biasconv = DarknetConv(anchors * (classes + 5), 1, is_batch_norm=False)
        
    def call(self, x, training=True):
        
        x = self.darkconv(x, training=training)
        x = self.biasconv(x, training=training)
        x = tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], self.anchors, self.classes+5))
        
        return x


class YoloV3(tf.keras.Model):
    
    def __init__(self, size, anchors, anchor_masks, classes ):
        super(YoloV3, self).__init__()       
        self.size = size
        self.anchors = anchors / size
        self.anchor_masks = anchor_masks
        self.classes = classes
        
        self.darknet53 = Darknet(name='yolo_darknet')
        
        
        self.yoloconv1 = YoloConv(512)
        self.output1 = YoloOutput(512, len(self.anchor_masks[0]), classes)
        
        self.yoloconv2 = YoloConv(256, is_first=False)
        self.output2 = YoloOutput(256, len(self.anchor_masks[1]), classes)
        
        self.yoloconv3 = YoloConv(128, is_first=False)
        self.output3 = YoloOutput(128, len(self.anchor_masks[2]), classes)      
        '''

        self.yolo_blocks = []
        self.yolo_outputs = []
        self.yolo_num_layers = [512, 256, 128]

        for i in range(len(self.yolo_num_layers)):
            self.yolo_blocks.append(YoloConv(self.yolo_num_layers[i]))
            self.yolo_outputs.append(YoloOutput(self.yolo_num_layers[i], len(self.anchor_masks[i]), classes)

        '''
        
    def call(self, x, training=True):
        
        route_1, route_2, x = self.darknet53(x, training=training)
        
        
        x = self.yoloconv1(x, training=training)
        output_0 = self.output1(x, training=training)
        
        x = self.yoloconv2((x, route_2), training=training)
        output_1 = self.output2(x, training=training)        
        
        x = self.yoloconv3((x, route_1), training=training)
        output_2 = self.output3(x, training=training)                
        '''
        outputs = []
        for i in range(len(self.yolo_num_layers)):
            x = yolo_blocks[i](x, training=training)
            outputs[i] = 

        ''' 

        boxes_0 = self.yolo_boxes(output_0, self.anchors[self.anchor_masks[0]], self.classes)
        boxes_1 = self.yolo_boxes(output_1, self.anchors[self.anchor_masks[1]], self.classes)        
        boxes_2 = self.yolo_boxes(output_2, self.anchors[self.anchor_masks[2]], self.classes)

        if training :
            print('traing true')
            return (boxes_0, boxes_1, boxes_2)        
        else:
            print('traing false')

            pred_0 = tf.reshape(boxes_0, (tf.shape(boxes_0)[0], len(self.anchor_masks[0]) * tf.shape(boxes_0)[1] * tf.shape(boxes_0)[2], 5 + self.classes))
            pred_1 = tf.reshape(boxes_1, (tf.shape(boxes_1)[0], len(self.anchor_masks[1]) * tf.shape(boxes_1)[1] * tf.shape(boxes_1)[2], 5 + self.classes))
            pred_2 = tf.reshape(boxes_2, (tf.shape(boxes_2)[0], len(self.anchor_masks[2]) * tf.shape(boxes_2)[1] * tf.shape(boxes_2)[2], 5 + self.classes))
            
            boxes = tf.concat([pred_0, pred_1, pred_2], axis=1)
            
            return self.yolo_nms(boxes, self.anchors, self.anchor_masks, self.classes)
        
    
    
    def yolo_boxes(self, pred, anchors, classes):
        
        grid_size = tf.shape(pred)[1]
    
        #pred = tf.reshape(pred, (tf.shape(pred)[0], len(anchors) * grid_size * grid_size, 5 + classes))

        box_centers, box_wh, confidence, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)
        

        box_centers = tf.sigmoid(box_centers)
        confidence = tf.sigmoid(confidence)
        class_probs = tf.sigmoid(class_probs)

        pred_box = tf.concat((box_centers, box_wh), axis=-1)

        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        
        box_centers = (box_centers + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors
        
        box_x1y1 = box_centers - box_wh /2
        box_x2y2 = box_centers + box_wh /2
        
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
        
        pred = tf.concat([bbox, confidence, class_probs], axis=-1)
        print(pred.shape)
        #logging.info(pred.shape)
        #pred = tf.reshape(pred, (tf.shape(pred)[0], len(anchors) * grid_size * grid_size, 5 + classes))
        
        return pred
    
    def yolo_nms(self, boxes, anchors, masks, classes):
        
        bbox, confs, class_probs = tf.split(boxes, [4,1,-1], axis=-1)
        
        scores = confs * class_probs
        '''
        logging.info(bbox.shape)
        bbox = tf.reshape(bbox, [-1, 4])
        scores = tf.reshape(scores, [-1, classes])

        mask = tf.greater_equal(scores, tf.constant(0.5))

        boxes_list, label_list, score_list = [], [], []

        for i in range(classes):
            
            filter_boxes = tf.boolean_mask(bbox, mask[:,i])
            filter_scores = tf.boolean_mask(scores[:,i], mask[:,i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                    scores=filter_scores,
                                                    max_output_size=tf.constant(50),
                                                    iou_threshold=tf.constant(0.5), name='nms_indices')
            
            label_list.append(tf.ones_like(tf.gather(filter_scores, nms_indices), 'int32')*i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_scores, nms_indices))
        #print("=> nms time=%.2f ms" %(1000*(time.time()-start)))
        
        boxes = tf.concat(boxes_list, axis=0)
        scores = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        
        above 864ms
        2000ms
        '''
        start = time.time()
        boxes, scores, classes, valid_detections = tf.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
            max_output_size_per_class=10,
            max_total_size=50,
            iou_threshold=0.5,
            score_threshold=0.5
        )
        logging.info("=> combined_non_max_suppression time=%.2f ms" %(1000*(time.time()-start)))
        
        return boxes, scores, classes, valid_detections
        

        #return boxes, scores, label