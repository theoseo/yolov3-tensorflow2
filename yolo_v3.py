# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ConvBlock(layers.Layer):
    
    def __init__(self, num_filters, kernel_size, strides=1, data_format='channels_last'):
        
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        
        self.conv = layers.Conv2D(num_filters,
                                            kernel_size,
                                            strides,
                                            padding = ('same' if strides == 1 else 'valid'), 
                                            data_format=data_format, 
                                            use_bias=False)
        self.batchnorm = layers.BatchNormalization(
                            axis = -1 if data_format == "channels_last" else 1,
                            #momentum=0.99,
                            momentum=0.9,
                            #epsilon=0.01)
                            epsilon=1e-05)
        self.leaky_relu = layers.LeakyReLU(alpha=0.1)

    def call(self, x, training=False):
        
        if self.strides > 1:
            x = self._fixed_padding(x, self.kernel_size)
            
        output = self.conv(x)
        output = self.batchnorm(output, training=training)
        output = self.leaky_relu(output)
        
        return output
    
    def _fixed_padding(self, inputs, kernel_size, mode='CONSTANT'):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if self.data_format == 'channels_first':
            padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
        else:
            padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode=mode)

        return padded_inputs    

class DarknetBlock(layers.Layer):
    
    def __init__(self, num_filters, data_format='channels_last'):
        
        super(DarknetBlock, self).__init__()

        self.conv1 = ConvBlock(num_filters, 1, data_format=data_format)
        self.conv2 = ConvBlock(num_filters*2, 3, data_format=data_format)
        self.add = layers.Add()
    
    def call(self, x, training=False):

        shortcut = x
        output = self.conv1(x,training=training)
        output = self.conv2(output,training=training)        
        output = self.add([output, shortcut])
            
        return output

class Darknet53(tf.keras.Model):
    
    def __init__(self, data_format='channels_last'):
        super(Darknet53, self).__init__()
        self.data_format = data_format
        self.num_blocks = [1, 2, 8, 8, 4]
        
        self.conv1 = ConvBlock(32, 3, 1, data_format)
        self.conv2 = ConvBlock(64, 3, 2, data_format)
        self.darkblock1 = DarknetBlock(32, data_format) 
        self.conv3 = ConvBlock(128, 3, 2, data_format)
        
        self.darkblock2 = [DarknetBlock(64, data_format) for _ in range(self.num_blocks[1])]
        
        self.conv4 = ConvBlock(256, 3, 2, data_format)

        self.darkblock3 = [DarknetBlock(128, data_format) for _ in range(self.num_blocks[2])]
        
        self.conv5 = ConvBlock(512, 3, 2, data_format)        
        
        self.darkblock4 = [DarknetBlock(256, data_format) for _ in range(self.num_blocks[3])]

        
        self.conv6 = ConvBlock(1024, 3, 2, data_format)
        
        self.darkblock5 = [DarknetBlock(512, data_format) for _ in range(self.num_blocks[4])]
    
    def call(self, x, training=True):
        
        output = self.conv1(x, training=training)
        output = self.conv2(output, training=training)
        output = self.darkblock1(output, training=training)
        output = self.conv3(output, training=training)
        
        for i in range(self.num_blocks[1]):
            output = self.darkblock2[i](output, training=training)

        output = self.conv4(output, training=training)
        
        for i in range(self.num_blocks[2]):
            output = self.darkblock3[i](output, training=training)

        route_1 = output        
        output = self.conv5(output, training=training)

        for i in range(self.num_blocks[3]):
            output = self.darkblock4[i](output, training=training)
        
        route_2 = output
        
        output = self.conv6(output, training=training)
        
        for i in range(self.num_blocks[4]):
            output = self.darkblock5[i](output, training=training)
        
        return route_1, route_2, output

class YoloBlock(tf.keras.Model):
    
    def __init__(self, num_filters, data_format='channels_last'):
        super(YoloBlock, self).__init__()
        self.num_filters = num_filters
        self.data_format = data_format

        self.conv1 = ConvBlock(num_filters, 1, data_format=data_format)
        self.conv2 = ConvBlock(num_filters*2, 3, data_format=data_format)        
        self.conv3 = ConvBlock(num_filters, 1, data_format=data_format)
        self.conv4 = ConvBlock(num_filters*2, 3, data_format=data_format)                
        self.conv5 = ConvBlock(num_filters, 1, data_format=data_format)

        self.conv6 = ConvBlock(num_filters*2, 3, data_format=data_format)                        
        
    def call(self, x, training=True):
        
        output = self.conv1(x, training=training)
        output = self.conv2(output, training=training)
        output = self.conv3(output, training=training)
        output = self.conv4(output, training=training)
        output = self.conv5(output, training=training)  
        
        route = output
        
        output = self.conv6(output, training=training)  
        
        return route, output

class DetectionLayer(layers.Layer):
    
    def __init__(self, img_size, num_classes, anchors, data_format='channels_last'):
        
        super(DetectionLayer, self).__init__()

        self.num_anchors = len(anchors)
        self.img_size = img_size
        self.anchors = anchors
        self.data_format = data_format
        self.num_classes = num_classes
        
        self.conv1 = layers.Conv2D(self.num_anchors * (5 + self.num_classes), 1, strides=1, data_format=data_format, use_bias=True)
        

    def call(self, x):     
        
        predictions = self.conv1(x)
        
        shape = predictions.get_shape().as_list()  
        grid_size = self._get_size(shape)
        dim = grid_size[0] * grid_size[1]
        bbox_attrs = 5 + self.num_classes
        
        predictions = tf.reshape(predictions, [-1, self.num_anchors * dim, bbox_attrs])
        
        stride = (self.img_size[0]//grid_size[0], self.img_size[1]//grid_size[1])
        anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in self.anchors]

        box_centers, box_sizes, confidence, classes = tf.split(
            predictions, [2, 2, 1, self.num_classes], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)
        #confidence = tf.nn.sigmoid(confidence)

        grid_x = tf.range(grid_size[0], dtype=tf.float32)
        grid_y = tf.range(grid_size[1], dtype=tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))

        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, self.num_anchors]), [1, -1, 2])

        box_centers = box_centers + x_y_offset

        anchors = tf.tile(anchors, [dim, 1])
        box_sizes = tf.exp(box_sizes) * anchors

        """
        https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/

        The last thing we want to do here, is to resize the detections map to the size of the input image. 
        The bounding box attributes here are sized according to the feature map (say, 13 x 13). 
        If the input image was 416 x 416, we multiply the attributes by 32, or the stride variable.
        """
        box_centers = box_centers * stride
        box_sizes = box_sizes * stride

        detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

        #classes = tf.nn.sigmoid(classes)
        predictions = tf.concat([detections, classes], axis=-1)        
        
        return predictions
  

    def _get_size(self, shape):
        if len(shape) == 4:
            shape = shape[1:]
        return shape[1:3] if self.data_format == 'channels_first' else shape[0:2] 


class YoloV3(tf.keras.Model):
      
    def __init__(self, num_classes, data_format):
        
        super(YoloV3, self).__init__()
        self.num_classes = num_classes
        self.data_format = data_format
        self.img_size = (416,416)
        
        self._ANCHORS = [(10, 13), (16, 30), (33, 23),
                    (30, 61), (62, 45), (59, 119),
                    (116, 90), (156, 198), (373, 326)]

        self.darket = Darknet53(data_format=self.data_format)
        
        self.yolo_block_512 = YoloBlock(512, data_format=self.data_format)
        self.detection1 = DetectionLayer(self.img_size, self.num_classes, self._ANCHORS[6:9], self.data_format)
        self.conv1 = ConvBlock(256, 1, data_format=self.data_format)
        self.upsampling1 = layers.UpSampling2D(data_format=self.data_format)
        
        self.yolo_block_256 = YoloBlock(256, data_format=self.data_format)
        self.detection2 = DetectionLayer(self.img_size, self.num_classes, self._ANCHORS[3:6], self.data_format)
        self.conv2 = ConvBlock(128, 1, data_format=self.data_format)
        self.upsampling2 = layers.UpSampling2D(data_format=self.data_format)
        
        self.yolo_block_128 = YoloBlock(128, data_format=self.data_format)    
        self.detection3 = DetectionLayer(self.img_size, self.num_classes, self._ANCHORS[0:3], self.data_format)
        
    def call(self, x, training=True):
        
        self.img_size = x.get_shape().as_list()[1:3]
        
        axis = 1        
        
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])  
            axis = -1
        
        route_1, route_2, inputs = self.darket(x, training=training)


        route, inputs = self.yolo_block_512(inputs, training=training)
        feature_map_1 = self.detection1(inputs)
        feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')
        
        """
        Authors take the feature map from 2 layers previous and upsample it by 2×
        They take a feature map from earlier in the network and merge it with 
        our upsampled features using concatenation. 
        
        This method allows us to get more meaningful semantic information 
        from the upsampled features and finer-grained information 
        from the earlier feature map.

        We then add a few more convolutional layers to process this combined feature map, 
        and eventually predict a similar tensor, although now twice the size.  

        We perform the same design one more time to predict boxes for the final scale. 
        Thus our predictions for the 3rd scale benefit from all the prior computation 
        as well as fine-grained features from early on in the network.               
        """

        inputs = self.conv1(route, training=training)
        inputs = self.upsampling1(inputs)

        inputs = tf.concat([inputs, route_2], axis=1 if self.data_format == 'channels_first' else 3)
        
        route, inputs = self.yolo_block_256(inputs, training=training)
        feature_map_2 = self.detection2(inputs)
        feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')
        
        inputs = self.conv2(route, training=training)
        upsample_size = route_1.get_shape().as_list()
        inputs = self.upsampling2(inputs)
        inputs = tf.concat([inputs, route_1], axis=1 if self.data_format == 'channels_first' else 3)
        
        _, inputs = self.yolo_block_128(inputs, training=training)
        feature_map_3 = self.detection3(inputs)
        feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')
        
        detections = tf.concat([feature_map_1, feature_map_2, feature_map_3], axis=1)
        detections = tf.identity(detections, name='detections')

        return detections
    
    def detector(self, x, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
        """
        Note: given by predicted, compute the receptive field
              and get boxes, confs and class_probs
        input_argument: x -> [None, 1, 10647, 85],
                        max_boxes
                        score_thresh
                        iou_thresh
        return : boxes, scores, label
        """
        detections = self.predict(x)

        center_x, center_y, width, height, conf_logits, prob_logits = tf.split(detections, [1, 1, 1, 1, 1, -1], axis=-1)
        
        x0 = center_x - width   / 2.
        y0 = center_y - height  / 2.
        x1 = center_x + width   / 2.
        y1 = center_y + height  / 2.

        boxes = tf.concat([x0, y0, x1, y1], axis=-1) 
        
        confs = tf.sigmoid(conf_logits)
        probs = tf.sigmoid(prob_logits)
        
        scores = confs * probs

        boxes = tf.reshape(boxes, [-1, 4])
        scores = tf.reshape(scores, [-1, self.num_classes])

        mask = tf.greater_equal(scores, tf.constant(score_thresh))

        boxes_list, label_list, score_list = [], [], []

        for i in range(self.num_classes):
            filter_boxes = tf.boolean_mask(boxes, mask[:,i])
            filter_scores = tf.boolean_mask(scores[:,i], mask[:,i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                    scores=filter_scores,
                                                    max_output_size=max_boxes,
                                                    iou_threshold=iou_thresh, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_scores, nms_indices), 'int32')*i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_scores, nms_indices))

        boxes = tf.concat(boxes_list, axis=0)
        scores = tf.concat(score_list, axis=0)
        label = tf.concat(label_list, axis=0)

        return boxes, scores, label
