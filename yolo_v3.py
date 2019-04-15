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
                            momentum=0.99,
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
    
    def __init__(self, data_format):
        super(Darknet53, self).__init__()
        self.data_format = data_format
        self.num_blocks = [1, 2, 8, 8, 4]
        
        self.conv1 = ConvBlock(32, 3, 1, data_format)
        self.conv2 = ConvBlock(64, 3, 2, data_format)
        self.darkb1 = DarknetBlock(32, data_format) 
        self.conv3 = ConvBlock(128, 3, 2, data_format)
        
        self.darkb2 = []
        for i in range(self.num_blocks[1]):
            self.darkb2.append(DarknetBlock(64, data_format))
        
        self.conv4 = ConvBlock(256, 3, 2, data_format)

        self.darkb3 = []
        for i in range(self.num_blocks[2]):
            self.darkb3.append(DarknetBlock(128, data_format))
        
        self.conv5 = ConvBlock(512, 3, 2, data_format)        
        
        self.darkb4 = []
        for i in range(self.num_blocks[3]):
            self.darkb4.append(DarknetBlock(256, data_format))        
        
        self.conv6 = ConvBlock(1024, 3, 2, data_format)
        
        self.darkb5 = []
        for i in range(self.num_blocks[4]):
            self.darkb5.append(DarknetBlock(512, data_format))                

    
    def call(self, x, training=True):
        
        output = self.conv1(x, training=training)
        output = self.conv2(output, training=training)
        output = self.darkb1(output, training=training)
        output = self.conv3(output, training=training)
        
        for i in range(self.num_blocks[1]):
            output = self.darkb2[i](output, training=training)

        output = self.conv4(output, training=training)
        
        for i in range(self.num_blocks[2]):
            output = self.darkb3[i](output, training=training)

        route_1 = output        
        output = self.conv5(output, training=training)

        for i in range(self.num_blocks[3]):
            output = self.darkb4[i](output, training=training)
        
        route_2 = output
        
        output = self.conv6(output, training=training)
        
        for i in range(self.num_blocks[4]):
            output = self.darkb5[i](output, training=training)
        
        return route_1, route_2, output

class YoloBlock(tf.keras.Model):
    
    def __init__(self, num_filters, data_format):
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

class DetectionLayer(tf.keras.Model):
    
    def __init__(self, num_classes, anchors, data_format):
        
        super(DetectionLayer, self).__init__()
        #https://www.tensorflow.org/alpha/guide/eager
        #Inherit from it to implement your own layer, and set self.dynamic=True in the constructor if the layer must be executed imperatively:
        self.num_anchors = len(anchors)

        self.data_format = data_format
        self.num_classes = num_classes
        
        self.conv1 = tf.keras.layers.Conv2D(self.num_anchors * (5 + self.num_classes), 1, strides=1, data_format=data_format, use_bias=True)
  

    def call(self, x):     
        feature_map = self.conv1(x)
        return feature_map
  

    def get_size(self, shape):
        if len(shape) == 4:
            shape = shape[1:]
        return shape[1:3] if self.data_format == 'channels_first' else shape[0:2]         

class YoloV3(tf.keras.Model):
    

    
    def __init__(self, num_classes, data_format):
        
        super(YoloV3, self).__init__()
        self.num_classes = num_classes
        self.data_format = data_format
        self._ANCHORS = [(10, 13), (16, 30), (33, 23),
                    (30, 61), (62, 45), (59, 119),
                    (116, 90), (156, 198), (373, 326)]
        '''
        self._ANCHORS = [(0.02403846153846154, 0.03125),
         (0.038461538461538464, 0.07211538461538461),
         (0.07932692307692307, 0.055288461538461536),
         (0.07211538461538461, 0.1466346153846154),
         (0.14903846153846154, 0.10817307692307693),
         (0.14182692307692307, 0.2860576923076923),
         (0.27884615384615385, 0.21634615384615385),
         (0.375, 0.47596153846153844),
         (0.8966346153846154, 0.7836538461538461)] 
        '''
        self.darket = Darknet53(data_format=self.data_format)
        

        self.yolo_block_512 = YoloBlock(512, data_format=self.data_format)
        self.detection1 = DetectionLayer(self.num_classes, self._ANCHORS[6:9], self.data_format)
        self.conv1 = ConvBlock(256, 1, data_format=self.data_format)
        self.upsampling1 = tf.keras.layers.UpSampling2D(data_format=self.data_format)
        
        self.yolo_block_256 = YoloBlock(256, data_format=self.data_format)
        self.detection2 = DetectionLayer(self.num_classes, self._ANCHORS[3:6], self.data_format)
        self.conv2 = ConvBlock(128, 1, data_format=self.data_format)
        self.upsampling2 = tf.keras.layers.UpSampling2D(data_format=self.data_format)
        
        self.yolo_block_128 = YoloBlock(128, data_format=self.data_format)    
        self.detection3 = DetectionLayer(self.num_classes, self._ANCHORS[0:3], self.data_format)
        
        

                
    def call(self, x, training=True):
        
        img_size = x.get_shape().as_list()[1:3]
        print(img_size)
        axis = 1        
        
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])  
            axis = -1
        
        route_1, route_2, inputs = self.darket(x, training=training)

        route, inputs = self.yolo_block_512(inputs, training=training)
        feature_map_1 = self.detection1(inputs)
        feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')
        
        inputs = self.conv1(route, training=training)
        upsample_size = route_2.get_shape().as_list()
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
        
        #detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
        #detections = tf.identity(detections, name='detections')
        
        #print(detections.get_shape())
        return feature_map_1, feature_map_2, feature_map_3
        #return dark_out
    
    '''    
    def detection_layer(self, predictions, anchors, img_size):
        num_anchors = len(anchors)
        shape = predictions.get_shape().as_list()
        print(shape)
        grid_size = self.get_size(shape)
        dim = grid_size[0] * grid_size[1]
        bbox_attrs = 5 + self.num_classes

        if self.data_format == 'channels_first':
            predictions = tf.reshape(
                predictions, [-1, num_anchors * bbox_attrs, dim]
            )
            predictions = tf.transpose(predictions, [0,2,1])

        predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

        stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

        anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

        box_centers, box_sizes, confidence, classes = tf.split(
            predictions, [2, 2, 1, self.num_classes], axis=-1
        )

        box_centers = tf.nn.sigmoid(box_centers)
        confidence = tf.nn.sigmoid(confidence)

        grid_x = tf.range(grid_size[0], dtype=tf.float32)
        grid_y = tf.range(grid_size[1], dtype=tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))

        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride

        anchors = tf.tile(anchors, [dim, 1])
        box_sizes = tf.exp(box_sizes) * anchors
        box_sizes = box_sizes * stride

        detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

        classes = tf.nn.sigmoid(classes)
        predictions = tf.concat([detections, classes], axis=-1)
        #print("Prediction Result : ")
        #print(predictions.get_shape())
        return predictions
    '''    
     