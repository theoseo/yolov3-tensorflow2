# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import  ImageDraw, Image
import colorsys

def load_weights(model, weights_file):
    """
    Loads and converts pre-trained weights.
    :param model: Keras model
    :param weights_file: name of the binary file.
    :return total_params: if load successfully end else -1
    """
                               
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    total_params = 0

    var_list = model.variables
    
    var_name_list = ['/'.join(x.name.split('/')[:-1]) for x in var_list]

    print(weights.shape)
    print(len(model.variables))
    print(len(model.trainable_variables))

    while i < len(model.trainable_variables):
        var1 = var_list[i]
        var2 = var_list[i + 1]

        print("%d - var1 : %s (%s)" %(i, var1.name.split('/')[-2], var1.name.split('/')[-1]))
        print("%d - var2 : %s (%s)" %(i, var2.name.split('/')[-2], var2.name.split('/')[-1]))        
        
        # do something only if we process conv layer        
        if 'conv2' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_normalization' in var2.name.split('/')[-2]:
                
                # load batch norm's gamma and beta params
                # beta bias
                # gamma kernel                
                gamma, beta = var_list[i + 1:i + 3]
                
                # Find mean and variance of the same name  
                layer_name = '/'.join(gamma.name.split('/')[:-1])
                mean_index = i + 3
                mean_index += var_name_list[i+3:].index(layer_name)
                mean, var = var_list[mean_index:mean_index+2] 

                batch_norm_vars = [beta, gamma, mean, var]
                
                for batch_norm_var in batch_norm_vars:
                    shape = batch_norm_var.shape.as_list()
                    num_params = np.prod(shape)
                    batch_norm_var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                  
                    batch_norm_var.assign(batch_norm_var_weights, name=batch_norm_var.name)

                # we move the pointer by 4, because we loaded 4 variables
                i += 2
            elif 'conv2' in var2.name.split('/')[-2]:
                # load biases
                print("%d - var2 : %s" %(i, var2.name.split('/')[-2]))
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params

                bias.assign(bias_weights, name=bias.name)

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
        
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            var1.assign(var_weights, name=var1.name)
            i += 1
            
        total_params = ptr
            
    return total_params if total_params == weights.shape else -1



def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names



def letter_box_image(image: Image.Image, output_height: int, output_width: int, fill_value)-> np.ndarray:
    """
    Fit image with final image with output_width and output_height.
    :param image: PILLOW Image object.
    :param output_height: width of the final image.
    :param output_width: height of the final image.
    :param fill_value: fill value for empty area. Can be uint8 or np.ndarray
    :return: numpy image fit within letterbox. dtype=uint8, shape=(output_height, output_width)
    """

    height_ratio = float(output_height)/image.size[1]
    width_ratio = float(output_width)/image.size[0]
    fit_ratio = min(width_ratio, height_ratio)
    fit_height = int(image.size[1] * fit_ratio)
    fit_width = int(image.size[0] * fit_ratio)
    fit_image = np.asarray(image.resize((fit_width, fit_height), resample=Image.BILINEAR))

    if isinstance(fill_value, int):
        fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

    to_return = np.tile(fill_value, (output_height, output_width, 1))
    pad_top = int(0.5 * (output_height - fit_height))
    pad_left = int(0.5 * (output_width - fit_width))
    to_return[pad_top:pad_top+fit_height, pad_left:pad_left+fit_width] = fit_image
    return to_return


def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size)-> np.ndarray:
    """
    Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
    :param letter_pos: The current position within letterbox image including fill value area.
    :param current_size: The size of whole image including fill value area.
    :param ori_image_size: The size of image before being letter boxed.
    :return:
    """
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio
    return to_return_pos


def draw_boxes(image, boxes, scores, labels, classes, detection_size,
               font='./data/font/FiraMono-Medium.otf', show=True):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """
    if boxes is None: return image
    draw = ImageDraw.Draw(image)
    # draw settings
    hsv_tuples = [( x / len(classes), 0.9, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    for i in range(len(labels)): # for each bounding box, do:
        bbox, score, label = boxes.numpy()[i], scores.numpy()[i], classes[labels.numpy()[i]]
        bbox_text = "%s %.2f" %(label, score)
        print(bbox_text)
        text_size = draw.textsize(bbox_text)
        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2,2) * ratio).reshape(-1))

        #draw.rectangle(bbox, outline=colors[labels[i]], width=3)
        draw.rectangle(bbox, outline=colors[labels[i]])
        text_origin = bbox[:2]-np.array([0, text_size[1]])
        draw.rectangle([tuple(text_origin), tuple(text_origin+text_size)], fill=colors[labels[i]])
        # # draw bbox
        draw.text(tuple(text_origin), bbox_text, fill=(0,0,0))

    image.show() if show else None
    return image


def draw_boxes2(image, boxes, scores, labels, nums, classes, detection_size,
               font='./data/font/FiraMono-Medium.otf', show=True):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """
    if boxes is None: return image
    draw = ImageDraw.Draw(image)
    # draw settings
    hsv_tuples = [( x / len(classes), 0.9, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    for i in range(nums): # for each bounding box, do:
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        bbox_text = "%s %.2f" %(label, score)
        print(bbox_text)
        text_size = draw.textsize(bbox_text)
        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2,2) * original_size).reshape(-1))

        #draw.rectangle(bbox, outline=colors[labels[i]], width=3)
        draw.rectangle(bbox, outline=colors[int(labels[i])])
        text_origin = bbox[:2]-np.array([0, text_size[1]])
        draw.rectangle([tuple(text_origin), tuple(text_origin+text_size)], fill=colors[int(labels[i])])
        # # draw bbox
        draw.text(tuple(text_origin), bbox_text, fill=(0,0,0))

    image.show() if show else None
    return image


def draw_boxes3(image, boxes, scores, labels, classes, detection_size,
               font='./data/font/FiraMono-Medium.otf', show=True):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """
    if boxes is None: return image
    draw = ImageDraw.Draw(image)
    # draw settings
    hsv_tuples = [( x / len(classes), 0.9, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    for i in range(len(labels)): # for each bounding box, do:
        bbox, score, label = boxes.numpy()[i], scores.numpy()[i], classes[labels.numpy()[i]]
        bbox_text = "%s %.2f" %(label, score)
        print(bbox_text)
        text_size = draw.textsize(bbox_text)
        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2,2) * original_size).reshape(-1))

        #draw.rectangle(bbox, outline=colors[labels[i]], width=3)
        draw.rectangle(bbox, outline=colors[labels[i]])
        text_origin = bbox[:2]-np.array([0, text_size[1]])
        draw.rectangle([tuple(text_origin), tuple(text_origin+text_size)], fill=colors[labels[i]])
        # # draw bbox
        draw.text(tuple(text_origin), bbox_text, fill=(0,0,0))

    image.show() if show else None
    return image

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    'image/object/view': tf.io.VarLenFeature(tf.string),
}

class Preprocess(object):
    
    def __init__(self, anchors, anchor_masks, train):
        #self.classes = classes
        self.train = train
        self.anchors = anchors / 416.
        self.anchor_masks = anchor_masks
        self.class_table = tf.lookup.StaticHashTable(
    tf.lookup.TextFileInitializer('./coco.names', tf.string, 0, tf.int64, -1, delimiter="\n"), -1)
        
       
    def __call__(self, tfrecord):
        

        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        x_train = tf.image.resize(x_train, (416, 416))
        x_train = x_train / 255.       
        
        class_text = tf.sparse.to_dense(x['image/object/class/text'], default_value='')
        labels = tf.cast(self.class_table.lookup(class_text), tf.float32)
        y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                            tf.sparse.to_dense(x['image/object/bbox/ymin']),
                            tf.sparse.to_dense(x['image/object/bbox/xmax']),
                            tf.sparse.to_dense(x['image/object/bbox/ymax']),
                            labels], axis=1)
        
        # calculate anchor index for true boxes
        anchors = tf.cast(self.anchors, tf.float32)
        anchor_area = anchors[..., 0] * anchors[..., 1]
        box_wh = y_train[..., 2:4] - y_train[..., 0:2]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                         (1, tf.shape(anchors)[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
            tf.minimum(box_wh[..., 1], anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

        y_train = tf.concat([y_train, anchor_idx], axis=-1)   
        
        y_outs = []
        grid_size = 13        
        for anchor_idxs in anchor_masks:
            y_outs.append(self.transform_targets_for_output(
                y_train, grid_size, anchor_idxs, 80))
            grid_size *= 2   
            
        return x_train, tuple(y_outs)
    
    @tf.function 
    def transform_targets_for_output(self, y_true, grid_size, anchor_idxs, classes):
        # y_true: (boxes, (x1, y1, x2, y2, class, best_anchor))

        y_true_out = tf.zeros(
            (grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

        anchor_idxs = tf.cast(anchor_idxs, tf.int32)

        indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 0, dynamic_size=True)
        idx = 0

        y_true_shape = y_true.get_shape().as_list()
        for j in tf.range(tf.shape(y_true)[0]):
            #if tf.equal(y_true[j][2], 0):
            #    continue

            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[j][5], tf.int32))
         
            if tf.reduce_any(anchor_eq):

                box = y_true[j][0:4]
                box_xy = (y_true[j][0:2] + y_true[j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[j][4]])

                idx += 1

        return tf.tensor_scatter_nd_update(
            y_true_out, indexes.stack(), updates.stack())

def create_dataset(buffer_size, batch_size, data_format, anchors, anchor_mask, data_dir=None):

    train_file_pattern = '../models/research/object_detection/urban/coco/coco_train.record-?????-of-?????'
    val_file_pattern = '../models/research/object_detection/urban/coco/coco_val.record-?????-of-?????'

    train_files = tf.data.Dataset.list_files(train_file_pattern)
    val_files = tf.data.Dataset.list_files(val_file_pattern)

    train_dataset = tf.data.TFRecordDataset(train_files)
    val_dataset = tf.data.TFRecordDataset(val_files)

    # https://www.tensorflow.org/alpha/guide/data_performance 
    #raw_dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2, 
    # num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #raw_dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2, 
    # num_parallel_calls=tf.data.experimental.AUTOTUNE)

    preprocess_train = Preprocess(anchors, anchor_masks, train=True)
    preprocess_val = Preprocess(anchors, anchor_masks, train=False)

    train_dataset = train_dataset.map(preprocess_train)
    val_dataset = val_dataset.map(preprocess_train)

    return train_dataset, val_dataset