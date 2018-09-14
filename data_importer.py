import os
import cv2 as cv
import numpy as np
import re


def import_image_and_xml(img_path, xml_path):
    # input : image path, xml
    # + labels?
    # process : importing, parsing
    # output : images, xmls
    images = []
    for _, _, files in os.walk(img_path):
        for file in files:
            image = cv.imread(img_path + file)
            images.append(image)
    images = np.array(images)

    gt_classes_batch = []
    gt_boxes_batch = []
    gt_image_shape_batch = []
    for _, _, files in os.walk(xml_path):
        for file in files:
            xml_org = ''
            f = open(xml_path + file, 'r')
            while True:
                line = f.readline()
                if not line: break
                xml_org += line
            f.close()

            names_start = [m.start() for m in re.finditer('<name>', xml_org)]
            names_end = [m.start() for m in re.finditer('</name>', xml_org)]
            boxes_start = [m.start() for m in re.finditer('<bndbox>', xml_org)]
            boxes_end = [m.start() for m in re.finditer('</bndbox>', xml_org)]
            size_start = [m.start() for m in re.finditer('<size>', xml_org)]
            size_end = [m.start() for m in re.finditer('</size>', xml_org)]

            true_idx = []
            for i, row in enumerate(names_start):
                if xml_org[row - 10:row - 4] == 'object': true_idx.append(i)

            names_start = [names_start[idx] for idx in true_idx]
            names_end = [names_end[idx] for idx in true_idx]
            boxes_start = [boxes_start[idx] for idx in true_idx]
            boxes_end = [boxes_end[idx] for idx in true_idx]

            gt_classes = []
            for i in range(0, len(names_end)):
                gt_classes.append((xml_org[names_start[i] + 6:names_end[i]]))

            gt_boxes = []
            for i in range(0, len(boxes_start)):
                xml_box = xml_org[boxes_start[i]:boxes_end[i]]
                xml_boxes = xml_box.split('\n')[1:-1]
                gt_box = []
                for row in xml_boxes:
                    gt_box.append(int(re.search(r'\d+', row).group()))
                gt_boxes.append(gt_box)

            gt_image_shape = {}
            gt_image_shape_rows = xml_org[size_start[0]:size_end[0]].split('\n')[1:-1]
            for row in gt_image_shape_rows:
                if 'height' in row:
                    gt_image_shape['height'] = int(re.search(r'\d+', row).group())
                elif 'width' in row:
                    gt_image_shape['width'] = int(re.search(r'\d+', row).group())

            gt_classes_batch.append(gt_classes)
            gt_boxes_batch.append(gt_boxes)
            gt_image_shape_batch.append(gt_image_shape)
    gt_classes_batch = np.array(gt_classes_batch)
    gt_boxes_batch = np.array(gt_boxes_batch)

    xmls = {'classes': gt_classes_batch, 'boxes':gt_boxes_batch}

    return images, xmls

def get_class_idx(name):
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    return class_names.index(name)+1


def get_class_name(idx):
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    return class_names[idx]
