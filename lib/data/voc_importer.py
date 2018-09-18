import os
import cv2
import numpy as np
import re

from ..util import find_path


def voc_xml_parser(img_path, xml_path):
    directory_list = [img_path]
    for _, dirs, _ in os.walk(img_path):
        directory_list += dirs

    images = []
    gt_classes_batch = []
    gt_boxes_batch = []
    gt_image_shape_batch = []
    for _, _, files in os.walk(xml_path):
        for filename in files:
            xml_data = ''
            with open(os.path.join(xml_path, filename), 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    xml_data += line

            file_start = [m.start() for m in re.finditer('<filename>', xml_data)][0]
            file_end = [m.start() for m in re.finditer('</filename>', xml_data)][0]
            names_start = [m.start() for m in re.finditer('<name>', xml_data)]
            names_end = [m.start() for m in re.finditer('</name>', xml_data)]
            boxes_start = [m.start() for m in re.finditer('<bndbox>', xml_data)]
            boxes_end = [m.start() for m in re.finditer('</bndbox>', xml_data)]
            size_start = [m.start() for m in re.finditer('<size>', xml_data)]
            size_end = [m.start() for m in re.finditer('</size>', xml_data)]

            imagename = xml_data[file_start+len('<filename>'):file_end]
            imagepath = find_path(directory_list, imagename)
            image = cv2.imread(imagepath)
            images.append(image)

            true_idx = []
            for i, row in enumerate(names_start):
                if xml_data[row - 10:row - 4] == 'object': 
                    true_idx.append(i)

            names_start = [names_start[idx] for idx in true_idx]
            names_end = [names_end[idx] for idx in true_idx]
            boxes_start = [boxes_start[idx] for idx in true_idx]
            boxes_end = [boxes_end[idx] for idx in true_idx]

            gt_classes = []
            for i in range(0, len(names_end)):
                gt_classes.append((xml_data[names_start[i] + 6:names_end[i]]))

            gt_boxes = []
            for i in range(0, len(boxes_start)):
                xml_box = xml_data[boxes_start[i]:boxes_end[i]]
                xml_boxes = xml_box.split('\n')[1:-1]
                gt_box = []
                for row in xml_boxes:
                    gt_box.append(int(re.search(r'\d+', row).group()))
                gt_boxes.append(gt_box)

            gt_image_shape_rows = xml_data[size_start[0]:size_end[0]].split('\n')[1:-1]
            for row in gt_image_shape_rows:
                if 'height' in row:
                    height = int(re.search(r'\d+', row).group())
                elif 'width' in row:
                    width = int(re.search(r'\d+', row).group())

            gt_classes_batch.append(np.array(gt_classes))
            gt_boxes_batch.append(np.array(gt_boxes))
            gt_image_shape_batch.append(np.array([height, width], dtype=np.float32))
    images = np.array(images)
    gt_classes_batch = np.array(gt_classes_batch)
    gt_boxes_batch = np.array(gt_boxes_batch)

    xmls = {'images': images, 'image_shape': gt_image_shape_batch, 'classes': gt_classes_batch, 'boxes':gt_boxes_batch}

    return xmls