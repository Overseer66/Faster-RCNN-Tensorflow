import cv2
import copy
import numpy as np

from config import config as CONFIG


def ImageSetExpand(image_set):
    image_set = copy.deepcopy(image_set)
    set_size = len(image_set['images'])

    image_set['org_boxes'] = []
    for idx in range(set_size):
        

        image = image_set['images'][idx]
        boxes = image_set['boxes'][idx]

        width = image.shape[1]
        height = image.shape[0]
        
        img_min_size = min(width, height)
        scale = CONFIG.TARGET_SIZE / img_min_size

        image_set['images'][idx] = ImageExpand(image, scale)
        image_set['image_shape'][idx] *= scale
        image_set['image_shape'][idx] = np.append(image_set['image_shape'][idx], scale)
        image_set['boxes'][idx] = [box * scale for box in boxes]
        image_set['org_boxes'].append(boxes)


    
    return image_set



def ImageExpand(img, scale):
    img = img.astype(np.float32)
    img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    return img
