import os

import cv2
import numpy as np


MODULE_DIR = os.path.dirname(__file__)


def adjust_img_color_space(img, in_cspace='RGB', out_cspace='RGB'):
    if in_cspace == out_cspace:
        return np.copy(img)

    if in_cspace == 'RGB':
        if out_cspace == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif out_cspace == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif out_cspace == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif out_cspace == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif out_cspace == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif in_cspace == 'HSV':
        if out_cspace == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        elif out_cspace == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2LUV)
        elif out_cspace == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2HLS)
        elif out_cspace == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2YUV)
        elif out_cspace == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2YCrCb)
    elif in_cspace == 'LUV':
        if out_cspace == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_LUV2RGB)
    elif in_cspace == 'HLS':
        if out_cspace == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    elif in_cspace == 'YUV':
        if out_cspace == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    elif in_cspace == 'YCrCb':
        if out_cspace == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    elif in_cspace == 'BGR':
        if out_cspace == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # not sure if this copy is required
        img = np.copy(img)
    return img


if __name__ == '__main__':
    in_path = os.path.join(MODULE_DIR, '../video_output_images/f1_ip.jpg')
    out_path = os.path.join(MODULE_DIR, '../video_output_images/f1_ip_color.jpg')
    in_img = cv2.imread(in_path)
    out_img = adjust_img_color_space(in_img, in_cspace='BGR')
    cv2.imwrite(out_path, out_img)
