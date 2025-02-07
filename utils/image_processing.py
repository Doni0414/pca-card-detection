import cv2
import numpy as np


def resize_image_keep_ratio(image, max_height, max_width):
    h, w = image.shape[0], image.shape[1]

    #######################
    # YOUR CODE GOES HERE #
    #######################
    ratio = w / h
    if w > max_width:
        w = max_width
        h = int(w / ratio)
    elif h > max_height:
        h = max_height
        w = int(h * ratio)
    image = cv2.resize(image, (w, h))
    return image


def get_card_mask_for_image(image, card_ratio, scale):
    h, w, = image.shape[0], image.shape[1]

    #######################
    # YOUR CODE GOES HERE #
    #######################
    mask = np.ones((h,w,3))
    
    if w > h:
        card_w = w
        card_h = int(card_w * card_ratio)

        h_diff = h - card_h
        h_outer_pad = int(h_diff / 2)

        mask[h_outer_pad:card_h + h_outer_pad,:,:] = 1

        inner_w = int(card_w * scale)
        inner_h = int(card_h * scale)

        w_inner_diff = card_w - inner_w
        h_inner_diff = card_h - inner_h
        top = int(h_inner_diff / 2)
        left = int(w_inner_diff / 2)

        mask[h_outer_pad + top:inner_h + h_outer_pad + top,left:inner_w + left,:] = 0
    else:
        card_h = h
        card_w = int(card_h * card_ratio)
        #TODO
        w_diff = w - card_w

    
    return mask
