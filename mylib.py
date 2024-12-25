import numpy as np


def flip_crater(img):
    '''
    Flips crater s.t. the shadow will always be on the r.h.s
    '''
    qtr_img_width = np.int16(img.shape[1]/4)
    half_img_width = np.int16(img.shape[1]/2)
    
    left_crater_side = img[:, qtr_img_width:half_img_width]
    right_crater_side = img[:, half_img_width:-qtr_img_width]

    if left_crater_side.mean() > right_crater_side.mean():
        pass
    else:
        img = np.fliplr(img)
        
    return img
    

def reflect_crater(img):    
    qtr_img_width = np.int16(img.shape[1]/4)
    half_img_width = np.int16(img.shape[1]/2)

    # Find side with shadow
    left_side = img[:, :half_img_width]
    right_side = img[:, half_img_width:]
    
    left_crater_side = img[:, qtr_img_width:half_img_width]
    right_crater_side = img[:, half_img_width:-qtr_img_width]

    if left_crater_side.mean() > right_crater_side.mean():
        img[:, half_img_width:] = np.fliplr(left_side)
    else:
        img[:, :half_img_width] = np.fliplr(right_side)
        
    return img