import imageio, json
from PIL import Image
import argparse
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from helper_functions import read_leads, convert_bounding_boxes_to_dict, rotate_bounding_box, get_lead_pixel_coordinate, rotate_points
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
import time
import random

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_directory', type=str, required=True)
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_directory', type=str, required=True)
    parser.add_argument('-r','--rotate',type=int,default=25)
    parser.add_argument('-n','--noise',type=int,default=25)
    parser.add_argument('-c','--crop',type=float,default=0.01)
    parser.add_argument('-t','--temperature',type=int,default=6500)
    return parser

# Main function for running augmentations
def get_augment(input_file,output_directory,rotate=25,noise=25,crop=0.01,temperature=6500,bbox=False, store_text_bounding_box=False, json_dict=None):
    filename = input_file
    image = Image.open(filename)
    
    image = np.array(image)
    
    lead_bbs = []
    leadNames_bbs = []
    
         
    lead_bbs, leadNames_bbs, lead_bbs_labels, startTime_bbs, endTime_bbs, plotted_pixels = read_leads(json_dict['leads'])
    
    if bbox:
        lead_bbs = BoundingBoxesOnImage(lead_bbs, shape=image.shape)
    if store_text_bounding_box:
        leadNames_bbs = BoundingBoxesOnImage(leadNames_bbs, shape=image.shape)
    
    images = [image[:, :, :3]]
    h, w, _ = image.shape
    rot = random.randint(-rotate, rotate)
    crop_sample = random.uniform(0, crop)
    #Augment in a sequential manner. Create an augmentation object
    seq = iaa.Sequential([
          iaa.Affine(rotate=rot),
          iaa.AdditiveGaussianNoise(scale=(noise, noise)),
          iaa.Crop(percent=crop_sample),
          iaa.ChangeColorTemperature(temperature)
          ])
    
    images_aug = seq(images=images)

    if bbox:
        augmented_lead_bbs = rotate_bounding_box(lead_bbs, [h/2,w/2], -rot)
    else:
        augmented_lead_bbs = []    
    if store_text_bounding_box:
        augmented_leadName_bbs = rotate_bounding_box(leadNames_bbs, [h/2,w/2], -rot)
    else:
        augmented_leadName_bbs = []   

    rotated_pixel_coordinates = rotate_points(plotted_pixels, [h/2, w/2], -rot)

    if bbox or store_text_bounding_box:
        json_dict['leads'] = convert_bounding_boxes_to_dict(augmented_lead_bbs, augmented_leadName_bbs, lead_bbs_labels, startTime_bbs, endTime_bbs, rotated_pixel_coordinates)

    head, tail = os.path.split(filename)

    f = os.path.join(output_directory,tail)
    plt.imsave(fname=f,arr=images_aug[0])

    return f

