# TODO: decide if we want to get the input and output images as parameters or as constants in a specific directory
# TODO: connecting to REQUEST and RESPONSE modules
# TODO: decide if we want to use GPU

"""       
This module coloring black & white images.
The input image and the output images are the program's parameters

MODULE_REQUIREMENTS:
This module require that 'colorization_release_v2.caffemodel' exists, 
there are two ways to deal with that:
1. Download that file (123 Mb) with an HTTP request. Done with running the following command:
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O ./models/colorization_release_v2.caffemodel

2. Download this file to the project (and change the path of CAFFE_MODEL constant)


HOW TO USE THE MODULE:
There are two different ways to runs the program: (option 1 is default) 
1) python3 ./coloringBW.py -img_in <path_to_input_image> -img_out <path_to_output_image>
2) (The constant way) python3 ./coloringBW 
    [needs to set the constant INPUT_IMAGE, OUTPUT_IMAGE first, and MODULE_USE constant as 2]


Another Comments:
1. There is an option to use GPU, just set GPU_USE true and GPU_DEVICE constant as 
    the gpu device you want to use.
"""

import numpy as np
import os
import skimage.color as color
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni
import caffe
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='iColor: deep interactive colorization')
    parser.add_argument('-img_in', dest='img_in', help='grayscale image to read in', type=str)
    parser.add_argument('-img_out', dest='img_out', help='colorized image to save off', type=str)

    args = parser.parse_args()
    return args


MODEL_DIR = './'
PROTOTXT = './colorization_deploy_v2.prototxt'
CAFFE_MODEL = './colorization_release_v2.caffemodel'

MODULE_USE = 1      # or MODULE_USE = 2 (in case of using option 2)
GPU_USE = False     # or GPU_USE = true (in case of using gpu)

# Define this constant in case of using the GPU
GPU_DEVICE = 0

# Define these constants in case of using the module as option 2
INPUT_IMAGE = ''    # example: './demo/ansel_adams.jpg'
OUTPUT_IMAGE = ''   # example: './demo/out.png'

if __name__ == '__main__':
    args = parse_args()

    if GPU_USE:
        caffe.set_mode_gpu()
        caffe.set_device(GPU_DEVICE)
    else:
        caffe.set_mode_cpu()


    # Select desired model
    net = caffe.Net(PROTOTXT, CAFFE_MODEL, caffe.TEST)

    (H_in, W_in) = net.blobs['data_l'].data.shape[2:]  # get input shape
    (H_out, W_out) = net.blobs['class8_ab'].data.shape[2:]  # get output shape

    pts_in_hull = np.load('./resources/pts_in_hull.npy')  # load cluster centers
    net.params['class8_ab'][0].data[:, :, 0, 0] = pts_in_hull.transpose(
        (1, 0))  # populate cluster centers as 1x1 convolution kernel
    # print 'Annealed-Mean Parameters populated'

    # load the original image
    img_rgb = caffe.io.load_image(args.img_in)
    if MODULE_USE == 2:
        img_rgb = caffe.io.load_image(INPUT_IMAGE)

    img_lab = color.rgb2lab(img_rgb)  # convert image to lab color space
    img_l = img_lab[:, :, 0]  # pull out L channel
    (H_orig, W_orig) = img_rgb.shape[:2]  # original image size

    # create grayscale version of image (just for displaying)
    img_lab_bw = img_lab.copy()
    img_lab_bw[:, :, 1:] = 0
    img_rgb_bw = color.lab2rgb(img_lab_bw)

    # resize image to network input size
    img_rs = caffe.io.resize_image(img_rgb, (H_in, W_in))  # resize image to network input size
    img_lab_rs = color.rgb2lab(img_rs)
    img_l_rs = img_lab_rs[:, :, 0]

    net.blobs['data_l'].data[0, 0, :, :] = img_l_rs - 50  # subtract 50 for mean-centering
    net.forward()  # run network

    ab_dec = net.blobs['class8_ab'].data[0, :, :, :].transpose((1, 2, 0))  # this is our result
    ab_dec_us = sni.zoom(ab_dec,
                         (1. * H_orig / H_out, 1. * W_orig / W_out, 1))  # upsample to match size of original image L
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)  # concatenate with original image L
    img_rgb_out = (255 * np.clip(color.lab2rgb(img_lab_out), 0, 1)).astype('uint8')  # convert back to rgb

    if MODULE_USE == 2 and OUTPUT_IMAGE != '':
        plt.imsave(OUTPUT_IMAGE, img_rgb_out)
    else:
        plt.imsave(args.img_out, img_rgb_out)
