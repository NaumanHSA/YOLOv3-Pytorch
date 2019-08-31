from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools
import GrabCut as gc
from matplotlib import pyplot as plt
from PIL import Image

OUT_PATH = 'det/'

class test_net(nn.Module):
    def __init__(self, snum_layer, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)

def arg_parse():

    # Parse arguements to the detect module
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()
    scales = args.scales

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('data/coco.names')

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    #Set the model in evaluation mode
    model.eval()
    
    read_dir = time.time()
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    if not os.path.exists(args.det):
        os.makedirs(args.det)
        
    load_batch = time.time()
    
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    image_names = [x[3] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        

    i = 0

    write = False
    # model(get_test_input(inp_dim, CUDA), CUDA)
    
    start_det_loop = time.time()
    
    objs = {}
    
    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
        
#        prediction = prediction[:,scale_indices]

        #get the boxes with object confidence > threshold
        #Convert the cordinates to absolute coordinates
        #perform NMS on these boxes, and save the results 
        #I could have done NMS and saving seperately to have a better abstraction
        #But both these operations require looping, hence 
        #clubbing these ops in one loop instead of two. 
        #loops are slower than vectorised operations. 
        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        end = time.time()

        prediction[:,0] += i*batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        #     print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        #     print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        #     print("----------------------------------------------------------")
        i += 1

        
        if CUDA:
            torch.cuda.synchronize()
    
    try:
        output
    except NameError:
        print("No detections were made")
        exit()
        
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

    output_recast = time.time()
    class_load = time.time()

    colors = pkl.load(open("pallete", "rb"))

    draw = time.time()

    cell_data = []
    person_data = []

    def annotate(top_left, bottom_right, image_path):

        print(os.path.join(images, image_path))
        image = Image.open(os.path.join(images, image_path))

        image_width = image.size[0]
        image_height = image.size[1]

        x1, y1 = int(top_left[0]), int(top_left[1])
        x2, y2 = int(bottom_right[0]), int(bottom_right[1])

        dw = 1. / image_width
        dh = 1. / image_height
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        x = x * dw
        y = y * dh
        w = w * dw
        h = h * dh

        labelfilename = os.path.join(images, (image_path.split('.')[0]) + '.txt')

        if labelfilename:
            with open(labelfilename, 'a+') as f:
                f.write('1 ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')

            print('file annotated : ' + image_path)

    def write(x, batches, results, names):

        images = None
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        name = names[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        # color = random.choice(colors)
        color = colors[3]
        cv2.rectangle(img, c1, c2, color, 3)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c3 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c3, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

        print(label)
        if label == 'person':
            annotate(c1, c2, name)

    images_all = list(map(lambda x: write(x, im_batches, orig_ims, image_names), output))

    # count = 0
    # for x in cell_data:
    #     x_center_c = int(((x[1][0] - x[0][0])/2) + x[0][0])
    #     y_center_c = int(((x[1][1] - x[0][1])/2) + x[0][1])
    #
    #     # print(str(x_center_c) + ' ' + str(y_center_c))
    #
    #     for y in person_data:
    #         x_center_p = int(((y[1][0] - y[0][0]) / 2) + y[0][0])
    #         y_center_p = int(((y[1][1] - y[0][1]) / 2) + y[0][1])
    #
    #         if abs(x_center_c - x_center_p) <= (((y[1][0] - y[0][0]) / 2)) and abs(y_center_c - y_center_p) <= (y[1][1] - y[0][1]):
    #             # print(str(x_center_c) + ' ' + str(y_center_c))
    #             # print(str(x_center_p) + ' ' + str(y_center_p) + '\n')
    #             # image = gc.grabcut(y[0], y[1], imlist[0])
    #             img = cv2.imread(imlist[0])
    #             # print(str(int(y[0][0])) + ' ' + str(int(y[0][1])))
    #             # print(str(int(y[1][0])) + ' ' + str(int(y[1][1])))
    #             image = img[int(y[0][1]):int(y[1][1]), int(y[0][0]):int(y[1][0])]
    #             # cv2.imshow('Image', image)
    #             # cv2.waitKey(0)
    #             cv2.imwrite(OUT_PATH + 'ext_' + str(count) + '.jpg', image)
    #             count += 1

    #
    # count = 0
    # for x in orig_ims:
    #     cv2.imwrite(OUT_PATH + 'img_' + str(count) + '.jpg', x)
    #     count += 1
    
    end = time.time()
    
    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()