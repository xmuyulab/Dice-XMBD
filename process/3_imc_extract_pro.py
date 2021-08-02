from warnings import filterwarnings 
filterwarnings('ignore')

from multiprocessing import Process, Pool
import os, time, random
import numpy as np
import scipy.ndimage as ndi
from skimage import measure,color
import os
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import random
import cv2
import tifffile as tiff
from PIL.Image import fromarray as show
import pandas as pd
from tqdm import tqdm
import time
import argparse


def measure_markers(raw_img,raw_mask,measure_label=False):
    if measure_label:
        labels=measure.label(raw_mask,connectivity=2)
    else:
        labels=raw_mask
        
    props = measure.regionprops(labels)
     
    mean_exp_list=[]
    labels_index=list(np.unique(labels))
    labels_index.pop(0)
    cell_pos = []
    cell_area = []
    cell_perimeter = []
    cell_minor_axis = []
    cell_major_axis = []
    for i in labels_index:
        imask=raw_mask.copy()
        imask[labels!=i]=0
        imask[labels==i]=255
        n_pixel=len(imask[imask==255])
        imasked = cv2.add(raw_img, np.zeros(np.shape(raw_img), dtype=np.float64), mask=imask.astype(np.uint8))
        #avg=np.mean(imasked,axis=(0,1))
        avg=np.sum(imasked,axis=(0,1))/n_pixel
        mean_exp_list.append(avg)
        cell_pos.append(props[i-1].centroid)
        cell_area.append(props[i-1].area)
        cell_perimeter.append(props[i-1].perimeter)
        cell_minor_axis.append(props[i-1].minor_axis_length)
        cell_major_axis.append(props[i-1].major_axis_length)
        
        
    mean_exp_matrix=np.matrix(mean_exp_list)
    
    labels_prop = {'index':labels_index, 'pos':cell_pos, 'area':cell_area, 'perimeter':cell_perimeter, 'minor_axis':cell_minor_axis,
                  'major_axis':cell_major_axis}
    return labels_prop,mean_exp_matrix

def matchsize(img, mask):
    x1, y1, z1 = img.shape
    x2, y2 = mask.shape

    mask = mask[:x1,:y1]
    
    print('img: {},{}, mask: {},{}'.format(x1,y1, x2, y2))
    return(mask)


def fun2(i):
    if os.path.exists(os.path.join(output_path,"{0}.csv".format(i))):
        pass
    else:
        raw_img=tiff.imread(os.path.join(img_path,i+img_suffix)).transpose(1,2,0).astype(np.float64)
        raw_mask = tiff.imread(os.path.join(mask_path,i+mask_suffix))
        
        if matchsize:
            raw_mask = matchsize(raw_img, raw_mask)
        
        labels_prop,mean_exp_matrix= measure_markers(raw_img=raw_img,raw_mask=raw_mask,measure_label=True)
        
        marker_exp=pd.DataFrame(mean_exp_matrix,columns=fullstacks_channel_names['marker'])
        #marker_exp.to_csv(os.path.join(output_path,"{0}.csv".format(images[i].split('.')[0])))

        #marker_exp['ImageNumber']=images[i].split('_')[2]
        marker_exp['ObjectNumber']=labels_prop['index']
        marker_exp['Position']=labels_prop['pos']
        marker_exp['area'] = labels_prop['area']
        marker_exp['perimeter'] = labels_prop['perimeter']
        marker_exp['minor_axis'] = labels_prop['minor_axis']
        marker_exp['major_axis']= labels_prop['major_axis']
        #marker_exp['metabricId'] = images[i].split('_')[0].replace('MB','MB-')
        #marker_exp['core_id'] = images[i].split('_')[1]

        #marker_exp.to_csv(os.path.join(output_path,"{0}.csv".format(images[i].split('.')[0])))
        marker_exp.to_csv(os.path.join(output_path,"{0}.csv".format(i)))
        print(os.path.join(output_path,"{0}.csv".format(i)))
        return marker_exp
    
    
####  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str, default="/workspace/data/predict_BRCA2/", help='the path of the data folder')
    parser.add_argument('--name', type=str, default="CP", help='the name of the data folder')
    parser.add_argument('--panel', type=str, default='/workspace/data/panel/BRCA2.csv', help='the path of the panel file => same ordered with img channel: at least need contain two columns("nuclear","membrane_cytoplasm")')
    parser.add_argument('--imgsuf', type=str, default='_fullstack.tiff', help='images suffix')
    parser.add_argument('--masksuf', type=str, default='_pred_Probabilities_cell_mask.tiff', help='mask images suffix')
    parser.add_argument('--imgdir', type=str, default="/mnt/davinci/temp/kong_IMC/analysis/fullstacks", help='the name of the data folder')
    parser.add_argument('--maskpath', type=str, default='/workspace/data/predict_BRCA2/mask')
    parser.add_argument('--matchsize', type=str, default='False')
    args = parser.parse_args()
    
    workdir = args.workdir
    panelpath = args.panel
    mask_path = args.maskpath
    pname = args.name
    img_path = args.imgdir
    fullstacks_channel_names = pd.read_csv(panelpath)
    img_suffix = args.imgsuf
    mask_suffix = args.masksuf
    
    if args.matchsize == 'False':
        matchsize = False
    else:
        matchsize = True
    
    output_path = os.path.join(workdir, 'protein', pname)
    
    masks=np.sort(os.listdir(mask_path))
    masks = [i for i in masks if 'cell_mask' in i]
    
    id1 = [i.split(img_suffix)[0] for i in os.listdir(img_path) if 'fullstack' in i]
    id2 = [i.split(mask_suffix)[0] for i in os.listdir(mask_path) if 'cell_mask' in i]
    img_name = list(set(id1) & set(id2))
    image_name = np.sort(img_name)
    print('process {} images'.format(len(image_name)))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    t0 = time.time()
    with Pool(40) as p:
        result2 = p.map(fun2,[i for i in image_name])
    print(time.time() - t0, "seconds process time")