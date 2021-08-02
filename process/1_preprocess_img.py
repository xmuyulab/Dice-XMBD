import os
import tifffile as tiff
import pandas as pd
import numpy as np
import argparse 
import math

def combine_channels(images_path, panel_path, channel_type, out_path, imgsuffix, imagej=True):
    images=[]
    for fn in os.listdir(images_path):
        if fn.endswith(imgsuffix):
            images.append(fn)
    print('process images: {}'.format(images))        
    print('process images number: {}'.format(len(images)))
    panel=pd.read_csv(panel_path)

    for i in range(len(images)):
        img=tiff.imread(os.path.join(images_path, images[i]))
        if img.shape[0] > 3:
            result=np.zeros((len(channel_type),img.shape[1],img.shape[2]))
            for j in range(len(channel_type)):#["nuclear","membrane_cytoplasm"]
                channels=list(panel[panel[channel_type[j]]==1].index)
                result[j,:,:] = np.sum(img[channels,:,:],axis=0)
            #print(result.shape)
            tiff.imwrite(os.path.join(out_path,images[i].replace(imgsuffix.split('.tiff')[0],'_combined')),result,imagej=imagej)


def combine2Img(images_path, combined_path, panel_info, imgsuffix, channel_nuc='nuclear', channel_mem='membrane_cytoplasm'):
#     images_path="/mnt/davinci/temp/kong_IMC/analysis/fullstacks/"
#     combined_path="/mnt/davinci/temp/kong_IMC/analysis/unet/combine2"

    if not os.path.exists(combined_path):
        os.makedirs(combined_path)

    combine_channels(images_path=images_path,
                    panel_path = panel_info,
                    channel_type=[channel_nuc,channel_mem],
                    out_path=combined_path,
                    imgsuffix = imgsuffix,
                    imagej=False)

def cropImg(crop512_combined_path, combined_path):
    
#     crop512_combined_path="/mnt/davinci/temp/kong_IMC/analysis/unet/crop512_combined2_image"
    if not os.path.exists(crop512_combined_path):
        os.makedirs(crop512_combined_path)

    for file in os.listdir(combined_path):
        img=tiff.imread(os.path.join(combined_path,file))
        z, y, x = img.shape

        if (y > 100) and (x > 100):
            for i in range(0, x, 512):
                for j in range(0, y, 512):
                    pos = str(i)+'_'+str(j)
                    img512 = img[:,i:i+512,j:j+512]
                    z2, y2, x2 = img512.shape
                    if (y2 < 512) or (x2 < 512):
                        padimg= np.pad(img512, ((0,0), (0, 512-y2), (0, 512-x2)), 'constant', constant_values=(0, 0))
                        img512 = padimg
                    tiff.imwrite(os.path.join(crop512_combined_path,file.replace("combined.tiff","combined_cropx{}_cropy{}.tiff".format(str(i),str(j)))),img512,imagej=False)
                            
                    
def mergeImg(prop_path, combined_path, out_path):
    
    files = os.listdir(prop_path)
    files = [i for i in files if '_pred_Probabilities' in i]
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    sampledic = {}

    for each in files:
        sample = each.split('_combined_cropx')[0]
        if sample not in sampledic.keys():
            sampledic[sample] = [each]
        else:
            sampledic[sample].append(each)
            
            
    for each in sampledic.keys():
        rois = sampledic[each]
        imgs = {}
        for roi in rois:
            x = roi.split('_pred_Probabilities')[0].split('_crop')[1][1:]
            y = roi.split('_pred_Probabilities')[0].split('_crop')[2][1:]
            imgs[str(x)+'_'+str(y)] = tiff.imread(os.path.join(prop_path, roi))
        original = tiff.imread(os.path.join(combined_path, each+'_combined.tiff'))

        z, y, x = original.shape
        ori_prop = np.zeros((512*math.ceil(x/512),512*math.ceil(x/512),3))

        for i in range(0, x, 512):
            for j in range(0, y, 512):
                pos = str(i)+'_'+str(j)
                ori_prop[i:i+512,j:j+512] = imgs[pos]
                

        tiff.imwrite('{}/{}_pred_Probabilities.tiff'.format(out_path,each),ori_prop.astype('uint16'))
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, help='pre or post')
    parser.add_argument('--workdir', type=str, default="/workspace/data/predict_BRCA2/", help='the path of the data folder')
    parser.add_argument('--propn', type=str, default="BRCA1_threshold-99.7_withAugnoise-0.5", help='the name of predicted probability in the data folder')
    parser.add_argument('--panel', type=str, default='/workspace/data/panel/BRCA2.csv', help='the path of the panel file => same ordered with img channel: at least need contain two columns("nuclear","membrane_cytoplasm")')
    parser.add_argument('--channel_n', type=str, default='nuclear', help='the column name for nuclear channel in panel file')
    parser.add_argument('--channel_m', type=str, default='membrane_cytoplasm', help='the column name for membrane channel in panel file')
    parser.add_argument('--imgdir', type=str, default="/workspace/data/predict_BRCA2/fullstacks/", help='the path of the img data folder')
    parser.add_argument('--imgsuf', type=str, default='_fullstack.tiff', help='images suffix')
    args = parser.parse_args()
    
    workdir = args.workdir
    process = args.process
    panelpath = args.panel
    imgsuffix = args.imgsuf
    imgdir = args.imgdir
    prop_path_name = args.propn
    channel_mem = args.channel_m
    channel_nuc = args.channel_n
    
    if process == 'pre':
        combine2Img(imgdir, os.path.join(workdir, 'combined2_image'), panelpath, imgsuffix, channel_nuc=channel_nuc, channel_mem=channel_mem)
        cropImg(os.path.join(workdir, 'crop512_combined2_image'), os.path.join(workdir, 'combined2_image'))
    elif process == 'post':
        prop_path = os.path.join(workdir, 'predictionProbability', prop_path_name)
        combined_path = os.path.join(workdir, 'combined2_image')
        out_path = os.path.join(prop_path, 'ori_prop')
        mergeImg(prop_path, combined_path, out_path)
    