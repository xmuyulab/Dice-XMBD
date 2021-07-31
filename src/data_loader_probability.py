import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
import torch
import tifffile as tiff
from PIL import Image
from skimage.transform import resize
import scipy.ndimage as scind
import random
import cv2



def read_tiff(img_path):
    img = tiff.imread(img_path)  #array
    img = img.astype(np.float64)
    return img


#normalization
def equalizeHist(img):
    img1=(img-np.min(img))/(np.max(img)-np.min(img))
    img1=img1*255
    equ = cv2.equalizeHist(img1.astype(np.uint8))
    return equ



def to3channel(array):
    if array.shape[0] == 2:
        array1=np.zeros((array.shape[0]+1,array.shape[1],array.shape[2]))
        array1[0:2,:,:]=array
        return array1
        
        
# transform [w,h],[n,w,h] images into [w,h],[w,h,n]
# 1. 2D image: change to int8 format
# 2. 3D image: first transpose, then change to int8 format 
def toPILImage(array):
    if len(array.shape) == 2:
        img=Image.fromarray(array.astype('uint8'))
    elif array.shape[0] == 2:
        array1=np.zeros((array.shape[0]+1,array.shape[1],array.shape[2]))
        array1[0:2,:,:]=array
        img=Image.fromarray(array1.transpose(1,2,0).astype('uint8'))
    elif array.shape[0] == 3:
        img=Image.fromarray(array.transpose(1,2,0).astype('uint8'))
    return img
    
    
def normalize(img):
    img = img - np.min(img)
    img = img / np.max(img) 
    return img


###Clips hot pixels to the maximum local neighbor intensity. Hot pixels are identified by thresholding on the difference
# between the pixel intensity and it's maximum local neighbor intensity.
def clip_hot_pixels(img, hp_filter_shape=[3,3], hp_threshold=50.0):
    if hp_filter_shape[0] % 2 != 1 or hp_filter_shape[1] % 2 != 1:
        raise ValueError("Invalid hot pixel filter shape: %s" % str(hp_filter_shape))
    hp_filter_footprint = np.ones(hp_filter_shape)
    hp_filter_footprint[int(hp_filter_shape[0] / 2), int(hp_filter_shape[1] / 2)] = 0
    #filter：
    #1 1 1
    #1 0 1
    #1 1 1
    max_img = scind.maximum_filter(img, footprint=hp_filter_footprint, mode='reflect')
    hp_mask = img - max_img > hp_threshold
    img = img.copy()
    img[hp_mask] = max_img[hp_mask]
    return img

def trans_binary(label):
    weight, high = label.shape
    for i in range(weight):
        for j in range(high):
            if label[i][j]>0:
                label[i][j]=1
    return label

def select_class_1(label, c):
    label_c = np.copy(label)
    label_c = (label_c==c).astype(np.uint8)
    return label_c

def one_hot_encode_truth(truth, num_class=23):# truth [512, 512]>>>>>[22, 512, 512]
    truth = torch.from_numpy(truth)

    truth = truth.unsqueeze(0)
    one_hot = truth.repeat(num_class,1,1) 
    arange  = torch.arange(1,num_class+1).view(num_class,1,1) #[1,2,3,4,...22]
    one_hot = (one_hot.int() == arange.int()).int()

    one_hot = one_hot.numpy().astype(np.uint8)
    return one_hot


def pad_one_image(array):
    _, width, height = array.shape
    pad_w = 0
    pad_h = 0
    
    # resize image
    if width<513:
        pad_w = 513-width
    if height<513:
        pad_h = 513-height

    array_pad = np.pad(array,((0, 0),
                              (pad_w//2, pad_w-pad_w//2),
                              (pad_h//2, pad_h-pad_h//2)),
                              mode="constant",
                              constant_values=(0,0))
    
    return array_pad


def pad_image(array, label,background=False):
    _, width, height = array.shape
    pad_w = 0
    pad_h = 0

    if width<513:
        pad_w = 513-width
    if height<513:
        pad_h = 513-height

    array_pad = np.pad(array,((0, 0),
                              (pad_w//2, pad_w-pad_w//2),
                              (pad_h//2, pad_h-pad_h//2)),
                              mode="constant",
                              constant_values=(0,0))
    label_pad = np.pad(label,((0, 0),
                              (pad_w//2, pad_w-pad_w//2),
                              (pad_h//2, pad_h-pad_h//2)),
                              mode="constant",
                              constant_values=(0,0))
    if background:
        label_pad[2,:,:] = np.pad(label[2,:,:],(
                              (pad_w//2, pad_w-pad_w//2),
                              (pad_h//2, pad_h-pad_h//2)),
                              mode="constant",
                              constant_values=(1,1))
    return array_pad, label_pad


def one_randomcrop(crop_size, image):
    z, x, y = image.shape
    size_z, size_x, size_y = crop_size

    if x < size_x or y < size_y or z < size_z:
        raise ValueError

    #z1 = np.random.randint(0, z - size_z)
    x1 = np.random.randint(0, x - size_x)
    y1 = np.random.randint(0, y - size_y)

    image = image[0: image.shape[0], x1: x1 + size_x, y1: y1 + size_y]

    return image


def randomcrop(crop_size, image, label):
    z, x, y = image.shape
    size_z, size_x, size_y = crop_size

    if x < size_x or y < size_y or z < size_z:
        raise ValueError

    #z1 = np.random.randint(0, z - size_z)
    x1 = np.random.randint(0, x - size_x)
    y1 = np.random.randint(0, y - size_y)

    image = image[0: image.shape[0], x1: (x1 + size_x), y1: (y1 + size_y)]
    label = label[0:label.shape[0], x1: (x1 + size_x), y1: (y1 + size_y)]   

    return image, label

def normalize_maxmin(a):
    for i in range(a.shape[0]):
        x=a[i,:,:]
        nor=(x-np.min(x))/(np.max(x)-np.min(x))
        nor=nor*255
        nor=nor.astype(np.uint8)
        a[i,:,:]=nor
    return a

#data.Dataset:
class CellDataset(data.Dataset):
    #创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self,image_dir,label_dir, clip=True,threshold=False,normalize=True,equalizeHistAug=False, resize_crop=True,transform = None,label_transform = None,com_transform=None,noise_transform=None):
        #n = len(os.listdir(train_dir)) #os.listdir(path)
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        img_list = []
        for img in np.sort(os.listdir(os.path.join(image_dir))):
            img_path = os.path.join(image_dir, img)
            label_path = os.path.join(label_dir, img.replace('combined', 'ilastik_s2_Probabilities')) 
            img_list.append([img_path, label_path])
        
        self.img_list = img_list
        self.transform = transform
        self.label_transform = label_transform
        self.com_transform=com_transform
        self.noise_transform=noise_transform
        self.clip=clip
        self.threshold=threshold
        self.normalize=normalize
        self.equalizeHistAug=equalizeHistAug
        self.resize_crop=resize_crop
        
    
    
    def __getitem__(self,index):
        x_path, y_path = self.img_list[index]
        image_name=x_path.split("/")[-1]
        img_x = read_tiff(x_path)
        if self.clip:
            img_x[0,:,:]=clip_hot_pixels(img_x[0,:,:])
            img_x[1,:,:]=clip_hot_pixels(img_x[1,:,:])
        if self.threshold:
            th0=np.percentile(img_x[0,:,:], self.threshold)
            th1=np.percentile(img_x[1,:,:], self.threshold)
            img_x[0,:,:][img_x[0,:,:]>th0]=th0
            img_x[1,:,:][img_x[1,:,:]>th1]=th1
        if self.normalize:
            img_x=normalize_maxmin(img_x)
        if self.equalizeHistAug:
            img_x[0,:,:]=equalizeHist(img_x[0,:,:])
            img_x[1,:,:]=equalizeHist(img_x[1,:,:])
        
                        
        #2. read img, normalization, transpose to (n*w*h)
        img_y0 = read_tiff(y_path)
        img_y0=img_y0/(2**16-1)
        img_y0=np.around(img_y0, 1)
        
        if img_y0.shape[0] != img_x.shape[1]:
            self.resize_crop = True
        else:
            self.resize_crop = False
        
        if self.resize_crop:
            img_y=resize(img_y0, (img_y0.shape[0]/2,img_y0.shape[1]/2, img_y0.shape[2]), mode='constant', preserve_range=True)
            img_y=img_y.transpose(2,0,1)
            
            img_x, img_y = pad_image(img_x, img_y,background=True)
            img_x, img_y = randomcrop([img_x.shape[0],512,512],img_x, img_y)
        else:
            img_y=img_y0.transpose(2,0,1)
            
        #augmenation    
        if self.transform is not None:
            img_x=toPILImage(img_x)
            img_x = self.transform(img_x)#input data 512*512*3
            img_x = np.asarray(img_x).transpose(2,0,1)
        else:
            img_x = to3channel(img_x)
        if self.label_transform is not None:
            img_y = self.label_transform(img_y)#2*512*512
               
        if self.com_transform is not None:
            img_x=toPILImage(img_x)
            img_y=Image.fromarray(np.uint8(img_y.transpose(1,2,0)*255))
                
            img_x,img_y = self.com_transform([img_x,img_y])
                      
            img_x = np.asarray(img_x).transpose(2,0,1)
            img_y = np.around((np.asarray(img_y)/255),1).transpose(2,0,1)
        
        #add noise      
        if self.noise_transform is not None:
            img_x=self.noise_transform(img_x)
            img_x=np.uint8(img_x)
        
        return img_x,img_y,image_name
    
    
    def __len__(self):
        return len(self.img_list)

class PredictCellDataset(data.Dataset):
    def __init__(self,image_dir,clip=True,threshold=False,normalize=True,equalizeHistAug=False,resize_crop=True, transform = None):
        self.image_dir = image_dir
        self.clip=clip
        self.threshold=threshold
        self.normalize=normalize
        self.equalizeHistAug=equalizeHistAug
        self.resize_crop=resize_crop
        img_list = []
        
        for img in np.sort(os.listdir(os.path.join(image_dir))):
            img_path = os.path.join(image_dir, img)
            
            img_list.append(img_path)
           
        self.img_list = img_list
        self.transform = transform
        
    def __getitem__(self,index):
        x_path = self.img_list[index]
        image_name=x_path.split("/")[-1]
        img_x = read_tiff(x_path)#input data float32-->uint8
        if self.clip:
            img_x[0,:,:]=clip_hot_pixels(img_x[0,:,:])
            img_x[1,:,:]=clip_hot_pixels(img_x[1,:,:])
        if self.threshold:
            th0=np.percentile(img_x[0,:,:], self.threshold)
            th1=np.percentile(img_x[1,:,:], self.threshold)
            img_x[0,:,:][img_x[0,:,:]>th0]=th0
            img_x[1,:,:][img_x[1,:,:]>th1]=th1
        if self.normalize:
            img_x=normalize_maxmin(img_x)

        if self.equalizeHistAug:
            img_x[0,:,:]=equalizeHist(img_x[0,:,:])
            img_x[1,:,:]=equalizeHist(img_x[1,:,:])
            
        #pad
        if self.resize_crop:
            img_x = pad_one_image(img_x)
            img_x = one_randomcrop([img_x.shape[0],512,512],img_x)

        if self.transform is not None:
            img_x = self.transform(img_x)
        else:
            img_x = to3channel(img_x)
        return img_x
    
    
    def __len__(self):
        return len(self.img_list)
    
    
# if __name__ == '__main__':
    
#     train_dir = '/mnt/md0/qy/project/unet/full_stacks'
#     label_dir = '/mnt/md0/qy/project/unet/probability'
#     cell_dataset = CellDataset(train_dir, label_dir)
#     dataloader = DataLoader(cell_dataset, batch_size=1)
#     print(len(dataloader))
#     for x,y in dataloader:
#         print(x.shape)
#         print(y.shape)