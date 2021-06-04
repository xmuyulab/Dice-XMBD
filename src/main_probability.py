import torch
import numpy as np
import random
import os
from data_loader_probability import CellDataset
from torch.utils.data import DataLoader
import unet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse 
from torch import optim
import tifffile as tiff
from loss import BinaryDiceLoss
import pandas as pd
import torchvision.transforms as transforms
import datetime
import torchvision.transforms.functional as tf
from self_augmentation import *
from data_loader_probability import PredictCellDataset
import stat

# Use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

time=datetime.datetime.now().strftime("%m-%d-%H-%M")


#1. Image flipping
def my_transform1(image_mask):
    image=image_mask[0]
    mask=image_mask[1]
    # Random horizontal or vertical flipping with 50% probability
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    if random.random() > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    return image, mask

#2. Image rotation
def my_transform2(image_mask):
    image=image_mask[0]
    mask=image_mask[1]

    # the rotating angle is randomly distributed in the range of [-180, 180]
    angle = transforms.RandomRotation.get_params([-180, 180])
    # do same operation on image and corresponding mask to make sure rotate them with same angle
    image = image.rotate(angle)
    mask = mask.rotate(angle)
    return image, mask

#3. Add random Gaussian noise    
def noise_augmentation(image):
    if random.randint(0,1)>0:
        aug_img=add_gaussian_noise(image)
        aug_img=np.uint8(aug_img)
    else:
        aug_img=np.uint8(image)
    return aug_img

def Dice(pred, label):
    if (pred.sum() + label.sum())==0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def binary(predict):
    predict_y = np.copy(predict)
    predict_y = (predict_y>0.5).astype(np.uint8)
    return predict_y

def trans_255(image):
    weight, high = image.shape
    for i in range(weight):
        for j in range(high):
            if image[i][j]>0:
                image[i][j]=255
    return image



n_input_channel=3
n_output_channel=3
def train():
    model = unet.UNet(n_input_channel,n_output_channel).to(device)
    #loss function
    criterion = torch.nn.BCELoss()
    #gradient descent
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)
    
    images_name=np.sort(os.listdir(train_img_dir))
    #data augmentation
    if "withAug" in augmentation:
        print("Augmentation coefficient:", augmentation_cof)
        trans_train = transforms.Compose(transforms = [
                transforms.ColorJitter(brightness=augmentation_cof, contrast=augmentation_cof, hue=0)
        ])
        trans_train_label = None
        trans_train_com = transforms.Compose([
                my_transform1,
                ])
        trans_val = transforms.Compose(transforms = [
                transforms.ColorJitter(brightness=augmentation_cof, contrast=augmentation_cof, hue=0),
        ])
        trans_val_label = None
        trans_val_com = transforms.Compose([
                my_transform1,
                ])
        if "noise" in augmentation:
            noise_transform=noise_augmentation
        else:
            noise_transform=None
    else:
        trans_train=None
        trans_train_label=None
        trans_train_com=None
        trans_val=None
        trans_val_label=None
        trans_val_com=None
        noise_transform=None

    
    #load dataset with data augmentation
    train_dataset = CellDataset(image_dir=train_img_dir,label_dir=train_label_dir,
                                threshold=th,
                                resize_crop=resize_crop,
                                transform=trans_train, 
                                label_transform=None, 
                                com_transform=trans_train_com,
                                noise_transform=noise_transform
                                )
    val_dataset = CellDataset(image_dir=test_img_dir,label_dir=test_label_dir,
                              threshold=th,
                              resize_crop=resize_crop,
                              transform=trans_val, 
                              label_transform=None,
                              com_transform=trans_val_com,
                              noise_transform=noise_transform
                              )
       
 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size) 
    val_dataloader = DataLoader(val_dataset, batch_size=1) 
    
    
    best_loss = 1000 #save model when loss lower than best_loss
    #num_epochs=80
    for epoch in range(num_epochs):
        i=0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #train
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        for x, y in train_dataloader:# batch_size=4
            inputs = x.to(device)
            labels = y.to(device)
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            outputs = model(inputs)#forward
            loss = criterion(outputs, labels)#compute loss
            loss.backward()#compute the gradient
            optimizer.step()#update parameter
            optimizer.zero_grad()#set gradient to zero
            epoch_loss += loss.item()
            
            i=i+1
            
        #validation
        model.eval()
        val_losses = 0
        for x, label in val_dataloader:
            x = x.to(device)
            label = label.to(device)
            x = x.to(torch.float32)
            label = label.to(torch.float32)
            y = model(x)
            val_loss = criterion(y, label)
            val_losses += val_loss.item()
        print("epoch %d train_loss:%0.4f  val_loss:%0.4f" % (epoch, epoch_loss, val_losses))
        pandas_loss=pd.DataFrame([epoch_loss, val_losses]).T
        pandas_loss.to_csv(loss_outpath, mode='a',encoding='utf-8',header=False,index=False)                   
        
        scheduler.step(val_losses) # In min mode, lr will be reduced when the quantity monitored(val_loss) has stopped decreasing;
        if val_losses < best_loss:
            print("******** New optimal found, saving state ********")
            best_loss = val_losses
            torch.save(model.state_dict(),model_name % num_epochs)# save model
            

def predict():
    model = unet.UNet(n_input_channel,n_input_channel)
    model.load_state_dict(torch.load(args.weight,map_location='cpu'))
    images_name=np.sort(os.listdir(predict_img_dir))
    cell_dataset = PredictCellDataset(image_dir=predict_img_dir,
                               threshold=th,
                               resize_crop=resize_crop,
                               transform=None,
                               )
    dataloaders = DataLoader(cell_dataset)#batch_size,default 1
    model.eval()
    print('starting')
    i=0
    
    with torch.no_grad():
        for x in dataloaders:
            image_name=images_name[i]
            img_input=torch.squeeze(x).numpy()[0,:,:]
           
            x = x.to(torch.float32)
            y=model(x)
            
            i=i+1
            print('predict successfully',i,image_name)
            
            img_y = torch.squeeze(y).numpy()
            prob=((2**16-1)*img_y).transpose(1,2,0)
            prob=prob.astype(np.uint16)
            tiff.imsave(os.path.join(predict_outpath,image_name.replace(".tiff",'_pred_Probabilities.tiff')), prob)
            

if __name__ == '__main__':
                        
    #parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, help='train or predict')
    parser.add_argument('--weight', type=str, help='the path of the model')
    parser.add_argument('--workdir', type=str, default="/workspace/data/", help='the path of the data folder')
    parser.add_argument('--preddir', type=str, default="/workspace/data/predict_BRCA2/", help='the new img dataset folder')
    parser.add_argument('--threshold', type=float, default=99.7, help='ceiling pixel value to the percentile of total pixel intensity')
    parser.add_argument('--noise', type=bool, default=True, help='add random noise to do data augmentation')
    parser.add_argument('--aug_cof', type=float, default=0.5)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--epoch',type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    ###0. project path
    #project_path="/mnt/public/qy/105backups/qy/qy/project/imc/METABRIC_IMC/train/"
    project_path=args.workdir
    test_dir="test"#validation dataset
    predict_dir=args.preddir#test dataset
    

    if args.action == 'train':
        ###1. image
        train_img_dir =os.path.join(project_path,'train/combined2_image/')
        test_img_dir = os.path.join(project_path,test_dir,"combined2_image")
        ###2. label
        train_label_dir = os.path.join(project_path,"train/probability")
        test_label_dir = os.path.join(project_path,test_dir,"probability")
        resize_crop=True
        
        th=args.threshold#default 99.7
        threshold="_threshold-"+str(th) if th else ""           
            
        augmentation_cof=args.aug_cof#default 0.5
        num_epochs = args.epoch
        learning_rate = args.lr
        batch_size = args.bs
        
        #data augmentation
        if not args.noise:
            augmentation="_withAug-"+str(augmentation_cof) if augmentation_cof else ""
        #data augmentation with noise
        else:
            augmentation="_withAugnoise-"+str(augmentation_cof) if augmentation_cof else "_withAugnoise"

        #print("training mode is: ",threshold+augmentation)
        print("training parameters: percentile ({}), augmentation ({})".format(th, augmentation[1:]))
        ###train outpath
        loss_outpath=os.path.join(project_path,"train","loss",time+threshold+augmentation+"_loss.csv")
        model_name=os.path.join(project_path,'model',time+threshold+augmentation+'_model_%d.pth')
        train()
    elif args.action == 'predict':
        ###1. image
        predict_img_dir=os.path.join(predict_dir,"crop512_combined2_image")
        resize_crop=True
        process_info=args.weight.split("/")[-1][12:][:-13].split('_')
        print('trained model parameters: {}'.format(process_info))

        th=args.threshold
        threshold="_threshold-"+str(th) if th else ""
        augmentation=""
        #training model with augmentation
        if "withAug" in process_info[-1]:
            out_subdir=threshold+"_"+process_info[-1]
        #training model without augmentation
        else:
            out_subdir=threshold
        if out_subdir[0] is "_":
            out_subdir=out_subdir[1:]  
        predict_outpath =os.path.join(predict_dir,"predictionProbability",out_subdir)
        if not os.path.exists(predict_outpath):
            os.mkdir(predict_outpath)
        print("predicted probability map folder:",predict_outpath)
        predict()