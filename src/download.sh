#!/bin/bash

DATAPATH="/home/xiaoxu/IMC/Dice-XMBD/data/" # change to /your_data_dir/
DICEPATH="/home/xiaoxu/Dice-XMBD/" # change to /path_to_Dice-XMBD/

mkdir -p "$DATAPATH"
mkdir -p "$DATAPATH"/BRCA1
mkdir -p "$DATAPATH"/model

# download model
wget https://ndownloader.sfigshare.com/files/28301040 -O "$DATAPATH"model/02-15-20-14_threshold-99.7_withAugnoise-0.5_model_80.pth

#download training and test dataset
wget https://ndownloader.figshare.com/files/28298646 -O "$DATAPATH"BRCA1/train.zip
wget https://ndownloader.figshare.com/files/28297719 -O "$DATAPATH"BRCA1/test.zip

# unzip files:
unzip "$DATAPATH"BRCA1/test.zip -d "$DATAPATH"BRCA1/
unzip "$DATAPATH"BRCA1/train.zip -d "$DATAPATH"BRCA1/

# Create a container with GPU and enter docker:
docker run -it --gpus all --name use-dice-xmbd -v "$DATAPATH"/:/mnt/data -v "$DICEPATH"/:/workspace/ xiaoxu9907/dice-xmbd:latest /bin/bash

# Create a container with CPU and enter docker:
# docker run -it --name use-dice-xmbd -v "$DATAPATH"/:/mnt/data -v "$DICEPATH"/:/workspace/ xiaoxu9907/dice-xmbd:latest /bin/bash