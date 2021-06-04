# ![image](https://raw.githubusercontent.com/xmuyulab/Dice-XMBD/main/figure/Dice-XMBD.jpg)
<img src="https://raw.githubusercontent.com/xmuyulab/Dice-XMBD/main/figure/Dice-XMBD.jpg" align=center />
Dice-XMBD is marker agnostic and can perform cell segmentation for multiplexed images of different channel configurations without modification. This project contains code, a generic model, and example test datasets for single cell segmentaion of multiplexed imaging data. 

## 1. Install with docker image
Open terminal and go to the folder you want to store the project. Then type the following command:
```
git clone https://github.com/xmuyulab/Dice-XMBD.git
```

Next, the required environment and dependencies can be builded from a dorker image.
Pull dorker image:
```
docker pull xiaoxu9907/dice-xmbd:latest
```


Create a container with GPU:
```
docker run -it --gpus all --name use-dice-xmbd \
-v /yourdatadir:/mnt/data \
-v /path_to_Dice-XMBD/:/workspace/ \
xiaoxu9907/dice-xmbd:latest /bin/bash
```


Create a container with CPU:
```
docker run -it --name use-dice-xmbd \
-v /yourdatadir:/mnt/data \
-v /path_to_Dice-XMBD/:/workspace/ \
xiaoxu9907/dice-xmbd:latest /bin/bash
```

Some docker commands:
```
docker exec -it use-dice-xmbd bash # enter the container
exit # or use ctrl+d to quit the container
docker start use-dice-xmbd
docker stop use-dice-xmbd
```

You may need to open permission for the folder if you want to modity the code in the docker container:
```
chmod +777 -R /path_to_Dice-XMBD/
```

Run an quick example with a trained model which can be downloaded [here](https://figshare.com/account/projects/115347/articles/14731563) (put the model in */path_to_Dice-XMBD/data/model* or use the following command):
```
# download the trained model
wget https://ndownloader.figshare.com/files/28301040 -O /path_to_Dice-XMBD/data/model/02-15-20-14_threshold-99.7_withAugnoise-0.5_model_80.pth

docker exec -it use-dice-xmbd bash

# predict pixel probability map
sh src/test.sh
```

## 2. Quick start
(0) resize image to 512*512 and combine multiple channels to 2 channel-input image (the given panel file need to have the same ordered with img channels: at least contains two columns("nuclear","membrane_cytoplasm"))
```
# example: 
python /workspace/process/1_preprocess_img.py --process pre
python /workspace/process/1_preprocess_img.py --process pre --workdir /mnt/data/yourimgs --panel /mnt/data/yourimgs_panel_info
```

(1) Training a model use your own training dataset which should contain *train* and *test* datasets, training datasets example can be download here: [train](https://figshare.com/account/projects/115347/articles/14730573) and [test](https://figshare.com/account/projects/115347/articles/14730480) (put datasets in */path_to_Dice-XMBD/data/* folder or use the following command).
```
# download training datasets and put in */path_to_Dice-XMBD/data/* folder
wget https://ndownloader.figshare.com/files/28297719 -O /path_to_Dice-XMBD/data/test.zip
wget https://ndownloader.figshare.com/files/28298646 -O /path_to_Dice-XMBD/data/train.zip

unzip /path_to_Dice-XMBD/data/test.zip
unzip /path_to_Dice-XMBD/data/train.zip

# enter the container
docker exec -it use-dice-xmbd bash

python /workspace/src/main_probability.py --action train --workdir /mnt/traindata/
 
```

(2) Get pixel probability map from a trained model:
```
# example: 
python /workspace/src/main_probability.py --action predict --weight='/workspace/data/model/02-15-20-14_threshold-99.7_withAugnoise-0.5_model_80.pth' --preddir '/workspace/data/predict_BRCA2/'
python /workspace/src/main_probability.py --action predict --weight='/workspace/data/model/02-15-20-14_threshold-99.7_withAugnoise-0.5_model_80.pth'

# merge resized images to the original size of image:
python /workspace/process/1_preprocess_img.py --process post
```

(3) Get single cell mask from CellProfiler:
Download [CellProfiler](https://cellprofiler.org/previous-releases) and use */process/2_generate_cellmask.cpproj* pipeline to post-process pixel probability maps from step(2)

(4) Extract mean protein intensity of single cell:
```
python /workspace/process/3_imc_extract_pro.py 
```


## 3. Dice-XMBD training dataset and other test datasets
Trainging datasets can be downloaded here: [BRCA1](https://figshare.com/account/home#/projects/115347), its original images can be found [here](https://idr.openmicroscopy.org/search/?query=Name:idr0076-ali-metabric/experimentA)
Other test datasets used in our paper can be downloaded from: [BRCA2](https://zenodo.org/record/3518284#.YLnmlS8RquU), [T1D1](https://data.mendeley.com/datasets/cydmwsfztj/1), [T1D2-part1](https://data.mendeley.com/datasets/9b262xmtm9/1), [T1D2-part2](https://data.mendeley.com/datasets/xbxnfg2zfs/1)
