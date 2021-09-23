# ![image](https://raw.githubusercontent.com/xmuyulab/Dice-XMBD/main/figure/Dice-XMBD.jpg)
[comment]:<img src="https://raw.githubusercontent.com/xmuyulab/Dice-XMBD/main/figure/Dice-XMBD.jpg" align=center />
Dice-XMBD<sup>1</sup> is marker agnostic and can perform cell segmentation for multiplexed images of different channel configurations without modification. This project contains code, a generic model, and example test datasets for single cell segmentaion of multiplexed imaging data. 

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

The image data in total might be large, we suggest to put image data in another folder having enough capacity (thereafter we refer as "your_data_dir").
Create a container with GPU:
```
docker run -it --gpus all --name use-dice-xmbd \
-v /your_data_dir:/mnt/data \
-v /path_to_Dice-XMBD/:/workspace/ \
xiaoxu9907/dice-xmbd:latest /bin/bash
```

Create a container with CPU:
```
docker run -it --name use-dice-xmbd \
-v /your_data_dir:/mnt/data \
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

## 2. Use a trained model
(1) Run an example with a trained model which can be downloaded [here](https://ndownloader.figshare.com/files/28301040) (put the model in */your_data_dir/model* or use the following command):
```
# download the trained model

DATAPATH="/home/xiaoxu/IMC/Dice-XMBD/data/" # change to /your_data_dir/
mkdir -p "$DATAPATH"/model
wget https://ndownloader.figshare.com/files/28301040 -O "$DATAPATH"model/02-15-20-14_threshold-99.7_withAugnoise-0.5_model_80.pth

docker exec -it use-dice-xmbd bash

# predict pixel probability map
python /workspace/src/main_probability.py --action predict --weight="/mnt/data/model/02-15-20-14_threshold-99.7_withAugnoise-0.5_model_80.pth" --preddir '/workspace/data/predict_BRCA2/'
```

(2) Get single cell mask from CellProfiler:
Download [CellProfiler 3.1.9](https://cellprofiler.org/previous-releases) and use */path_to_Dice-XMBD/process/2_generate_cellmask.cpproj* pipeline (from [Bernd Bodenmiller lab<sup>2</sup>](https://github.com/BodenmillerGroup/ImcSegmentationPipeline/tree/main/cp3_pipelines)) to post-process pixel probability maps from step(1). Some modules in this pipeline are required in the folder */path_to_Dice-XMBD/process/ImcPluginsCP/plugins*, which can be downloaded in [Bernd Bodenmiller lab<sup>2</sup>](https://github.com/BodenmillerGroup/ImcPluginsCP) as well. 

For CellProfiler GUI: modify the CellProfiler Preferences to the plugin folder (path_to/ImcPluginsCP/plugins) and then restart cellprofiler.

For command line: --plugins-directory=path_to/ImcPluginsCP/plugins


(3) Extract mean protein intensity of single cells:
```
python /workspace/process/3_imc_extract_pro.py 
```


## 3. Quick start to train a model
(0) resize image to 512*512 and combine multiple channels to 2 channel-input image (the given panel file need to have the same ordered with img channels: at least contains two columns("nuclear","membrane_cytoplasm"))
```
# example: 
python /workspace/process/1_preprocess_img.py --process pre
python /workspace/process/1_preprocess_img.py --process pre --workdir /mnt/data/yourimgs --panel /mnt/data/yourimgs_panel_info
```

(1) Training a model use your own training dataset which should contain *train* and *test* datasets, training datasets example can be download here: [train](https://ndownloader.figshare.com/files/28298646) and [test](https://ndownloader.figshare.com/files/28297719) (put datasets in */your_data_dir/data/* folder or use the following command).
```
# download training datasets and put in */path_to_Dice-XMBD/data/* folder
DATAPATH="/home/xiaoxu/IMC/Dice-XMBD/data/" # change to /your_data_dir/

mkdir -p "$DATAPATH"
mkdir -p "$DATAPATH"/BRCA1

#download training and validation dataset
wget https://ndownloader.figshare.com/files/28298646 -O "$DATAPATH"BRCA1/train.zip
wget https://ndownloader.figshare.com/files/28297719 -O "$DATAPATH"BRCA1/test.zip

# unzip files:
unzip "$DATAPATH"BRCA1/test.zip -d "$DATAPATH"BRCA1/
unzip "$DATAPATH"BRCA1/train.zip -d "$DATAPATH"BRCA1/

# enter the container
docker exec -it use-dice-xmbd bash

python /workspace/src/main_probability.py --action train --workdir /mnt/data/BRCA1
```

(2) Predict pixel probability map from a trained model:
```
# example: 
python /workspace/src/main_probability.py --action predict --weight='/mnt/data/model/02-15-20-14_threshold-99.7_withAugnoise-0.5_model_80.pth' --preddir '/workspace/data/predict_BRCA2/'
python /workspace/src/main_probability.py --action predict --weight='/mnt/data/your_dataset/model/model_name.pth' --preddir '/mnt/data/your_test_dataset' --testname 'combined2_image'

# merge resized images to the original size of image:
python /workspace/process/1_preprocess_img.py --process post
```

(3) Get single cell mask from CellProfiler:
Download [CellProfiler](https://cellprofiler.org/previous-releases) and use */path_to_Dice-XMBD/process/2_generate_cellmask.cpproj* pipeline to post-process pixel probability maps from step(2)

(4) Extract mean protein intensity of single cells:
```
python /workspace/process/3_imc_extract_pro.py 
```


## 4. Dice-XMBD training dataset and other test datasets
Trainging datasets can be downloaded here: [BRCA1](https://ndownloader.figshare.com/files/28298646), the corresponding original images can be found [here](https://idr.openmicroscopy.org/about/download.html) with accession code idr0076. 
Other test datasets used in our paper can be downloaded from: [BRCA2](https://zenodo.org/record/3518284#.YLnmlS8RquU), [T1D1](https://data.mendeley.com/datasets/cydmwsfztj/1), [T1D2-part1](https://data.mendeley.com/datasets/9b262xmtm9/1), [T1D2-part2](https://data.mendeley.com/datasets/xbxnfg2zfs/1)


## 5. References
[1] Xiao X, Qiao Y, Jiao Y, Fu N, Yang W, Wang L, Yu R and Han J (2021) Dice-XMBD: Deep Learning-Based Cell Segmentation for Imaging Mass Cytometry. Front. Genet. 12:721229. [https://doi.org/10.3389/fgene.2021.721229](https://doi.org/10.3389/fgene.2021.721229)
[2] BodenmillerGroup. 2020. ImcPluginsCP. [https://github.com/BodenmillerGroup/ImcPluginsCP](https://github.com/BodenmillerGroup/ImcPluginsCP); ImcSegmentationPipeline. [https://github.com/BodenmillerGroup/ImcSegmentationPipeline/tree/main/cp3_pipelines](https://github.com/BodenmillerGroup/ImcSegmentationPipeline/tree/main/cp3_pipelines)


