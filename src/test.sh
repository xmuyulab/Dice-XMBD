#!/bin/bash

DATAPATH="/home/xiaoxu/IMC/Dice-XMBD/data/" # change to /your_data_dir/

#### preprocess image data
# python /workspace/process/1_preprocess_img.py --process pre 

#### train a model
python /workspace/src/main_probability.py --action train --epoch 30 --workdir "$DATAPATH"BRCA1 --bs 2

# #### predict pixel probability maps by a trained model
# python /workspace/src/main_probability.py --action predict --weight='/workspace/data/model/02-15-20-14_threshold-99.7_withAugnoise-0.5_model_80.pth' --preddir '/workspace/data/predict_BRCA2/'

#### merge figure back
# python /workspace/process/1_preprocess_img.py --process post


#### generate cell masks by using Cellprofiler pipeline ('/workspace/process/2_generate_cellmask.cpproj')

#### extract single cell protein
# python /workspace/process/3_imc_extract_pro.py 