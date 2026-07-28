# FSMODNet: A Closer Look at Few-Shot Detection in Multispectral Data

## 📄 Abstract

Few-shot multispectral object detection (FSMOD) addresses the challenge of detecting objects across visible and thermal modalities with minimal annotated data. In this paper, we explore this complex task and introduce a framework named "FSMODNet" that leverages cross‑modality feature integration to improve detection performance even with limited labels.
By effectively combining the unique strengths of visible and thermal imagery using deformable attention, the proposed method demonstrates robust adaptability in complex illumination and environmental conditions. Experimental results on two public datasets show promising object detection performance in challenging low-data regimes.


## 📷 Overview
<p align="center">
        <img src="figs/fsmodnet.png" alt="FSMODNet Framework" width="90%">
</p>

## 🧩 Installation
1. Clone this repo
```bash
git clone https://github.com/Manuelnkegoum-8/FSMODNet.git
cd FSMODNet
```
2. Install required packages
```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install natten==0.20.1+torch270cu128 -f https://whl.natten.org
pip install -r requirements.txt
``` 
3. Compiling CUDA operators
As the framework is developed upon Deformable DETR, you need to compile Deformable Attention.
```bash
cd models/dino/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

## 📁 Datasets 
Download and place FLIR and M3FD datasets in data/ with the expected folder structure.

### FLIR
```bash
data/FLIR/
├── Coco_Annotations/         # COCO-style JSON annotations
├── JPEGImages/               # RGB–thermal image pairs
├── few_shot/                 # Few-shot split files (per class/split/shot)
``` 
### M3FD
You need to put each dataset imgs in the directory according to the train.txt and val.txt.
```bash
data/M3FD/
├── Coco_Annotations/     
├── train.txt
├── test.txt
├── visible/
│   ├── train/
│   └── test/
├── infrared/
│   ├── train/
│   └── test/
├── few_shot/                 # Few-shot split files
``` 

## 🏁 Training
### Base training
We take FLIR as example. First, create meta_learning.sh and copy the following commands to it.
```bash
dataset=flir
split=1
backbone=resnet50
exps_dir=exps
output_dir=$exps_dir/${dataset}/split${split}/${backbone}
python training.py --config configs/fsmodnet/meta_learning/${dataset}_${split}.py --output_dir $output_dir
```
Then run the command below
```bash
./meta_learning.sh
```

### Few-shot finetuning
We take FLIR as example. First, create fs.sh and copy the following commands to it.
```bash
dataset=flir
split=1
spectrum=both
backbone=resnet50
exps_dir=exps
ckpt=$exps_dir/${dataset}/split${split}/${backbone}/detector.ckpt

for shot in 5 10
do
for seed in 1 2 3 4 5 6 7 8 9 10
do
output_dir=$exps_dir/${dataset}/split${split}/${backbone}/${shot}shot/seed${seed}
python training.py  --config configs/fsmodnet/meta_learning/few_shot/${dataset}_${split}.py  --output_dir $output_dir --checkpoint $ckpt \
    --cfg-options train_data.ann_file=few_shot/seed${seed}/${shot}.json  support_data.ann_file=few_shot/seed${seed}/${shot}.json
done
done

```
Then run the command below
```bash
./fs.sh
```
## 📌 Citation

## 👍 Acknowledgement
FSMODNet is built on previous works such as [Meta-DETR](https://github.com/ZhangGongjie/Meta-DETR),[DINO-DETR](https://github.com/IDEA-Research/DINO) and [NATTEN](https://github.com/SHI-Labs/NATTEN). We sincerely thank the authors of these foundational works for making their code and research publicly available, which greatly facilitated the development of FSMODNet.
