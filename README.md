# PLN

### Introduction

This is the Implementation of《PLN: Parasitic-Like Network for Barely Supervised Medical Image Segmentation》

### Usage

1. Clone the repository:
```
git clone https://github.com/ShumengLI/PLN.git 
cd PLN
```
2. Put the data in `data/LA/2018LA_Seg_Training Set` and `data/LA/processed_h5`.

3. Train the model
```
python train_pln.py --exp model_name
```
Params are the best setting in our experiment.

4. Test the model
```
python test.py --model model_name
```
Our best model is uploaded.

### Acknowledgement

Part of the code is origin from [UA-MT](https://github.com/yulequan/UA-MT) and [VoxelMorph-torch](https://github.com/zuzhiang/VoxelMorph-torch). 
