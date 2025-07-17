# LivDet-Iris 2025 Baselines 
This repository provides instruction on how to train and test the baseline algorithms for the LivDet-Iris 2025 competition. The selected models include ResNet101, DenseNet121, and ViT-B/16.

### Conda Environment ###
To set up the environment:
```
conda env create -f  environment.yml
conda activate PADbaselines
```

### Training ###
To train a basline model on your own dataset, use:
```
python train.py -csvPath <path_to_train_csv_file> -method <DenseNet | resnet | vit> -outputPath <path_to_result_dir>
```
Example of CSV file format (each row indicates the data split (train or test), class label (Live or Attack), and the corresponding image filename.)
```
train,Live,imageFile1.png
train,Attack,imageFile2.png
test,Live,imageFile3.png
test,attack,imageFile4.png
```
Note that all training images are cropped around the iris region with a 16-pixel padding.

### Validation ###

To evaluate a trained model, use: 
```
python submission_baseliene_xxx.py test_set.csv test_results_xxx.csv
```
Replace `xxx` with the model name: `densenet`, `resnet`, or `vit`.

### Weights ### 
Baselines trained only on the authentic samples: [weights](https://notredame.box.com/s/2l93vl2uawyba9y4u1ph1fe18qgdwzap)

Baselines trained on both authentic and synthetic samples: [weights](https://notredame.box.com/s/4fynvx52klc662472i4jze0aisb0enc1)

Mask circle: [weights](https://notredame.box.com/s/fajf9tzgzbbv2m7potm9v7xyldaqedb7)


