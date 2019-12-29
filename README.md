## Introduction ##
This repository contains an example to represent how to implement segmentation using Unet on the simple dataset (Carvana)
download [here](https://www.kaggle.com/c/carvana-image-masking-challenge)
This is paper which explained about [Unet](https://arxiv.org/pdf/1505.04597.pdf). Please refer this paper for a deeper understanding.

## Dependencies ##
Python3, tensorflow2, numpy, opencv 

## Usage ##

### Data structure ###
Data structure should be structured as follows:
- data
	- images
		- img
	- masks
		- img
	
### Training ###
Please follow the steps as follows:
1. Creates weight folder with name "weight". This folder to save model file(.h5) after training process finish.
2. Run train.py script. You also can change (epochs, validation split, image size...) to be suitable in your case

### Test ###
Checks for the existence of h5 file in weight folder. After that, run test.py. You can change the image path at line 33 that you want

## Result ## 
This model archives 99% accuracy on validation set.
![alt text](https://github.com/vanloc19bk96/carvana_segmentation/blob/master/result/accuracy.PNG)	
![alt text](https://github.com/vanloc19bk96/carvana_segmentation/blob/master/result/loss.PNG)
![alt text](https://github.com/vanloc19bk96/carvana_segmentation/blob/master/result/result.PNG)
