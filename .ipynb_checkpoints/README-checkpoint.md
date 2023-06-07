# DLMIA-UNet
### Description
environment.yml contains all the required packages to run model. Can be directly installed using conda

restructure_data.py: Run this file to restructure dataset with train-test split. Change the paths within the script accordingly

dataset.py: Contains dataset class that directly reads images from directory and does preprocessing we had done earlier. This uses transformations from transformations.py

model.py: Contains 3DUNet model

Run main.py to start training. 

Testing script yet to be implemented.



