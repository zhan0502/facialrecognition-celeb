#Library import
import numpy as np
from math import cos, pi 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor 
from torch.utils import data
import os
from torch.autograd import Variable
print("This is to infer result for submission")
image_size = 178
device = torch.device('cuda')
#create the flatten function only for pytorch 1.0.1 as this is a default function 
# number of channels in the training images. For color images this is 3
nc = 3
 
# size of feature maps  
n = 64
 
bs = 64

workers = 2

# number of training epochs (to save time, we only train for 3 epochs in this tutorial)

# num_epochs = 5

# learning rate for optimizers
lr = 0.002

# beta1 hyperparam for Adam optimizers
beta1 = 0.5
# number of workers for dataloader
workers = 2

 
# size of feature maps in discriminator
ndf = 128

# number of training epochs (to save time, we only train for 3 epochs in this tutorial)
 
# num_epochs = 5
#for label smoothing 
eps = 0.005
 #with open("test.txt", "r", encoding = "utf8") as file:
    #lines = file.readlines()[1:15289] 

test_dir = '/home/projects/52000146/ACV_P1/testset/'
img_dir = '/home/projects/52000146/ACV_P1/testset/testset/'
images = os.listdir(img_dir)
img_name = np.array(images)
label_1 = 1 - eps
label_0 = eps/(2-1)
 
######################test load resnet
from torchvision.models.resnet import resnet50
model =  resnet50(pretrained=False)
num_features =  model.fc.in_features
model.fc = nn.Linear(num_features , 40)

mymodel = torch.load('resnet178.pth')
########################################################################
#test data
def preprocess_test(datadir):
    test_transforms = transforms.Compose([
        transforms.Resize((image_size)),
        transforms.RandomCrop((image_size)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
#        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) 

    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
 
    testloader = torch.utils.data.DataLoader(test_data,
                    batch_size=bs,shuffle=False, num_workers=workers)
    return testloader
testloader= preprocess_test(test_dir)
 

output_matrix = torch.zeros([img_name.shape[0], 40]) 
for test_i, image in enumerate(testloader):  
        image = image[0].to(device) 
        output_test = mymodel(image)
         
        #print(test_i*bs,(1+test_i)*bs)
        for idx_val in range(output_test.shape[0]):
            for j_val in range(output_test.shape[1]):
                if output_test[idx_val][j_val] >= 0.5:
                    p = label_1
                else:
                    p = label_0
                output_matrix[idx_val+(test_i*bs)][j_val] = p
 
 
output_matrix  = np.where(output_matrix == 0.995, 1, -1) 
output_image = img_name
with open("predictions.txt", "w", encoding = "utf8") as file:
    for i in range(len(output_image)):
        l = ''
        for k, j in enumerate(output_matrix[i]):
            if k == 39:
                l =  l + str(int(j.item())) 
            else:
                l =  l + str(int(j.item())) + ' '
        if i == len(output_image) -1:
            L = output_image[i] + ' '+ l 
        else:
            L = output_image[i]+ ' ' + l + "\n" 
        #print(L)
        
        # \n is placed to indicate EOL (End of Line) 
        file.writelines(L) 
    