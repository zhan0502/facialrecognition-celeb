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
import time
from torch.autograd import Variable

#image_size = 128
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
    
with open("list_attr_celeba.txt", "r", encoding = "utf8") as file:
    lines = file.readlines()[1:] 

img_names = []
attr_values = []
for i, line in enumerate(lines):
    if i == 0: 
        attribute_name = line
        attribute_name = attribute_name.rstrip()
        attribute_list = attribute_name.split()
    else: 
        line = line.rstrip()
        line_list = line.split()
        img_name = line_list[0]
        attr_value = line_list[1:]
        img_names.append(img_name)
        attr_values.append(attr_value) 
attr_matrix = np.array(attr_values)
img_name = np.array(img_names)
print(img_name.shape, attr_matrix.shape)
print(attribute_list)



train_dir = '/home/projects/52000146/ACV_P1/data/'
val_dir ='/home/projects/52000146/ACV_P1/val/'
test_dir = '/home/projects/52000146/ACV_P1/test/'
#change image size to 128  

def preprocess(datadir):
    test_transforms = transforms.Compose([
        #transforms.Resize((image_size)),
        transforms.RandomCrop((178)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) 

    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
 
    testloader = torch.utils.data.DataLoader(test_data,
                    batch_size=bs,shuffle=False, num_workers=workers)
    return testloader
 
trainloader = preprocess(train_dir)
validloader = preprocess(val_dir)

attr_matrix_int = attr_matrix.astype(int)
attr_matrix_norm = np.where(attr_matrix_int<0, 0, 1) 

label_1 = 1 - eps
label_0 = eps/(2-1)
attr_matrix_norm = np.where(attr_matrix_norm<0.5, label_0, label_1) 
train_label = attr_matrix_norm[0:162770]
valid_label = attr_matrix_norm[162770 :182637]
print(  train_label.shape[0], valid_label.shape[0])
#train_target_tensor =  torch.Tensor(train_label[:,0]) 
#valid_target_tensor =  torch.Tensor(valid_label[:,0]) 
train_target_tensor =  torch.Tensor(train_label) 
valid_target_tensor =  torch.Tensor(valid_label) 

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
#model
#https://pytorch.org/hub/pytorch_vision_resnet/
from torchvision.models.resnet import resnet50
model =  resnet50(pretrained=False)
num_features =  model.fc.in_features
model.fc = nn.Linear(num_features , 40)

# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
model.eval()
#print(model)
model.to(device) 

class WeightedLoss(nn.Module):
    def __init__(self, a = 0.01, b = 2, c =1, d = 20):
        super().__init__()
        self.easy_feature = a
        self.difficult_feature = b
        self.normal  = c
        self.super_difficult = d
        #self.inbalance_feature = c
        
    def forward(self, pred_logits, target):
        pred= pred_logits.sigmoid()
        ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
        loss = torch.zeros([pred.shape[0], pred.shape[1]])

        for i in range(pred.shape[1]):
                if i == 4 or 10 or 14 or 15 or 17 or 26 or 35:
                    loss[:,i]= self.easy_feature * ce[:,i]
                
                elif i == 1 or 2 or 3 or 7 or 11 or 19 or 21 or 25 or 27 or 32 or 33:
                    loss[:,i]= self.super_difficult* ce[:,i]
                    
                elif i == 31 or 34 or 39:
                    loss[:,i]= self.difficult_feature * ce[:,i]
                else: 
                    loss[:,i]= self.normal * ce[:,i]
    
        return loss  

class BCELoss(nn.Module):
    def __init__(self, weight= None, size_average=None, reduce=None, reduction: str = 'mean'):  
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target): 
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma=5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred_logits, target):
        pred= pred_logits.sigmoid()
        
        ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
        
        alpha = target * self.alpha + (1. - target)*(1. - self.alpha)
        pt = torch.where(target ==1, pred, 1-pred)
    
        return alpha * (1. - pt) ** self.gamma*ce

class Weighted_focal_Loss(nn.Module):
    def __init__(self, a = 0.01, b = 2, c =1, d = 20,  alpha = 0.25, gamma=10):
        super().__init__()
        self.easy_feature = a
        self.difficult_feature = b
        self.normal  = c
        self.super_difficult = d
        self.alpha = alpha
        self.gamma = gamma
        #self.inbalance_feature = c
        
    def forward(self, pred_logits, target):
        pred= pred_logits.sigmoid()
        ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
        loss = torch.zeros([pred.shape[0], pred.shape[1]])

        for i in range(pred.shape[1]):
                if i == 4 or 10 or 13 or 14 or 15 or 16 or 17 or 22 or 26 or 28 or 29 or 30 or 35 or 38:
                    loss[:,i]= self.easy_feature * ce[:,i]
                
                elif i == 2 or 18 or 19 or 20 or 21 or 31 or 36:
                    loss[:,i]= self.super_difficult* ce[:,i]
                    
                elif i == 1  or 6 or 7 or 8  or 25 or 27 or 33 or 39:
                    loss[:,i]= self.difficult_feature * ce[:,i]
                else: 
                    loss[:,i]= self.normal * ce[:,i]
                    
        alpha = target * self.alpha + (1. - target)*(1. - self.alpha)
        pt = torch.where(target ==1, pred, 1-pred)
        loss = loss.to(device)
        
        return alpha * (1. - pt) ** self.gamma*loss
class NormalLoss(nn.Module):
    def __init__(self, a = 1, b = 1, c =1, d = 1):
        super().__init__()
        self.easy_feature = a
        self.difficult_feature = b
        self.normal  = c
        self.super_difficult = d
        #self.inbalance_feature = c
        
    def forward(self, pred_logits, target):
        pred= pred_logits.sigmoid()
        ce =  F.binary_cross_entropy_with_logits(pred_logits, target, reduction ='none')
        loss = torch.zeros([pred.shape[0], pred.shape[1]])

        for i in range(pred.shape[1]):
                if i == 4 or 10 or 13 or 14 or 15 or 16 or 17 or 22 or 26 or 28 or 29 or 30 or 35 or 38:
                    loss[:,i]= self.easy_feature * ce[:,i]
                
                elif i == 2 or 18 or 19 or 20 or 21 or 31 or 36:
                    loss[:,i]= self.super_difficult* ce[:,i]
                    
                elif i == 1  or 6 or 7 or 8  or 25 or 27 or 33 or 39:
                    loss[:,i]= self.difficult_feature * ce[:,i]
                else: 
                    loss[:,i]= self.normal * ce[:,i]
    
        return loss      
#
#criterion = NormalLoss()
#criterion = Weighted_focal_Loss() 
#optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))    


#
model.train()

# overall accuracy
def train_val(validloader):
    #change here to see if the overrun of cuda can be solved
    val_correct = np.zeros([40]) 
    for i, image in enumerate(validloader):  
                image = image[0].to(device) 
                output_val = model(image)
                target_val = valid_target_tensor[i*bs:(1+i)*bs].to(device) 
                #print(i*bs,(1+i)*bs,output_val.shape[0])

                for idx_val in range(output_val.shape[0]):
                    for j_val in range(output_val.shape[1]):
                        if output_val[idx_val][j_val] >= 0.5:
                            p = label_1
                        else:
                            p = label_0
                        if target_val[idx_val][j_val] == p:
                            val_correct[j_val] =val_correct[[j_val]] + 1


    val_accuracy = val_correct/(len(valid_target_tensor)*1)
    for i in range(val_correct.shape[0]):
        val_result = attribute_list[i] + ": " + str(round(val_accuracy[i],5))
        print(val_result)
    print("average: " + str(sum(val_accuracy)/40))
    return val_accuracy
num_epochs = 20
for epoch in range(num_epochs):
    #running_loss =0
    #start = time.time() 
    # learning rate strategy : 
    
    if epoch < 5:
        lrs = np.linspace(0, lr, 6)[1:]
        learning_rate = lrs[epoch]
    else:
        learning_rate= (1+ cos(epoch*pi/num_epochs))*1/2*lr
         
        
 
    # create a new optimizer at the beginning of each epoch: give the current learning rate.   
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, 0.999))    
    
    

    for i, image in enumerate(trainloader):

   
        correct = 0 
        start = time.time()
        model.zero_grad() 
        image = image[0].to(device) 
        output = model(image)
        target = train_target_tensor[i*bs:(1+i)*bs] .to(device)  
        loss = F.binary_cross_entropy_with_logits(output, target)
         
        # Backward
        #optimizer.zero_grad()  # Set gradients to zero
        #loss.backward()        # From the loss we compute the new gradients
        loss.sum().backward()
        optimizer.step()       # Update the parameters/weight
        #running_loss += loss.detach().item()
 
   
        correct_matrix = torch.zeros([target.shape[0], target.shape[1]]).to(device)
        for idx in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[idx][j] >= 0.5:
                    p = label_1
                    
                else:
                    p = label_0
                  
                if target[idx][j] == p:
                    correct = correct + 1
                    correct_matrix[idx][j] = 1
                
        
        #average_loss = loss.sum().item()/(bs*40)
        accuracy = round(correct/(output.shape[1]*output.shape[0]), 5)
        #train_loss = train_loss/len(train_dataloader)
        stop = time.time()
        duration = stop-start
        if i % 100 == 0 or i == len(trainloader)-1:
            print('epoch {}  batch {}/{}  loss {:.3f} accuracy {:.3f} time {} '.format(
                epoch, i, len(trainloader)-1, loss.item(), accuracy#, learning_rate
                         , str(round(duration, 4))+"s."))
    
    test_accuracy = sum(correct_matrix)
    val_accuracy = train_val(validloader)

print(val_accuracy)
torch.save(model, 'resnet178.pth')