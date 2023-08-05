import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
import time 

from torch.utils.tensorboard import SummaryWriter

train_raw=datasets.MNIST(root="datas",train=True,download=True,transform=ToTensor())
val_raw=datasets.MNIST(root="datas",train=False,download=True,transform=ToTensor())

train_input=DataLoader(train_raw,batch_size=32,shuffle=True)
val_input=DataLoader(val_raw,batch_size=32,shuffle=False)

board=SummaryWriter()

train_image,train_label=next(iter(train_input))
train_grid=torchvision.utils.make_grid(train_image)

val_image,val_label=next(iter(val_input))
val_grid=torchvision.utils.make_grid(val_image)

board.add_image("Train_grid",train_grid)
board.add_image("Val_grid",val_grid)
board.close()


if torch.cuda.is_available():
    device =("cuda")
else:
    device =("cpu")


class image_classifier(nn.Module): 
    def __init__(self):
        super(image_classifier,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)      
        ) 
    def forward(self,x):
        return self.model(x)

    
model=image_classifier().to(device)
optimizer=Adam(params=model.parameters(),lr=0.01)
loss_function=nn.CrossEntropyLoss()


def train_one_epoch():
    total_images=0
    correct_images=0
    running_loss1=0

    for i,data in enumerate(train_input):  
        img,lab=data
        img,lab=img.to(device),lab.to(device)
        
        optimizer.zero_grad()
        
        output=model(img)
        train_loss=loss_function(output,lab)
        train_loss.backward()
        optimizer.step()
        
        _,train_pred=torch.max(output.data, 1)
        
        
        total_images=total_images+lab.size(0)
        correct_images=correct_images+(train_pred==lab).sum().item()
        train_accuracy=(correct_images/total_images)
        
           
        running_loss1 += train_loss
    avg_loss = running_loss1 / (i + 1)
    
    return avg_loss,train_accuracy
    
epoch=30

start=time.time()
for epoch in range(epoch):
    print('EPOCH {}:'.format(epoch + 1))

    model.train(True)
    train_avg_loss,train_accuracy=train_one_epoch()
    print("Training loss : {} and Training accuracy : {}".format(train_avg_loss,train_accuracy))
    
    
    running_loss=0
    model.eval()
    total_images=0
    correct_image=0
    
    with torch.no_grad():
        for i1 ,data in enumerate(val_input):
            v_image,v_label=data
            v_image,v_label=v_image.to(device),v_label.to(device)
            v_output=model(v_image)
            v_loss=loss_function(v_output,v_label)
            running_loss=running_loss+v_loss
            
            _,v_prediction=torch.max(v_output.data,1)
            total_images =total_images+v_label.size(0)
            correct_image=correct_image+(v_prediction==v_label).sum().item()
            v_accuracy=(correct_image/total_images)
    val_avg_loss=running_loss/(i1+1)
        
    print("Validation loss : {} and Validation accuracy : {}".format(val_avg_loss,v_accuracy))
            
    board.add_scalars('Training vs. Validation Loss',
                    { 'Training' : train_avg_loss, 'Validation' : val_avg_loss },
                    epoch+ 1)
    
    board.add_scalar('Training Loss',train_avg_loss,epoch+ 1)
    board.add_scalar('Validation Loss',val_avg_loss,epoch+ 1)
    board.add_scalar('Training Accuracy',train_accuracy,epoch+ 1)
    board.add_scalar('Validation Accuracy',v_accuracy,epoch+ 1)
    board.close()
    
    
