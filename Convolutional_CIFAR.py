#!/usr/bin/env python
# coding: utf-8

# In[209]:


import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random, os
from PIL import Image
import glob
from resnet20_cifar import resnet20
import thop
from thop import profile, clever_format


# In[2]:


if torch.cuda.is_available():
    # CUDA is available, you can proceed to use it
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    # CUDA is not available, use CPU
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')


# In[3]:


# ----------------- prepare training data -----------------------
train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# ----------------- prepare testing data -----------------------
test_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)


# In[206]:


# Make DataLoaders for the training and test data, with a batch size of 64
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)


# In[5]:


resnet_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


# In[6]:


class_dict = {0: "airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}


# In[7]:


# Hyperparameters
num_epochs = 10
learning_rate = 0.001


# In[8]:


# Set the seeding for the training process
seed = 1000

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[28]:


class Convolutional_CIFAR(nn.Module):
    def __init__(self):
        super (Convolutional_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.out = nn.Linear(200, 10)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        x = leaky_relu(self.conv1(x))
        x = self.pool(leaky_relu(self.conv2(x)))
        x = leaky_relu(self.conv3(x))
        x = leaky_relu(self.conv4(x))
        x = self.pool(leaky_relu(self.conv5(x)))
        x = x.view(-1, 512 * 3 * 3)
        x = leaky_relu(self.bn1(self.fc1(x)))
        x = leaky_relu(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x


# In[60]:


def train(save_dir = "model",
         model_name = "convolutional-cifar"):
    
    train_list = []
    test_list = []
    
    model = Convolutional_CIFAR()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # If there is no save directory yet, make one
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = model.to(device)

    # Training loop
    for epoch in range(1, num_epochs+1):
        train_correct = 0
        train_total = 0
        for step, (input, labels) in enumerate(train_loader):
            model.train()   # set the model in training mode

            input = input.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(input)

            train_loss = loss_func(outputs, labels)

            train_loss.backward()

            optimizer.step()

            # Every 50 steps, calculate training accuracy, test the model, and print updated results
            if step % 50 == 0:
                # Progress update
                train_acc, accuracy = progress_update(outputs=outputs, labels=labels,
                                                      train_correct=train_correct,
                                                      train_total=train_total,
                                                      train_loss=train_loss,
                                                      model=model, test_loader=test_loader,
                                                      loss_func=loss_func, epoch=epoch, step=step)

                # Save the model
                save_model(model, model_name, accuracy)
                
            # Track training and test accuracies at the end of each epoch (step=750)
            if step == 750:
                train_list.append(train_acc)
                test_list.append(accuracy)
                
        scheduler.step()


# In[70]:


def progress_update(outputs=None, labels=None, train_correct=None, train_total=None,
                    train_loss=None, model=None, test_loader=None, loss_func=None, epoch=None, step=None):
    # Take model's predictions on training set
    # Calculate its accuracy on training set
    _, predicted = torch.max(outputs.data, 1)
    train_total += labels.size(0)
    train_correct += (predicted == labels).sum()
    train_acc = (100 * (train_correct/train_total))
    train_acc = train_acc.item()

    # Now test the model on the test set, get the test accuracy and test loss
    accuracy, test_loss = test(model=model, test_loader=test_loader, loss_func=loss_func)

    # Show the results
    print('Epoch[{}]:Step[{}] Train Loss: {:.2f}\tTrain Acc: {:.2f}%\tTest Loss: {:.2f}\t\tTest Acc: {:.2f}%'.format(
    epoch, step, train_loss, train_acc, test_loss, accuracy))
    
    return train_acc, accuracy


# In[257]:


def test(model=None, test_image=None, test_loader=None, loss_func=None):
    correct = 0
    total = 0

    if model is None:
        model = load_model()

    model = model.to(device)

    # Transform the image to match the format expected by the model
    # This is primarily for single image inferences
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # If a test_loader is not provided, create a DataLoader to test a single image
    if test_loader is None:
        try:
            inference(test_image=test_image, model=model, transform=transform)
        except Exception as e:
            print(e)
            print()
            print("Pass in an image to be classified!")
            print("If you already did, but are still getting an error, make sure you have trained the model already")
            print("Call 'python classify.py train' from the command line first, let it train, then test your image")
            print()
    # If a test_loader IS provided, i.e. every 50 steps during the training loop, execute the code below
    else:
        try:
            accuracy, test_loss = evaluate(model=model, test_loader=test_loader,
                                                    loss_func=loss_func, correct=correct, total=total)
            return accuracy, test_loss
        except Exception as e:
            print(e)
            print()
            print("If you are here, I have no idea how")
            print("You might not have a model trained and saved to your present working directory")
            print("Call the train method on this classifier to train and save a model, then test it")
            print()


# In[258]:


def inference(test_image=None, model=None, transform=None):
    # Load the test image and apply transformations
    test_image = Image.open(test_image).convert('RGB')
    test_image = transform(test_image).unsqueeze(0)
    test_image = test_image.to(device)

    model.eval()  # switch the model to evaluation mode
    with torch.no_grad():
        def hook_fn(module, input, output):
            global activation
            activation = output

        # Attach the hook to the first convolutional layer
        first_conv_layer = model.conv1
        hook = first_conv_layer.register_forward_hook(hook_fn)
                
        output = model(test_image)
        label_value, label_index = torch.max(output, 1)
        answer = class_dict[label_index.item()]
        print("This is an image of a(n): ", answer)

        hook.remove()
                
        # Check if activation was captured
        if activation is not None:
            # Access the first layer activation (feature maps)
            feature_maps = activation[0]
            num_filters = feature_maps.size(0)
            # Visualize the feature maps
            fig, axarr = plt.subplots(4, 8)  # Assuming 32 filters, adjust as needed
            for i in range(num_filters):
                ax = axarr[i // 8, i % 8]
                ax.imshow(feature_maps[i].cpu(), cmap='viridis')
                ax.axis('off')
            plt.show()
            plt.savefig('CONV_rslt.png')
        else:
            print("No activation captured.")


# In[259]:


def evaluate(model=None, test_loader=None, loss_func=None, correct=None, total=None):
    model.eval()  # switch the model to evaluation mode
    with torch.no_grad():
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_loss = loss_func(outputs, labels)

            # Increment total number of observations seen by
            # number of items in this batch
            total += labels.size(0)

            # Increment total number of correct predictions by
            # number of correct predictions in this batch
            correct += (predicted == labels).sum()

        accuracy = (100 * (correct/total))
        accuracy = accuracy.item()
        return accuracy, test_loss


# In[260]:


class ResnetDataWithTransform(Dataset):
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample, label = self.original_dataset[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# In[261]:


def make_resnet20():
    loss_func = nn.CrossEntropyLoss()
                
    transformed_resnet_dataset = ResnetDataWithTransform(test_data, transform=resnet_transform)
    
    # Create a new DataLoader with the transformed dataset
    resnet_test_loader = Data.DataLoader(transformed_resnet_dataset, batch_size=64, shuffle=False)

    model = resnet20()
    model_path = "./resnet20_cifar10_pretrained.pt"
    model.load_state_dict(torch.load(model_path))    
    accuracy, test_loss = test(model=model, test_loader=resnet_test_loader, loss_func=loss_func)
    print("ResNet20 Test Loss: {:.2f}\tResNet20 Test Accuracy: {:.2f}".format(test_loss, accuracy))


# In[193]:


def save_highest_accuracy(accuracy):
    with open("highest_accuracy.txt", "w") as f:
        f.write(str(accuracy))


# In[194]:


def load_highest_accuracy():
    if os.path.exists("highest_accuracy.txt"):
        with open("highest_accuracy.txt", "r") as f:
            return float(f.read())
    else:
        return 0.0  # Default to 0.0 if the file doesn't exist


# In[195]:


def save_model(model, model_name, accuracy):

    highest_accuracy = load_highest_accuracy()

    # Only save the model if its accuracy is higher than the previous model's
    if accuracy > highest_accuracy:

        save_highest_accuracy(accuracy)

        file_path = "./model/{}.pt".format(model_name)

        print("New highest accuracy. Saving model ...")
        print()

        torch.save(model.state_dict(), file_path)


# In[196]:


def load_model(model=None, device="cuda"):

    if model is None:
        model = Convolutional_CIFAR()

    try:
        model.load_state_dict(torch.load("./model/convolutional-cifar.pt"))
        device = torch.device(device)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return None


# In[197]:


# model = Convolutional_CIFAR()

# # Create a sample input tensor
# input_tensor = torch.randn(1, 3, 32, 32)  

# # Profile the model to count MACs
# macs, params = profile(model, inputs=(input_tensor,))
# macs, params = clever_format([macs, params], "%.3f")
# print(f"MACs: {macs}, Parameters: {params}")


# In[198]:


# res_model = resnet20()
# res_model_path = "./resnet20_cifar10_pretrained.pt"
# res_model.load_state_dict(torch.load(res_model_path)) 

# # Profile the model to count MACs
# macs, params = profile(res_model, inputs=(input_tensor,))
# macs, params = clever_format([macs, params], "%.3f")
# print(f"MACs: {macs}, Parameters: {params}")


# In[199]:


# i = 10
# image = test_data[i][0]
# print(test_data[i][1])

# plt.imshow(transforms.ToPILImage()(image))
# plt.show()


# In[201]:


# test(test_image="/home/jts75596/deep_learning/HW2/plane.png")


# In[112]:


# train_list = []
# test_list = []
# train()


# In[29]:


# epoch_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[30]:


# plt.plot(epoch_list, train_list, label="Training Accuracy", color="blue")
# plt.plot(epoch_list, test_list, label="Testing Accuracy", color="orange")

# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Accuracies At the End of Each Epoch")
# plt.legend()
# plt.show()


# In[31]:


# train_list


# In[32]:


# highest_acc_list = [80.21, 80.63, 80.72]
# print("Mean accuracy: ", np.mean(highest_acc_list))
# print("Standard deviation: ", np.std(highest_acc_list))

