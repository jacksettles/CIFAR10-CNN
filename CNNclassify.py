#!/usr/bin/env python
# coding: utf-8

# In[4]:


import argparse
import os
import sys
from Convolutional_CIFAR import Convolutional_CIFAR, train, test, make_resnet20


# In[5]:


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Train or test CIFAR10_Classifier")
    
    parser.add_argument('action', choices = ['train', 'test', 'resnet20'], help = "Action to perform (train, test, or resnet20)")
    parser.add_argument('filename', nargs = "?", help = "File name to test (required for testing)")
    
    args = parser.parse_args()
    
    if args.action == "train":
        train()
    elif args.action == "test":
        if args.filename:
            test(test_image = args.filename)
        else:
            parser.error('--filename is required for test action')
    elif args.action == "resnet20":
        make_resnet20()


# In[ ]:




