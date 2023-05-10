# pytorch-chatbot-tutorial
Tutorial from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

## Overview
Simple chatbot leveraging machine learning capabilities with Pytorch

## Training the Chatbot
Firstly, we'll need to train the chatbot on patterns to look for so it will know how to respond to the user. This is accomplished via the `train.py` program. Simple run the following. 
```
python3 train.py
```
This will kick off the learning with printed statistics to console such as the epoch and the loss associated with the model. The lower our loss, the better the predictions will be.

## Starting the chatbot
Once we have trained the model for the chatbot, we can now begin chatting with the bot by using the following:
```
python3 chat.py
``` 
## Cool docs
Neural Networks: https://towardsdatascience.com/neural-networks-forward-pass-and-backpropagation-be3b75a1cfcc