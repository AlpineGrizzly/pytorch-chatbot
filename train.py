# train.py
# Train our ML and neural net model

import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

def main():
    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(x_train)
            self.x_data = x_train
            self.y_data = y_train

        def __getitem__(self, idx):
            return self.x_data[idx], self.y_data[idx]

        def __len__(self):
            return self.n_samples

    with open('intents.json', 'r') as f:
        intents = json.load(f)
    
    all_words = []
    tags = []
    xy = []
    
    # Create our all_words list and x and y training data
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))
    
    # Filter, sort, and stem our all_words list
    ignore_words = ['?', '!', '.', ',']
    all_words =[stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    
    # x and y training data arrays
    x_train = []
    y_train = []
    
    for (pattern_sentence, tag) in xy: 
    # Append our bag of words to x training data
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag) # Append to our training data
    
    # Append our labels and tags to the y training data
        label = tags.index(tag)
        y_train.append(label) # CrosEntropyLoss
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    learning_rate = 0.001
    num_epochs = 1000
    
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Leveraging the GPU if available
    model = NeuralNet(input_size, hidden_size, output_size).to(device)    # Create our model from the neural net
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels= labels.to(device)
    
            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
    
            # Backpropagation and optimizer step 
            optimizer.zero_grad() # Gradient
            loss.backward() #retain_graph=True can be used to save information
            optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, loss={loss.item():.4f}')
    
    print(f'Final loss, loss={loss.item():.4f}')
    
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }
    
    FILE = "data.pth"
    torch.save(data, FILE) # Serialize and save to file
    
    print(f'Training complete, filed saved to {FILE}')
    
if __name__ == "__main__":
    main()

