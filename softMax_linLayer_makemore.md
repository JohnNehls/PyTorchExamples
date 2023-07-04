---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    custom_cell_magics: kql
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name Generation Part 1: Single Linear Layer
- Notes were made on character-level neural network while watching [Kaparth's video on makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2). 
- His repo is found [here](https://github.com/karpathy/makemore). 

## Overview
- Breakdown the list of words into character input and character target
- Create a probaility of next character only from the current character
   - '.' used as the end of word character
   - The method of generating probabilites is improved in steps
      - You likely do not want to "Run All" cells. 
- Generate words by sampeling the resultant probability distiributions

## Imports

```python
import torch
import torch.nn.functional as F # used for one-hot encoding 
import numpy as np
import matplotlib.pyplot as plt
```

## Traning Set: Human Names

```python
words = open("./data/names.txt").read().splitlines()
words = ['.' + w + '.' for w in words] # create character to signal begining and end of a word
print(f"number of words = {len(words)}")
words 
```


### Create dictionary for integer to character conversion

```python
chars = sorted(set(''.join(words)))
ctoi = {c:i for i,c in enumerate(chars)}
itoc = {i:c for i,c in enumerate(chars)}
```

## Clean the Data
- turn the data into one-hot encoded input for each character


```python
xs, ys = [], []
for w in words:
    for tmp in list(zip(w, w[1:])):                
        xs.append(tmp[0])
        ys.append(tmp[1])

# xs are the inputs and ys ar the next characters
xs = torch.tensor([ctoi[x] for x in xs])
ys = torch.tensor([ctoi[y] for y in ys])
num =xs.numel() # number of examples in training set

print("number of examples: ", num)

xenc = F.one_hot(xs,num_classes=27).float() # one_hot input
yenc = F.one_hot(ys,num_classes=27).float() # one_hot target (used for niave loss)
```

## Initialize the Weights
Since we are using single, linear layer, mapping 27 vector to a 27 vector the weights will be a 27x27 matrix. 

```python
g = torch.Generator().manual_seed(0)
W = torch.randn(27, 27) 
```

### GPU or CPU
Placing data and weights on the GPU with ID=0 if it is present

```python

if torch.cuda.is_available():
    device=torch.device("cuda:0")
    # Cannot use a simple .to(0) after creating leaf, leaf will remain on cpu
    W = W.cuda().detach().requires_grad_() # place the requires grad leaf on gpu
else:
    device="cpu"
    W = W.requires_grad_() # cpu leaf

xenc = xenc.to(device)
yenc = yenc.to(device)
ys = ys.to(device)
```

## Train the Single Linear Layer
All cells above must be run for any of the flollowing training cells to run successfully. 


### Pedantic Examples
Each onsiders each individual example--useful for learning the process, but too slow to be usable. 


#### Niave loss, per Example

```python
epochs = 2

for i in range(epochs):
    lr = 100 # large learning rate given that the loss ~0.05
    
    loss = torch.zeros(1).to(device) #reset each epoch 

    for x, y in zip(xenc, yenc):
        logit = x @ W # log counts
        counts = logit.exp()
        probs = counts/counts.sum()
        """ 
        Niave Loss: compare the full probability row to the example's target
           Exmaple: Mean ( | [.03, .01, ... ,.04] - [0, ...,1, ... ,0] | ) 
        """
        loss += (probs - y).abs().mean()
    
    loss = loss/num

    print(f'{i} loss = {loss.item()}')
    
    W.grad = None
    loss.backward()
    
    W.data += -lr*W.grad
    W.grad.data.zero_()
```


#### Improved Loss, Per Example


##### New Loss
Here we show an example of how comparing the our loss function works. If we look at the negative log of the element we want to be one, we see how decreasing the loss, drives the element toward 1. 

```python
x = np.linspace(1e-5,1,1000)
plt.plot(x,-np.log(x))
plt.xlabel('x')
plt.ylabel('- log(x)')
plt.title('Loss: minimized by bringing value (x) closer to 1')
```

##### Learning Example

```python
epochs = 2

for i in range(epochs):
    lr = 1 # smaller learning rate given that the loss ~5

    loss = torch.zeros(1).to(device) #reset each epoch 
    
    for x, y in zip(xenc, ys):
        logit = x @ W # log counts
        counts = logit.exp()
        probs = counts/counts.sum()
        """
        Kaparthy's Loss: log of the probabilities of correct pediction, log, negate, mean 
        This drives the exact element we want be 1 nearer to it.
        - works much better than niave loss function
        """
        loss += -probs[y].log()

    loss = loss/num

    print(f'{i} loss = {loss.item()}')
    
    W.grad = None
    loss.backward()
    
    W.data += -lr*W.grad
    W.grad.data.zero_()
```


### Full Matrix Multiplication Example
Each epoch is a single matrix multiplication. 
- The ith row of W holds the logits of the next following the itoc[i] character. 

```python
epochs = 10000 # decrease if not using a GPU! Takes about 1min with an rtx 4 series

for i in range(epochs):
    # Learning rate decreases from lr_start-> lr_end over epochs 
    # It is generally a good idea to reduce the learning rate like this
    lr_start, lr_end = 100, 10
    lr = lr_start - (lr_start-lr_end)*i/epochs
    
    logit = xenc @ W # log counts
    counts = logit.exp()
    probs = counts/counts.sum(1,keepdim=True)
    
    """
    Kaparthy's Loss: log of the probabilities of correct pediction, log, negate, mean 
    This drives the exact element we want be 1 nearer to it.
    - works much better than niave loss function
    """
    loss = -probs[torch.arange(num),ys].log().mean() 
    
    print(f'epoch: {i:04d},  lr: {lr:.3e}, loss: {loss}')
    
    W.grad = None
    loss.backward()
    
    W.data += -lr*W.grad
    W.grad.data.zero_()
```

## Normalize and Plot the Weights

```python
#torch.save(W.data, "./model.weights") # one way to save the weights

weights = W.cpu().detach().data
weightsN = np.zeros(weights.shape)

counts = weights.exp()
weightsN = np.array(counts/counts.sum(1, keepdim=True))

plt.imshow(weightsN)
```

#### Simple check of the results

```python
maxLoc = np.where(weightsN == weightsN.max())

print('Inspect the plot:')
print(f'\tLargest probability: {itoc[maxLoc[0][0]]} to {itoc[maxLoc[1][0]]}')

print(f'\nProbability estimate at maxLocation: {weightsN[maxLoc][0]:.1%}')

print('\nQuick check of occurance in training set:')
!cat ./data/names.txt | grep q | wc
!cat ./data/names.txt | grep qu | wc
print(f'\tOccures often, {206/272:.1%}')
```

## Generate Names


Now we get to generate names from the probability distributions:

```python
for i in range(10):
    genWord = ''
    next = 0
    nextchar=''
    
    while nextchar  != '.':
        nextchar = np.random.choice(chars, p=weightsN[next,:])
        genWord+=nextchar
        next = ctoi[nextchar]

    print(genWord)
```

