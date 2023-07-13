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

# Name Generation Part 2&3: Multi-Layer Perceptron with Batch Normalization
- Notes were made on character-level neural network while watching [Kaparth's videos on makemore](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
- His repo is found [here](https://github.com/karpathy/makemore). 

## Overview
- Breakdown the list of words into character input and character target
- Create a probaility of next character from previous `block_size` characters
   - '.' used as padding and as indication of the end of a word
- Split data into train, validation, and test sets
- Train neural net with following structure:
   - embed character in lower dimensional space, from  27dim to `ebed_dim`
   - use the embeding to convert each context character
   - use a 1 hidden linear layer 
   - batch normalization layer
   - a Tanh activation
   - then to 27 node output layer for classification
   - move from softmax calculation to PyTorch's CrossEntropy

- Finally generate words by sampeling the resultant probability distiributions

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
words = [w  for w in words] # create character to signal begining and end of a word
N_words = len(words)
print(f"number of words     = {N_words}")
print(f"mean word length    = {np.array([len(word) for word in words]).mean() }")
print(f"std_dev word length = {np.array([len(word) for word in words]).std() }")
```

### Create dictionary for integer to character conversion

```python
chars = sorted(set('.'.join(words)))
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
print(itos)
```

### Split Data into training, validation and test sets

```python
import random
random.shuffle(words)
words_tr = words[:int(0.8*N_words)]
words_val = words[int(0.8*N_words):int(0.9*N_words)]
words_test = words[int(0.9*N_words):]
```

```python
block_size = 7 # Context length

def Create_XY_from_words(words):
    X, Y = [], []
    for w in words[:]:
        context= [0] * block_size
        print
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join([itos[i] for i in context]),' ---> ', itos[ix])
            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

Xtr, Ytr = Create_XY_from_words(words_tr)
Xval, Yval = Create_XY_from_words(words_val)
Xtest, Ytest = Create_XY_from_words(words_test)

N = Xtr.shape[0]
```

```python
Xtr.shape, Xtr.dtype, Ytr.shape, Ytr.dtype
```


### Create the character embedding
- Want a linear transformationf from character space (27 elements) to the embedded space (contineous 2D space).
- These will be updated in backpropagation step? 

#### What is embedding (ChatGPT 3.5)
In machine learning, an embedding refers to the process of representing high-dimensional data, such as text or images, in a lower-dimensional space. It involves mapping the data from its original form to a continuous vector space, where each data point is represented by a set of numerical values called embeddings.

Embeddings are designed to capture the underlying relationships and semantic meaning of the data. For example, in natural language processing (NLP), word embeddings are commonly used to represent words as dense vectors. These vectors are trained on large amounts of text data, and words with similar meanings or context tend to have similar vector representations. This allows machine learning algorithms to work with words as numerical inputs rather than discrete symbols, enabling them to capture the semantic relationships between words.

Embeddings can be learned through various techniques such as neural networks, matrix factorization, or dimensionality reduction algorithms. The training process involves optimizing the embeddings to minimize a specific objective, such as preserving semantic relationships or improving the performance of a downstream machine learning task.

Once the embeddings are learned, they can be used as input features for various machine learning models, such as neural networks or clustering algorithms. The lower-dimensional representation provided by embeddings often helps improve the efficiency and effectiveness of these models, as they reduce the dimensionality of the input space and capture relevant features of the data.

```python
embed_dim = 10
C = torch.randn((27,embed_dim))
C[5] # use this rather than matrix multiplication with onehot character input
C[torch.tensor([5,6,7])] # can access based on list or tensor

print(C[Xtr].shape)
# print(C[X][1][2])
# print(C[stoi['e']])
```

```python
#tensors have a 1-d storage and a view
# print(emb.storage())
```

## Initialize weights and biases

```python

if torch.cuda.is_available():
    device=torch.device("cuda:0")
    
else:
    device="cpu"
    
N_hidden = 300 # number of nodes in hidden layer

# Embedding transformation
C = torch.randn((27,embed_dim), device=device) 
# Linear hidden layer
W1 = torch.randn((block_size*embed_dim,N_hidden), device=device) \
    /(block_size*embed_dim)**0.5 # Simple Kaiming init
#b1 = torch.randn(N_hidden, device=device) * 0 # batch norm make this redundent
# Linear output layer
W2 = torch.randn((N_hidden,27), device=device) *.01
b2 = torch.randn(27, device=device) * 0

#batchNormalization paramters
gamma = torch.ones((1, N_hidden), device=device) 
beta  = torch.zeros((1,N_hidden), device=device) 
std_running =   torch.ones((1, N_hidden), device=device)
mean_running  = torch.zeros((1, N_hidden), device=device)

#place input data on device
Xtr = Xtr.to(device)
Xval = Xval.to(device)
Xtest = Xtest.to(device)
Ytr = Ytr.to(device)
Yval = Yval.to(device)
Ytest = Ytest.to(device)


params = [W1, W2, b2, C, gamma, beta]
for param in params:
    param.requires_grad_()

print(f"Number of parameters in the model: {sum(p.nelement() for p in params):,}")
```

```python
emb = C[Xtr] # embeddings of input
```


```python
# most efficient way to reshape/flatten (no new memory)
print(emb.view(-1,emb.shape[1]*emb.shape[2])[1])
```

#### Create function to return loss for a given dataset

```python
nameDict= {'train': (Xtr, Ytr), 'validation': (Xval, Yval), 'test': (Xtest, Ytest)}
def Loss_of_set(setName):
    X,Y = nameDict[setName]
    with torch.no_grad():
        emb = C[X] 
        hpa = emb.view(emb.shape[0],-1) @ W1
        hpa = gamma * (hpa - hpa.mean(dim=0))/hpa.std(dim=0) + beta

        h = torch.tanh(hpa)

        logits = h @ W2 + b2

        return F.cross_entropy(logits, Y).item()
Loss_of_set('train')
Loss_of_set('validation')
```

## Learning loop

```python
loss_tr = []
loss_val = []
```

```python
lr_start, lr_end = 0.01, 0.005
epochs = 20000

for epoch in range(epochs):
    lr = lr_start - (lr_start-lr_end)*epoch/epochs

    # mini-batch # more sample, the more accurate the gradient
    ix = torch.randint(0,Xtr.shape[0],(1600,)) # set for GPU
    
    # embeddings of input    
    emb = C[Xtr[ix]] 

    ### forward step ###
    hpa = emb.view(emb.shape[0],-1) @ W1 + b1
    
    ## batch normalization layer ##
    # reference Ioffe, Szegedy 2015
    # Makes activation input unit std and zero mean at initilization
    # Allows model to learn scale and offest (gamma and beta)
    # Has a regularization effect
    batchStd = hpa.std(dim=0, keepdim=True)
    batchMean = hpa.mean(dim=0, keepdim=True)
    hpa = gamma * (hpa-batchMean)/batchStd + beta

    with torch.no_grad(): # keep a running estimate for inference
        std_running  = 0.95 * std_running  + 0.05 * batchStd
        mean_running = 0.95 * mean_running + 0.05 * batchMean

    ## activation ##
    h = torch.tanh(hpa)

    ## output layer ##
    logits = h @ W2 + b2

    ## Softmax Classification for teaching purposes
    # counts = logits.exp()
    # probs = counts/counts.sum(1, keepdim=True)
    # loss = -probs[torch.arange(N),Y].log().mean() 
    
    ## Cross Entropy Loss - Saves memory and fuses kernels for faster backward pass
    # Also more well-behaved where exp(largePos) -> inf
    # Does this by using an offset
    loss = F.cross_entropy(logits, Ytr[ix])

    ### Backward step ###
    loss.backward()

    for param in params:
        param.data += -lr*param.grad
        param.grad.data.zero_()
 
    ### Output and Error Logging ###
    if epoch%1000 == 0:
        loss_tr.append(Loss_of_set('test')) #recalc to get calc for full test set
        loss_val.append(Loss_of_set('validation'))
        print(f"{epoch:05d}/{epochs} epoch | train/val loss: {loss_tr[-1]:.4f}/{loss_val[-1]:.4f} | lr:{lr:.3f}")
```

### Simple Overfitting Check

```python
plt.plot(loss_tr, label='training')
plt.plot(loss_val, label='validation')
plt.legend()
plt.grid()
plt.title("training and validation loss")
plt.xlabel("epoch [thousand]")
plt.ylabel("negative log loss")
plt.ylim((2,2.2))
```

## Generate Names

```python
W1c = W1.cpu().detach().data
b1c = b1.cpu().detach().data
W2c = W2.cpu().detach().data
b2c = b2.cpu().detach().data
Cc = C.cpu().detach().data
gammac = gamma.cpu().detach().data
betac = beta.cpu().detach().data
std_runningc = std_running.cpu().detach().data
mean_runningc = mean_running.cpu().detach().data
```


```python
for _ in range(10):
    input = [0] * block_size
    input = torch.cat([Cc[i] for i in input])
    name = []
    nextChar = None
    while nextChar != '.':

        hpa = input @ W1c + b1c
        hpa = (hpa-mean_runningc)/std_runningc
        hpa = gammac*hpa + betac
        logit = torch.tanh(hpa) @ W2c +b2c
        c = logit.detach().exp()
        p = c /c.sum()
        p = p.to('cpu')
        p = np.array(p)
        nextChar =np.random.choice(chars, p=p)
        input =torch.cat((input[embed_dim:],Cc[stoi[nextChar]]))
        name.append(nextChar)

    print(''.join(c for c in name[:-1]))
```
