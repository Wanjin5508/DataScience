![[Pasted image 20240715104215.png]]

# 1 Intro

![[Pasted image 20240715104336.png]]
》》 适合结构化和非结构化的数据


## Tensors
![[Pasted image 20240715104458.png]]

![[Pasted image 20240715104612.png]]

![[Pasted image 20240715104654.png]]

![[Pasted image 20240715104735.png]]


## #ue Creating tensors and accessing attributes

Tensors are the primary **data structure** in PyTorch and will be the building blocks for our deep learning models. They share many similarities with NumPy arrays but have some unique attributes too.

```python
# Import PyTorch
import torch

list_a = [1, 2, 3, 4]

# Create a tensor from list_a
tensor_a = torch.tensor(list_a)

# Display the tensor device
print(tensor_a.device)

# Display the tensor data type
print(tensor_a.dtype)
```
![[Pasted image 20240715105147.png]]

## #ue  Creating tensors from NumPy arrays

Tensors are the fundamental data structure of PyTorch. You can create complex deep learning algorithms by learning how to manipulate them.

The `torch` package has been imported, and two NumPy arrays have been created, named `array_a` and `array_b`. Both arrays have the same dimensions.

- Create two tensors, `tensor_a` and `tensor_b`, from the NumPy arrays `array_a` and `array_b`, respectively.
- Subtract `tensor_b` from `tensor_a`.
- Perform an element-wise multiplication of `tensor_a` and `tensor_b`.
- Add the resulting tensors from the two previous steps together.


```python
# Create two tensors from the arrays
tensor_a = torch.from_numpy(array_a)
tensor_b = torch.from_numpy(array_b)  # 注意这里的区别

# Subtract tensor_b from tensor_a 
tensor_c = tensor_a - tensor_b

# Multiply each element of tensor_a with each element of tensor_b
tensor_d = tensor_a * tensor_b

# Add tensor_c with tensor_d
tensor_e = tensor_c + tensor_d
print(tensor_e)
```


## Creating our first neural network

### 首先创建一个没有隐藏层的 2-layer network
![[Pasted image 20240715105709.png]]

![[Pasted image 20240715105826.png]]

![[Pasted image 20240715105930.png]]

![[Pasted image 20240715110048.png]]
》》 注意学习如何调整 tune 权重和偏置


![[Pasted image 20240715110201.png]]
》》》 全连接层即线性层，指的是每个神经元与下一层的每个神经元相连接

### 堆叠多个层
![[Pasted image 20240715110420.png]]
》》 注意每个线性层的 输入输出维度 要满足矩阵运算的条件

![[Pasted image 20240715110659.png]]


## #ue  Your first neural network

In this exercise, you will implement a small neural network containing two **linear** layers. The first layer takes an eight-dimensional input, and the last layer outputs a one-dimensional tensor.

The `torch` package and the `torch.nn` package have already been imported for you.

- Create a neural network of linear layers that takes a tensor of dimensions 1×8 as input and outputs a tensor of dimensions 1×1. 》》 整个模型的输入张量的维度是 1 * 8， 输出是 1 by 1.
- Use any output dimension for the first layer you want.

```python
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[2, 3, 6, 7, 9, 3, 2, 1]])

# Implement a small neural network with exactly two linear layers
model = nn.Sequential(nn.Linear(8, 4),
                      nn.Linear(4, 1)
                     )

output = model(input_tensor)
print(output)
```

###### Hint

- Recall that you can use `nn.Linear()` 要大写！！ to create a linear layer, which takes two arguments, `in_features` and `out_features`.
- Since the dimension of the input tensor is 1x8, this means it has 8 features; so the first argument of the first linear layer should be 8.
- Each *subsequent layer* of a neural network must have an input dimension ***equal*** to the previous layer's output dimension.
- Since we are looking for a one-dimensional output, the second argument of the last linear layer should be 1.


## Discovering activation functions
使用激活函数，给神经网络添加非线性部分

在此之前我们只是用了线性层，即使堆叠多个线性层，模型的输出始终与输入有着线性相关。这对于复杂问题的解决是毫无意义的

![[Pasted image 20240715112038.png]]

The **sigmoid** and **softmax** functions are two of the most popular activation functions in deep learning. They are both usually used as the last step of a neural network. Sigmoid functions are used for binary classification problems, whereas softmax functions are often used for multi-class classification problems. This exercise will familiarize you with creating and using both functions.

### sigmoid
主要用于 2分类 问题

![[Pasted image 20240715112205.png]]

![[Pasted image 20240715112325.png]]

sigmoid 不会改变输入张量的维度，只是将输入张量转化成 0 和 1 之间的数

![[Pasted image 20240715112505.png]]

### softmax
![[Pasted image 20240715112633.png]]
》》 输出的概率分布中，数值最高的类别则表示正确的分类


![[Pasted image 20240715112839.png]]

# 2 Training Our First Neural Network with PyTorch
训练过程中，loss function 非常重要。因此需要首先理解损失函数的作用。

整个训练过程 实际上就是通过 梯度下降 算法 寻找损失函数最小化时的参数组合

## Running a forward pass

![[Pasted image 20240715113902.png]]


![[Pasted image 20240715113949.png]]
》》》 在训练过程中， 反向传播是正向传播的补充

### 二元分类问题中的正向传播

![[Pasted image 20240715114136.png]]

输入张量共有 5 行，即5个样本 （datapoint），这些样本指的是动物个体。
每个样本都有 6 列，表示 6 个特征 （features），说明模型需要 6 个神经元来处理这些特征。那么网络的输入层的第一个参数 是 6

网络搭建如下：
![[Pasted image 20240715114442.png]]


输出：
![[Pasted image 20240715114517.png]]
》》》 输出的张量中同样由 5 个样本组成，即我们输入到模型中的 5 个动物实例。
![[Pasted image 20240715114623.png]]


### multi-class
![[Pasted image 20240715114728.png]]

![[Pasted image 20240715114820.png]]

### regression：预测连续数值
#### #ue From regression to multi-class classification

Recall that the models we have seen for binary classification, multi-class classification and regression have all been similar, barring a few tweaks to the model.

In this exercise, you'll start by building a model for regression, and then tweak the model to perform a multi-class classification.

- Create a neural network with exactly four linear layers, which takes the input tensor as input, and outputs a regression value, using any shapes you like for the hidden layers.

###### Hint

- Recall that in order to return a regression value, the model should have ***no activation*** function at the end and should ***return an output of size one***.
- ***Any*** hidden layer shapes should work, as long as the input dimensions of each layer are the same as the output dimensions of each previous layer.

回归任务，只用 4 个线性层即可：
~~~python
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Implement a neural network with exactly four linear layers
model = nn.Sequential(
    nn.Linear(11, 10),
    nn.Linear(10, 18),
    nn.Linear(18, 20),
    nn.Linear(20, 1)
)

output = model(input_tensor)
print(output)
~~~

多分类任务，使用 softmax，注意最后的输出层的第二个参数，等于标签数量

~~~python
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Update network below to perform a multi-class classification with four labels
model = nn.Sequential(
  nn.Linear(11, 20),
  nn.Linear(20, 12),
  nn.Linear(12, 6),
  nn.Linear(6, 4), 
  
  nn.Softmax(dim=-1)
)

output = model(input_tensor)
print(output)
~~~


#### #ue  Building a binary classifier in PyTorch

Recall that a small neural network with a single linear layer followed by a sigmoid function is a binary classifier. It acts just like a logistic regression.

In this exercise, you'll practice building this small network and interpreting the output of the classifier.

- Create a neural network that takes a tensor of dimensions 1x8 as input, and returns an output of the correct shape for binary classification.
- Pass the output of the linear layer to a sigmoid, which both takes in and return a single float.

```python
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 2, 3, 6, 8, 9]])

# Implement a small neural network for binary classification
model = nn.Sequential(
  nn.Linear(8,1),
  nn.Sigmoid()
)

output = model(input_tensor)
print(output)
```

## Using loss functions to assess model predictions

![[Pasted image 20240715120438.png]]

![[Pasted image 20240715120617.png]]
int 值和 张量的比较，可以通过独热编码来实现 》》 将表示类别的整数转化成**相同维度的**张量：
![[Pasted image 20240715120703.png]]
![[Pasted image 20240715120846.png]]


![[Pasted image 20240715120935.png]]


![[Pasted image 20240715121026.png]]
》 输出是 loss 值

![[Pasted image 20240715121153.png]]



### #ue  Creating one-hot encoded labels

One-hot encoding is a technique that turns a single integer label into a vector of N elements, where N is the number of classes in your dataset. This vector only contains zeros and ones. In this exercise, you'll create the one-hot encoded vector of the label `y` provided.

You'll practice doing this manually, and then make your life easier by leveraging the help of PyTorch! Your dataset contains three classes.

NumPy is already imported as `np`, and `torch.nn.functional` as `F`. The `torch` package is also imported.

If you implement a custom dataset, you can make it output the one-hot encoded label directly. Indeed, you can add the one-hot encoding step to the `__getitem__` method such that the returned label is already one-hot encoded!

~~~python
y = 1
num_classes = 3

# Create the one-hot encoded vector using NumPy
one_hot_numpy = np.array([0, 1, 0])

# Create the one-hot encoded vector using PyTorch
one_hot_pytorch = F.one_hot(torch.tensor(y), num_classes)
~~~

### #ue  Calculating cross entropy loss

Cross entropy loss is the most used loss for **classification** problems. In this exercise, you will create inputs and calculate cross entropy loss in PyTorch. You are provided with the **ground truth label** `y` and a vector of `scores` predicted by your model.

You'll start by creating a one-hot encoded vector of the ground truth label `y`, which is a required step to compare `y` with the scores predicted by your model. Next, you'll create a cross entropy loss function. Last, you'll call the loss function, which takes `scores` (model predictions before the final softmax function), and the one-hot encoded ground truth label, as inputs. It outputs a single float, the loss of that sample.

`torch`, `torch.nn` as `nn`, and `torch.nn.functional` as `F` have already been imported for you.

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# Create a one-hot encoded vector of the label y
one_hot_label = F.one_hot(torch.tensor(y), scores.shape[1])

# Create the cross entropy loss function
criterion = nn.CrossEntropyLoss()

# Calculate the cross entropy loss
loss = criterion(scores.double(), one_hot_label.double())
print(loss)
~~~

![[Pasted image 20240715122158.png]]


### Using derivatives to update model parameters

![[Pasted image 20240718172907.png]]
>> Gradient

![[Pasted image 20240718173002.png]]


反向传播的原理：
![[Pasted image 20240718173116.png]]

![[Pasted image 20240718173155.png]]
> .backward() 用于计算上一行实例化的损失函数 loss 的梯度

如下可以获取每一层的 weight 和 bias 的梯度 （使用 .grad 属性）
层的索引从0 开始



![[Pasted image 20240718173532.png]]
》》 手动更新神经网路第一层的 权重 和 偏置。更新的逻辑都一样，即 $$ new \ weight=old \ weight - learning\ rate*gradient $$ 

![[Pasted image 20240718173843.png]]

![[Pasted image 20240718174011.png]]

![[Pasted image 20240718174050.png]]
> step() 函数： optimizer 计算梯度，并自动更新模型的参数

## Estimating a sample

In previous exercises, you used linear layers to build networks.

Recall that the operation performed by `nn.Linear()` is to take an input X and apply the transformation $W*X+b$ ,where W and b are two tensors (called the weight and bias).

A critical part of training PyTorch models is to calculate gradients of the weight and bias tensors with respect to a loss function.

In this exercise, you will calculate weight and bias tensor gradients using cross entropy loss and a sample of data.

The following tensors are provded:

- `weight`: a 2×9-element tensor
- `bias`: a 2-element tensor
- `preds`: a 1×2-element tensor containing the model predictions
- `target`: a 1×2-element one-hot encoded tensor containing the ground-truth label

~~~python
criterion = nn.CrossEntropyLoss()

# Calculate the loss
loss = criterion(preds, target)

# Compute the gradients of the loss
loss.backward()

# Display gradients of the weight and bias tensors in order
print(weight.grad)
print(bias.grad)
~~~
![[Pasted image 20240718175028.png]]


### #ue  Accessing the model parameters

A PyTorch model created with the `nn.Sequential()` is a module that contains the different layers of your network. Recall that each layer parameter can be accessed by indexing the created model directly. In this exercise, you will practice accessing the parameters of different **linear** layers of a neural network. You won't be accessing the sigmoid.

~~~python
model = nn.Sequential(nn.Linear(16, 8),
                      nn.Sigmoid(),
                      nn.Linear(8, 2))

# Access the weight of the first linear layer
weight_0 = model[0].weight

# Access the bias of the second linear layer
bias_1 = model[2].bias
~~~

> Parameters and gradients are usually automatically calculated and updated, as we've seen in the video. However, on your deep learning journey, you'll find that accessing the model parameters or gradients is a great way to debug training.

### #ue  Updating the weights manually

Now that you know how to access weights and biases, you will manually perform the job of the PyTorch optimizer. PyTorch functions can do what you're about to do, but it's helpful to do the work manually at least once, to understand what's going on under the hood.

A neural network of three layers has been created and stored as the `model` variable. This network has been used for a forward pass and the loss and its derivatives have been calculated. A default learning rate, `lr`, has been chosen to scale the gradients when performing the update.

~~~python
weight0 = model[0].weight
weight1 = model[1].weight
weight2 = model[2].weight

# Access the gradients of the weight of each linear layer
grads0 = weight0.grad
grads1 = weight1.grad
grads2 = weight2.grad

# Update the weights using the learning rate and the gradients
weight0 = weight0 - lr * grads0
weight1 = weight1 - lr * grads1
weight2 = weight2 - lr * grads2
~~~


### #ue  Using the PyTorch optimizer

In the previous exercise, you manually updated the weight of a network. You now know what's going on under the hood, but this approach is not scalable to a network of many layers.

Thankfully, the PyTorch SGD optimizer does a similar job in a handful of lines of code. In this exercise, you will practice the last step to complete the training loop: updating the weights using a PyTorch optimizer.

A neural network has been created and provided as the `model` variable. This model was used to run a forward pass and create the tensor of predictions `pred`. The one-hot encoded tensor is named `target` and the cross entropy loss function is stored as `criterion`.

`torch.optim` as `optim`, and `torch.nn` as `nn` have already been loaded for you.

~~~python
# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss = criterion(pred, target)
loss.backward()

# Update the model's parameters using the optimizer
optimizer.step()
~~~

## Writing our first training loop
![[Pasted image 20240718180317.png]]

![[Pasted image 20240718180356.png]]
>> 使用深度学习做回归任务：
>>  target 是连续的数值，要注意以下几点：
>>  1. softmax 和 sigmoid 不能被用于 回归模型 的最后一个激活函数。最后一层是 线性层
>>  2. 损失函数也不能使用 cross entropy，用 MSE

![[Pasted image 20240718180727.png]]

![[Pasted image 20240718180845.png]]
》》该数据集有 4 个输入特征，和一个目标（输出）

以下是一个训练循环，由两个 for 组成：
![[Pasted image 20240718181157.png]]
- 注意内层 for 循环，dataloader 在每次迭代中提供 一个 batch 的数据
- 将 optimizer 的梯度归零的原因是，optimizer 在默认情况下会保存前面步骤的梯度


### #ue  Using the MSELoss

Recall that we can't use cross-entropy loss for regression problems. The mean squared error loss (MSELoss) is a common loss function for regression problems. In this exercise, you will practice calculating and observing the loss using NumPy as well as its PyTorch implementation.

The `torch` package has been imported as well as `numpy` as `np` and `torch.nn` as `nn`.

~~~python
y_hat = np.array(10)
y = np.array(1)

# Calculate the MSELoss using NumPy
mse_numpy = np.mean( (y_hat-y) ** 2)

# Create the MSELoss function
criterion = nn.MSELoss()

# Calculate the MSELoss using the created loss function
mse_pytorch = criterion(torch.tensor(y_hat).float(), torch.tensor(y).float())
print(mse_pytorch)
~~~
![[Pasted image 20240718182011.png]]

 ###### Hint

- The MSE loss is the squared difference between predictions and targets.
- Recall the `.MSELoss()` function, which is part of the `nn` module.
- The `criterion` takes `.float()` tensors as input, so remember to call `.float()` on your arguments to criterion!

### #ue  Writing a training loop

In `scikit-learn`, the whole training loop is contained in the `.fit()` method. In PyTorch, however, you implement the loop manually. While this provides control over loop's content, it requires a custom implementation.

You will write a training loop every time you train a deep learning model with PyTorch, which you'll practice in this exercise. The `show_results()` function provided will display some sample ground truth and the model predictions.

The package imports provided are: pandas as `pd`, `torch`, `torch.nn` as `nn`, `torch.optim` as `optim`, as well as `DataLoader` and `TensorDataset` from `torch.utils.data`.

The following variables have been created: `dataloader`, containing the dataloader; `model`, containing the neural network; `criterion`, containing the loss function, `nn.MSELoss()`; `optimizer`, containing the SGD optimizer; and `num_epochs`, containing the number of epochs.
[[#A deeper dive into neural network architecture]]
~~~python
# Loop over the number of epochs and the dataloader
for i in range(num_epochs):
  for data in dataloader:
    # Set the gradients to zero
    optimizer.zero_grad() #################################
    # Run a forward pass
    feature, target = data
    prediction = model(feature)    
    # Calculate the loss
    loss = criterion(prediction, target)    
    # Compute the gradients
    loss.backward()
    # Update the model's parameters
    optimizer.step()
show_results(model, dataloader)
~~~
![[Pasted image 20240718182544.png]]

# 3 Neural Network Architecture and Hyperparameters
Hyperparameters are parameters, often chosen by the user, that control model training. The type of activation function, the number of layers in the model, and the learning rate are all hyperparameters of neural network training. Together, we will discover the most critical hyperparameters of a neural network and how to modify them.


## Discovering activation functions between layers
前面学的 softmax 和 sigmoid 激活函数只能用于网络的最后一层

本章会学习能够用于 线性层之间的 激活函数

![[Pasted image 20240718194253.png]]
> 梯度消失：
> 因为 local gradient 是前面的 gradient 的函数。如果x 的值 过大或者过小，梯度将会接近零，从而阻止了 weight 的更新和变化

#### ReLU
修正线性单元，这个激活函数没有上线，因此即使 x 的值非常大，其梯度也不会收敛，从而避免了梯度消失问题
![[Pasted image 20240718194733.png]]

Relu 是一个很好的默认激活函数

![[Pasted image 20240718194837.png]]


### #ue  Implementing ReLU

The rectified linear unit (or ReLU) function is one of the most common activation functions in deep learning.

It overcomes the training problems linked with the sigmoid function you learned, such as the **vanishing gradients problem**.

In this exercise, you'll begin with a ReLU implementation in PyTorch. Next, you'll calculate the gradients of the function.

The `nn` module has already been imported for you.

~~~python
# Create a ReLU function with PyTorch
relu_pytorch = nn.ReLU()

# Apply your ReLU function on x, and calculate gradients
x = torch.tensor(-1.0, requires_grad=True)
y = relu_pytorch(x)
y.backward() # calculate gradients

# Print the gradient of the ReLU function for x
gradient = x.grad
print(gradient)
~~~
![[Pasted image 20240718195447.png]]

## A deeper dive into neural network architecture
![[Pasted image 20240718200005.png]]

![[Pasted image 20240718200132.png]]

![[Pasted image 20240718200249.png]]
![[Pasted image 20240718200329.png]]


# Manipulating the capacity of a network

In this exercise, you will practice creating neural networks with different capacities. The capacity of a network reflects the number of parameters in said network. To help you, a `calculate_capacity()` function has been implemented, as follows:

```
def calculate_capacity(model):
  total = 0
  for p in model.parameters():
    total += p.numel()
  return total
```

This function returns the number of parameters in the your model.

The dataset you are training this network on has `n_features` features and `n_classes` classes. The `torch.nn` package has been imported as `nn`.

~~~python
n_features = 8
n_classes = 2

input_tensor = torch.Tensor([[3, 4, 6, 2, 3, 6, 8, 9]])

# Create a neural network with less than 120 parameters
model = nn.Sequential(
    nn.Linear(n_features, 6),
    nn.Linear(6, 4),
    nn.Linear(4, n_classes)
)
output = model(input_tensor)

print(calculate_capacity(model))
~~~


## Learning rate and momentum
![[Pasted image 20240718201254.png]]

![[Pasted image 20240718201337.png]]
 在越接近最优点的时候，optimizer 走的步幅越小 》》》 这是因为 step size 等于 梯度 乘以 lr
 -> x 越接近 0，函数的走势越缓

![[Pasted image 20240718201744.png]]

![[Pasted image 20240718201755.png]]

![[Pasted image 20240718201836.png]]
![[Pasted image 20240718201846.png]]
![[Pasted image 20240718201921.png]]

![[Pasted image 20240718201941.png]]

## Layer initialization and transfer learning
![[Pasted image 20240718202504.png]]
![[Pasted image 20240718202539.png]]
![[Pasted image 20240718202615.png]]

![[Pasted image 20240718202708.png]]

torch.save & torch.load: 可以用于任何的 pytorch 对象，可以是一个单一的 layer 或者是 整个模型

迁移学习的一种特定的形式： 微调。有个潜规则：我们通常会冻结前面的层，只微调接近输出层的层
![[Pasted image 20240718203040.png]]
>>>> set the `require_grad` of the first layer's weight to false


### #ue  Fine-tuning process

You are training a model on a new dataset and you think you can use a fine-tuning approach instead of training from scratch (i.e., training from randomly initialized weights).

To fine-tune a model on a new task, you need to follow a certain number of steps that are listed here.

![[Pasted image 20240718222345.png]]


### #ue Freeze layers of a model

You are about to fine-tune a model on a new task after loading pre-trained weights. The model contains three linear layers. However, because your dataset is small, you only want to train the last linear layer of this model and freeze the first two linear layers.

The model has already been created and exists under the variable `model`. You will be using the `named_parameters` method of the model to list the parameters of the model. Each parameter is described by a name. This name is a string with the following naming convention: `x.name` where `x` is the index of the layer.

Remember that a linear layer has two parameters: the `weight` and the `bias`.

~~~python
for name, param in model.named_parameters():    
  
    # Check if the parameters belong to the first layer
    if name == '0.weight' or name == '0.bias':
      
        # Freeze the parameters
        param.requires_grad = False
  
    # Check if the parameters belong to the second layer
    if name == '1.weight' or name == '1.bias':
      
        # Freeze the parameters
        param.requires_grad = False
~~~

### #ue  Layer initialization

The initialization of the weights of a neural network has been the focus of researchers for many years. When training a network, the method used to initialize the weights has a direct impact on the final performance of the network.

As a machine learning practitioner, you should be able to experiment with different initialization strategies. In this exercise, you are creating a small neural network made of two layers and you are deciding to initialize each layer's weights with the uniform method.

~~~python
layer0 = nn.Linear(16, 32)
layer1 = nn.Linear(32, 64)

# Use uniform initialization for layer0 and layer1 weights
nn.init.uniform_(layer0.weight)
nn.init.uniform_(layer1.weight)

model = nn.Sequential(layer0, layer1)
~~~

>>  The uniform initialization is one of the many different initialization strategies but they all tend to initialize weights with small values.

# 4 A deeper dive into loading data
本节介绍了两个类：Dataset 和 Dataloader
以及如何调整

![[Pasted image 20240718232159.png]]
》》 动物的名字这列不被用作 特征（输入），因为名字和目标 type 无关

![[Pasted image 20240718232329.png]]

![[Pasted image 20240718232730.png]]

## Recalling ：TensorDataset
 DataLoader() and TensorDataset() can be used to set data up for Training a PyTorch model.

TensorDataset() is a class that can be used to create a Pytorch dataset for tensors.

我们可以将上一步准备好的 X 和 Y 作为参数传递给 TensorDataset，但是要注意参数的格式是float

![[Pasted image 20240718233257.png]]
》》 我们可以用索引来访问 dataset 实例中的某个元素，这样得出的是一个tuple。也就是说，我们需要 unpack 才能访问 dataset 中的具体的内容
![[Pasted image 20240718233450.png]]
》》》 也就是是，TensorDataset（） 可以自动将 input 和 output 分隔开

## Recalling DataLoader
一旦我们使用 TensorDataset 创建了 Dataset 的实例，我们就可以将这个 dataset 实例传递给 Dataloader 类

随后我们可以定义一些能够客制化的参数，这些参数可以被作为实参传给 DataLoader

当我们迭代一个 dataloader 时，
- batch_size 决定了单次迭代中，有多少个样本被从 dataset 中提取出来
- shuffle 告知 dataloader 每次迭代将数据打散
[[#ue Writing a training loop]]
![[Pasted image 20240718234319.png]]

![[Pasted image 20240718234431.png]]
如上图，我们的数据共有 5 行，并将 batch size 设置为 2 ，也就是每次迭代，dataloader将数据集中的下面两个 sample 拿出来，不放回并打乱剩下的样本顺序。经过前两次迭代，dataloader 中还有一个样本，这最后一个样本会在第三次迭代时被取出。


### #ue  Using the TensorDataset class

In practice, loading your data into a PyTorch dataset will be one of the first steps you take in order to create and train a neural network with PyTorch.

The `TensorDataset` class is very helpful when your dataset can be loaded directly as a NumPy array. Recall that `TensorDataset()` can take one or more NumPy arrays as input.

In this exercise, you'll practice creating a PyTorch dataset using the TensorDataset class.

`torch` and `numpy` have already been imported for you, along with the `TensorDataset` class.

~~~python
import numpy as np
import torch
from torch.utils.data import TensorDataset

np_features = np.array(np.random.rand(12, 8))
np_target = np.array(np.random.rand(12, 1))

# Convert arrays to PyTorch tensors
torch_features = torch.tensor(np_features)
torch_target = torch.tensor(np_target)

# Create a TensorDataset from two tensors
dataset = TensorDataset(torch_features, torch_target) ###############

# Return the last element of this dataset
print(dataset[-1])
~~~

### #ue  From data loading to running a forward pass

In this exercise, you'll create a PyTorch `DataLoader` from a pandas DataFrame and call a model on this dataset. Specifically, you'll run a **forward pass** on a neural network. You'll continue working with fully connected neural networks, as you have done so far.

You'll begin by subsetting a loaded DataFrame called `dataframe`, converting features and targets NumPy arrays, and converting to PyTorch tensors in order to create a PyTorch dataset.

This dataset can be loaded into a PyTorch `DataLoader`, batched, shuffled, and used to run a forward pass on a custom fully connected neural network.

NumPy as `np`, pandas as `pd`, `torch`, `TensorDataset()`, and `DataLoader()` have been imported for you.

~~~python
# Load the different columns into two PyTorch tensors
#这里用 to_numpy 方法和 .values 属性都能将df转化成 array
features = torch.tensor(dataframe[['ph', 'Sulfate', 'Conductivity', 'Organic_carbon']].to_numpy()).float()
target = torch.tensor(dataframe['Potability'].to_numpy()).float()

# Create a dataset from the two generated tensors
dataset = TensorDataset(features, target)

# Create a dataloader using the above dataset
dataloader = DataLoader(dataset, shuffle=True, batch_size=2)
x, y = next(iter(dataloader))

# Create a model using the nn.Sequential API
model = nn.Sequential(
    nn.Linear(features.shape[1], 8),
    nn.Linear(8, 1)
)
output = model(features)
print(output)
~~~

## Evaluating model performance
![[Pasted image 20240719000941.png]]

![[Pasted image 20240719001015.png]]


![[Pasted image 20240719001226.png]]
》》》 最后一行 的 len 实际上就是 我们的 dataset 中 的batches 的数量

val loss 时注意特殊处理步骤，即将模型调整到 评估模式：
![[Pasted image 20240719001810.png]]
q：　train 和 val 是交替进行的吗


![[Pasted image 20240719002130.png]]
》》》 过拟合出现，当 val loss 很高但是 training loss 很低

接下来是除了 loss 之外的另一种 衡量标准：
![[Pasted image 20240719002535.png]]
如果使用了独热编码处理 类别数据，那么我们需要在 metric 函数中添加参数 argmax，从而获取类别数字，而不是独热码组成的向量。

### #ue  Writing the evaluation loop

In this exercise, you will practice writing the evaluation loop. Recall that the evaluation loop is similar to the training loop, except that you will not perform the gradient calculation and the optimizer step.

The `model` has already been defined for you, along with the object `validationloader`, which is a dataset.

~~~python
# Set the model to evaluation mode
model.eval()
validation_loss = 0.0

with torch.no_grad():
  
  for data in validationloader:
    
      outputs = model(data[0])
      loss = criterion(outputs, data[1])
      
      # Sum the current loss to the validation_loss variable
      validation_loss += loss.item()
      
# Calculate the mean loss value
validation_loss_epoch = validation_loss / len(validationloader)
print(validation_loss_epoch)

# Set the model back to training mode
model.train()
~~~
![[Pasted image 20240719002955.png]]
> Now you can ***add this validation loop after each training loop***. Looking at ground truths, model predictions, and losses together is a great way to visualize if your model is learning!


### #ue  Calculating accuracy using torchmetrics

In addition to the losses, you should also be keeping track of the accuracy during training. By doing so, you will be able to select the epoch when the model performed the best.

In this exercise, you will practice using the `torchmetrics` package to calculate the accuracy. You will be using a sample of the facemask dataset. This dataset contains three different classes. The `plot_errors` function will display samples where the model predictions do not match the ground truth. Performing such error analysis will help you understand your model failure modes.

The `torchmetrics` package is already imported. The model `outputs` are the probabilities returned by a softmax as the last step of the model. The `labels` tensor contains the labels as one-hot encoded vectors.

~~~python
# Create accuracy metric using torch metrics
metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
for data in dataloader:
    features, labels = data
    outputs = model(features)
    
    # Calculate accuracy over the batch
    acc = metric(outputs.softmax(dim=-1), labels.argmax(dim=-1))
    
# Calculate accuracy over the whole epoch
acc = metric.compute()

# Reset the metric for the next epoch 
metric.reset()
plot_errors(model, dataloader)
~~~

![[download.svg]]

## Fighting overfitting

![[Pasted image 20240719004354.png]]
![[Pasted image 20240719004427.png]]

![[Pasted image 20240719004523.png]]
p: 表示将一个神经元设为 0 的概率
![[Pasted image 20240719004607.png]]
![[Pasted image 20240719004646.png]]


![[Pasted image 20240719004758.png]]

![[Pasted image 20240719004830.png]]
![[Pasted image 20240719004845.png]]

### #ue  Experimenting with dropout

The dropout layer randomly zeroes out elements of the input tensor. Doing so helps fight overfitting. In this exercise, you'll create a small neural network with at least two linear layers, two dropout layers, and two activation functions.

The `torch.nn` package has already been imported as `nn`. An `input_tensor` of dimensions 1×3072 has been created for you.

~~~python
# Using the same model, set the dropout probability to 0.8
model = nn.Sequential(
    nn.Linear(3072, 16),
    nn.ReLU(),
    nn.Dropout(p=0.8)
)
model(input_tensor)
~~~

Adding dropout layers is a parameter-free way to fight overfitting.


## Improving model performance
![[Pasted image 20240719005622.png]]

![[Pasted image 20240719005721.png]]
![[Pasted image 20240719005802.png]]
![[Pasted image 20240719005837.png]]
![[Pasted image 20240719010001.png]]

### #ue  Implementing random search

Hyperparameter search is a computationally costly approach to experiment with different hyperparameter values. However, it can lead to performance improvements. In this exercise, you will implement a random search algorithm.

You will randomly sample 10 values of the learning rate and momentum from the uniform distribution. To do so, you will use the `np.random.uniform()` function.

The `numpy` package has already been imported as `np`.

~~~python
values = []
for idx in range(10):
    # Randomly sample a learning rate factor 2 and 4
    factor = np.random.uniform(2, 4)
    lr = 10 ** -factor
    
    # Randomly sample a momentum between 0.85 and 0.99
    momentum = np.random.uniform(0.85, 0.99)
    
    values.append((lr, momentum))
~~~





