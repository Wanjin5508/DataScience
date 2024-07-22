# 1 Training Robust Neural Networks

![[Pasted image 20240722150120.png]]
![[Pasted image 20240722150200.png]]
![[Pasted image 20240722150239.png]]

**Dataset：**

![[Pasted image 20240722150420.png]]
>>> 注意 pytorch dataset 类，以及需要实现的具体函数


**DataLoader：**
pass the dataset to the pytorch Dataloader, and set some parameters:
![[Pasted image 20240722151041.png]]
使用 next(iter(Dataloader))
可见，批量设置为 2 的含义，就是 Dataloader 的每次迭代，都只处理 两个样本，同时注意迭代器输出的是一个 元组 ， 要 unpack

#### 两种定义模型的方式
![[Pasted image 20240722152215.png]]
》》》 注意，nn 类中，网络的层定义在属性里，激活函数写在 forward 中


### #ue  PyTorch Dataset

Time to refresh your PyTorch Datasets knowledge!

Before model training can commence, you need to load the data and pass it to the model in the right format. In PyTorch, this is handled by Datasets and DataLoaders. Let's start with building a PyTorch Dataset for our water potability data.

In this exercise, you will define a class called `WaterDataset` to load the data from a CSV file. To do this, you will need to implement the three methods which PyTorch expects a Dataset to have:

- `.__init__()` to load the data,
- `.__len__()` to return data size,
- `.__getitem()__` to extract features and label for a single sample.

The following imports that you need have already been done for you:

```
import pandas as pd
from torch.utils.data import Dataset
```

~~~python
class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        # Load data to pandas DataFrame
        df = pd.read_csv(csv_path)
        # Convert data to a NumPy array and assign to self.data
        self.data = df.to_numpy()
        
    # Implement __len__ to return the number of data samples
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        # Assign last data column to label
        label = self.data[idx, -1]
        return features, label
~~~

》》》 后面练习多注意这个数据集，应该是个分类模型


### #ue  PyTorch DataLoader

Good job defining the Dataset class! The `WaterDataset` you just created is now available for you to use.

The next step in preparing the training data is to set up a `DataLoader`. A PyTorch `DataLoader` can be created from a `Dataset` to load data, split it into batches, and perform transformations on the data if desired. Then, it yields a data sample ready for training.

In this exercise, you will build a `DataLoader` based on the `WaterDataset`. The `DataLoader` class you will need has already been imported for you from `torch.utils.data`. Let's get to it!

~~~python
# Create an instance of the WaterDataset
dataset_train = WaterDataset('water_train.csv')

# Create a DataLoader based on dataset_train
dataloader_train = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True,
)

# Get a batch of features and labels
features, labels = next(iter(dataloader_train))
print(features, labels)
~~~
![[Pasted image 20240722155652.png]]

### #ue  PyTorch Model

You will use the OOP approach to define the model architecture. Recall that this requires setting up a model class and defining two methods inside it:

- `.__init__()`, in which you define the layers you want to use;
    
- `forward()`, in which you define what happens to the model inputs once it receives them; this is where you pass inputs through pre-defined layers.
    

Let's build a model with three linear layers and ReLU activations. After the last linear layer, you need a sigmoid activation instead, which is well-suited for binary classification tasks like our water potability prediction problem. Here's the model defined using `nn.Sequential()`, which you may be more familiar with:

```
net = nn.Sequential(
  nn.Linear(9, 16),
  nn.ReLU(),
  nn.Linear(16, 8),
  nn.ReLU(),
  nn.Linear(8, 1),
  nn.Sigmoid(),
)
```

Let's rewrite this model as a class!

~~~python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the three linear layers
        self.fc1 = nn.Linear(9,16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        # Pass x through linear layers adding activations
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
~~~

## Optimizers, training, and evaluation
![[Pasted image 20240722172206.png]]
![[Pasted image 20240722172229.png]]

![[Pasted image 20240722172415.png]]

![[Pasted image 20240722172456.png]]
![[Pasted image 20240722172540.png]]
![[Pasted image 20240722172600.png]]
![[Pasted image 20240722172638.png]]
![[Pasted image 20240722172841.png]]

》》Reshape(-1, 1)  
**This operation will result in a 2D array with a shape (n, 1) , where n is the number of elements in your original array**.


### #ue  Optimizers

It's time to explore the different optimizers that you can use for training your model.

A custom function called `train_model(optimizer, net, num_epochs)` has been defined for you. It takes the optimizer, the model, and the number of epochs as inputs, runs the training loops, and prints the training loss at the end.

Let's use `train_model()` to run a few short trainings with different optimizers and compare the results!

~~~python
import torch.optim as optim

net = Net()

# Define the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_model(
    optimizer=optimizer,
    net=net,
    num_epochs=10,
)
~~~

### #ue  Model evaluation

With the training loop sorted out, you have trained the model for 1000 epochs, and it is available to you as `net`. You have also set up a `test_dataloader` in exactly the same way as you did with `train_dataloader` before—just reading the data from the test rather than the train directory.

You can now evaluate the model on test data. To do this, you will need to write the evaluation loop to iterate over the batches of test data, get the model's predictions for each batch, and calculate the accuracy score for it. Let's do it!

- Set up the evaluation metric as `Accuracy` for binary classification and assign it to `acc`.
- For each batch of test data, get the model's outputs and assign them to `outputs`.
- After the loop, compute the total test accuracy and assign it to `test_accuracy`.

~~~python
import torch
from torchmetrics import Accuracy

# Set up binary accuracy metric
acc = Accuracy(task='binary')

net.eval()
with torch.no_grad():
    for features, labels in dataloader_test:
        # Get predicted probabilities for test data batch
        outputs = net(features)
        preds = (outputs >= 0.5).float()
        acc(preds, labels.view(-1, 1))

# Compute total test accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")
~~~
![[Pasted image 20240722174053.png]]

## Vanishing and exploding gradients
![[Pasted image 20240722174150.png]]
![[Pasted image 20240722174215.png]]

![[Pasted image 20240722174247.png]]

![[Pasted image 20240722174305.png]]
![[Pasted image 20240722174348.png]]
![[Pasted image 20240722174441.png]]

![[Pasted image 20240722174523.png]]
把模型的 layer 的 weight 属性作为参数

![[Pasted image 20240722174645.png]]

![[Pasted image 20240722174737.png]]
》》 如果神经元的输入是非正数，那么神经元就死了

![[Pasted image 20240722174835.png]]
![[Pasted image 20240722174932.png]]

![[Pasted image 20240722175005.png]]
![[Pasted image 20240722175049.png]]

### #ue  Initialization and activation

The problems of unstable (vanishing or exploding) gradients are a challenge that often arises in training deep neural networks. In this and the following exercises, you will expand the model architecture that you built for the water potability classification task to make it more immune to those problems.

As a first step, you'll improve the weights initialization by using He (Kaiming) initialization strategy. To do so, you will need to call the proper initializer from the `torch.nn.init` module, which has been imported for you as `init`. Next, you will update the activations functions from the default ReLU to the often better ELU.

~~~python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
        # Apply He initialization
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='sigmoid')

    def forward(self, x):
        # Update ReLU activation to ELU
        x = nn.functional.elu(self.fc1(x))
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
~~~

### #ue  Batch Normalization

As a final improvement to the model architecture, let's add the batch normalization layer after each of the two linear layers. The batch norm trick tends to accelerate training convergence and protects the model from vanishing and exploding gradients issues.

Both `torch.nn` and `torch.nn.init` have already been imported for you as `nn` and `init`, respectively. Once you implement the change in the model architecture, be ready to answer a short question on how batch normalization works!

~~~python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        # Add two batch normalization layers
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)
        
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity="sigmoid") 
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)

        # Pass x through the second set of layers
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)

        x = nn.functional.sigmoid(self.fc3(x))
        return x
~~~









# 4 Multi-Input & Multi-Output Architectures
> Build multi-input and multi-output models, demonstrating how they can handle tasks requiring more than one input or generating multiple outputs. You will explore how to design and train these models using PyTorch and delve into the crucial topic of loss weighting in multi-output models. This involves understanding **how to balance** the importance of different tasks when training a model to perform multiple tasks simultaneously.

![[Pasted image 20240722180409.png]]

![[Pasted image 20240722180442.png]]

现在使用上述数据来建立一个 2-input 模型，来对手写字母进行分类
- 第一个输入是字母的 图片 ，例如拉丁字母 K
- 第二个输入是该字母源于哪个字母表，以独热码的形式给出
![[Pasted image 20240722180752.png]]
这两个输入被分开处理，然后将对应的 representations 链接起来：
![[Pasted image 20240722221759.png]]

最终模型的输出是 一个 类别：
![[Pasted image 20240722221839.png]]

要想实现这样一个***多输入***的模型，我们需要两个要素：
1. 一个 custom Dataset
2. 一个合适的模型架构

![[Pasted image 20240722222115.png]]
客制化的 Dataset 类，即继承Torch提供的 Dataset 类：
需要 transform 和 sample 属性，其中sample是一个列表，其中的元素是 tuple，每个tuple包含三个元素，分别是 1）文件地址，2）字母表，3）标签

![[Pasted image 20240722222344.png]]


![[Pasted image 20240722222433.png]]
![[Pasted image 20240722222450.png]]
0 表示增加行，1 表示增加列

接下来确定模型架构：这里由两部分组成
![[Pasted image 20240722222703.png]]

![[Pasted image 20240722222746.png]]
》》》 classifier 属性即：将上面两个input 叠加起来
![[Pasted image 20240722222844.png]]

在 forward 方法中，需要将两个 inputs ***分开***传入。
![[Pasted image 20240722223032.png]]



![[Pasted image 20240722223156.png]]
要搞清楚我们自定义的 DataLoader 中包含什么元素

### #ue  Two-input dataset

Building a multi-input model starts with crafting a custom dataset that can supply all the inputs to the model. In this exercise, you will build the Omniglot dataset that serves triplets consisting of:

- The image of a character to be classified,
- The one-hot encoded alphabet vector of length 30, with zeros everywhere but for a single one denoting the ID of the alphabet the character comes from,
- The target label, an integer between 0 and 963.

You are provided with `train_samples`, a list of 3-tuples comprising an image's file path, its alphabet vector, and the target label. Also, the following imports have already been done for you, so let's get to it!

```
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
```

~~~python
class OmniglotDataset(Dataset):
    def __init__(self, transform, samples):
		# Assign transform and samples to class attributes
        self.transform = transform
        self.samples = samples
                    
    def __len__(self):
		# Return number of samples
        return len(self.samples)

    def __getitem__(self, idx):
      	# Unpack the sample at index idx
        img_path, alphabet, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        # Transform the image 
        img_transformed = self.transform(img)
        return img_transformed, alphabet, label
~~~

### #ue Two-input model

With the data ready, it's time to build the two-input model architecture! To do so, you will set up a model class with the following methods:

- `.__init__()`, in which you will define sub-networks by grouping layers; this is where you define the two layers for processing the two inputs, and the ***classifier that returns a classification score for each class.***
- `forward()`, in which you will pass both inputs through corresponding pre-defined sub-networks, concatenate the outputs, and pass them to the classifier.
    
`torch.nn` is already imported for you as `nn`. Let's do it!

~~~python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define sub-networks as sequential models
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128)
        )
        self.alphabet_layer = nn.Sequential(
            nn.Linear(30, 8),
            nn.ELU(), 
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 8, 964), 
        )
        
    def forward(self, x_image, x_alphabet):
		# Pass the x_image and x_alphabet through appropriate layers
        x_image = self.image_layer(x_image)
        x_alphabet = self.alphabet_layer(x_alphabet)
        # Concatenate x_image and x_alphabet
        x = torch.cat((x_image, x_alphabet), dim=1)
        return self.classifier(x)
~~~

## Multi-output models
如果我们需要使用同样的 input 来预测多个目标，那么就需要~

在 multi-label classification 问题中，输入可以同时属于**多个类别**
![[Pasted image 20240722225123.png]]
》》 最下方：在非常深的模型中，Blocks 由 layers 组成。通常会在每个blockk后添加额外的输出来预测相同的目标。这样的目的在于，模型前面的部分能够学习到对任务有利的特征。通常的形式是 regularization，以便促进模型的鲁棒性

![[Pasted image 20240722225516.png]]
![[Pasted image 20240722225548.png]]
![[Pasted image 20240722225639.png]]
![[Pasted image 20240722225758.png]]

![[Pasted image 20240722230116.png]]
这里的损失函数只是简单将两个输出损失相加，相当于每个输出的权重为50%。如果我们不需要将两种输出看作同样重要，那么每个输出对应的损失可以被赋予相应的权重。

### #ue  Two-output Dataset and DataLoader

In this and the following exercises, you will build a two-output model to predict both the character and the alphabet it comes from based on the character's image. As always, you will start with getting the data ready.

The `OmniglotDataset` class you have created before is available for you to use along with updated `samples`. Let's use it to build the Dataset and the DataLoader.

The following imports have already been done for you:

```
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
```

~~~python
# Print the sample at index 100
print(samples[100])

# Create dataset_train
dataset_train = OmniglotDataset(
    transform=transforms.Compose([
        transforms.ToTensor(), ######################查！！！
      	transforms.Resize((64, 64)),
    ]),
    samples=samples,
)

# Create dataloader_train
dataloader_train = DataLoader(
    dataset_train, shuffle=True, batch_size=32,
)
~~~

### #ue  Two-output model architecture

In this exercise, you will construct a multi-output neural network architecture capable of predicting the character and the alphabet.

Recall the general structure: in the `.__init__()` method, you ***define layers*** to be used in the forward pass later. In the `forward()` method, you will first pass the input image through a couple of layers to obtain its ***embedding***, which in turn is fed into two separate classifier layers, one for each output.

`torch.nn` is already imported under its usual alias, so let's build a model!

~~~python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128)
        )
        # Define the two classifier layers
        self.classifier_alpha = nn.Linear(128, 30)
        self.classifier_char = nn.Linear(128, 964)
        
    def forward(self, x):
        x_image = self.image_layer(x) #image embedding
        # Pass x_image through the classifiers and return both results
        output_alpha = self.classifier_alpha(x_image)
        output_char = self.classifier_char(x_image)
        return output_alpha, output_char
~~~

### #ue  Training multi-output models

When training models with multiple outputs, it is crucial to ensure that the loss function is defined correctly.

In this case, the model produces two outputs: predictions for the alphabet and the character. For each of these, there are corresponding ground truth labels, which will allow you to calculate two separate losses: one incurred from incorrect alphabet classifications, and the other from incorrect character classification. Since in both cases you are dealing with a multi-label classification task, the Cross-Entropy loss can be applied each time.

Gradient descent can optimize only one loss function, however. You will thus define the total loss as the sum of alphabet and character losses.

- Calculate the alphabet classification loss and assign it to `loss_alpha`.
- Calculate the character classification loss and assign it to `loss_char`.
- Compute the total loss as the sum of the two partial losses and assign it to `loss`.

~~~python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05)

for epoch in range(1):
    for images, labels_alpha, labels_char in dataloader_train:
        optimizer.zero_grad()
        outputs_alpha, outputs_char = net(images)
        # Compute alphabet classification loss
        loss_alpha = criterion(
            outputs_alpha, labels_alpha
        )
        # Compute character classification loss
        loss_char = criterion(outputs_char, labels_char)
        # Compute total loss
        loss = loss_alpha + loss_char
        loss.backward()
        optimizer.step()
~~~















