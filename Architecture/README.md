# Neural Network Basics: Introduction Using PyTorch, Keras, and TensorFlow

This repository provides a simple introduction to building basic neural network architectures using PyTorch, Keras, and TensorFlow. It covers the fundamental concepts of neurons, model structures, and training, with code examples in each framework.

## Table of Contents
- [Introduction](#introduction)
- [Neural Network Structure](#neural-network-structure)
- [Backpropagation and Losses](#backpropagation-and-losses)
- [Optimizers](#optimizers)
- [Framework Implementations](#framework-implementations)
- [Loss Function and Evaluation](#loss-function-and-evaluation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Neural networks are essential for modern machine learning, simulating the way the human brain processes information. This project demonstrates how to build and train simple feedforward neural networks using three popular frameworks: PyTorch, Keras, and TensorFlow.

## Neural Network Structure
### 1. Building Blocks: Neurons
A neuron performs basic computations involving:
- **Weighted sum of inputs**.
- **Bias addition**.
- **Activation function** to produce the output.

### 2. Combining Neurons into a Network
Neurons are connected to form different layers:
- **Input layer**: Receives raw input data.
- **Hidden layer(s)**: Process data using connected neurons.
- **Output layer**: Produces the final prediction.

## Backpropagation and Losses
Backpropagation is the core algorithm for training neural networks. It computes the gradient of the loss function with respect to each weight in the network, allowing the network to adjust the weights and minimize the loss during training.

### What is Backpropagation?
Backpropagation involves two main steps:
1. **Forward Pass**: Compute the output of the network given the input and calculate the loss.
2. **Backward Pass**: Compute the gradients of the loss function with respect to the network’s parameters using the chain rule. This step updates the weights in the direction that reduces the loss.

### Loss Functions
Loss functions measure how well the predictions of the network match the actual targets. Here are some common loss functions:

- **Mean Squared Error (MSE)**:
  Used for regression tasks, MSE calculates the average squared difference between the predicted and actual values.
  ```python
  def mse_loss(y_true, y_pred):
      return ((y_true - y_pred) ** 2).mean()


## Optimizers
Optimizers play a crucial role in training neural networks. They update the model's weights based on the gradients computed during backpropagation and control how the network learns over time. Different optimizers can affect the training speed and the quality of the final model.

### Common Optimizers
Below are some popular optimizers used in neural network training:

1. **Stochastic Gradient Descent (SGD)**
   - **Description**: Updates weights using the gradient of the loss function. It takes a small step in the direction that decreases the loss.
   - **Pros**: Simple and effective for smaller problems.
   - **Cons**: May converge slowly and get stuck in local minima.
   ```python
   optimizer = optim.SGD(model.parameters(), lr=0.01)
   ```

2. **Adam**
   - **Description**: Combines the advantages of AdaGrad and RMSProp. It adapts the learning rate for each parameter and maintains separate learning rates for each parameter.
   - **Pros**: Converges quickly and works well for a wide range of problems.
   - **Cons**: May require tuning of hyperparameters.
   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   ```

## Framework Implementations

### 1. PyTorch
PyTorch is a popular deep learning framework known for its flexibility and dynamic computation graphs. It provides a rich set of tools for building and training neural networks.

#### Example: Building a Simple Neural Network in PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

### 2. Keras
Keras is a high-level neural networks API that runs on top of TensorFlow, CNTK, or Theano. It provides a user-friendly interface for building and training neural networks.

#### Example: Building a Simple Neural Network in Keras
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
```

### 3. TensorFlow
TensorFlow is an open-source deep learning library developed by Google. It provides a comprehensive ecosystem of tools, libraries, and community resources for building and deploying machine learning models.

#### Example: Building a Simple Neural Network in TensorFlow
```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Loss Function and Evaluation
### Loss Function
The loss function measures the difference between the predicted output and the actual target. It guides the optimization process by providing feedback on how well the model is performing.

#### Example: Mean Squared Error Loss
```python
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
```

### Evaluation Metrics
Evaluation metrics help assess the performance of the model on unseen data. Common metrics include accuracy, precision, recall, and F1 score, depending on the task.

#### Example: Accuracy Calculation
```python
def accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
```

## Getting Started
To get started with this project, you need to have Python installed on your system. You can install the required dependencies using the following command:
```bash
pip install torch tensorflow keras
```

## Usage
Run the provided code examples to experiment with building and training neural networks in PyTorch, Keras, and TensorFlow.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
```