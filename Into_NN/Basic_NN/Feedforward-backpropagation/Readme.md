## Neural Network Propagation Summary
### Overview
This provides an overview and implementation of feedforward and backward propagation in neural networks.

#### Feedforward Propagation
Feedforward propagation is the process of moving input data through the neural network to produce an output prediction. It involves the following steps:

* Input Layer: The input data is fed into the neural network through the input layer.
* Hidden Layers: The input data passes through one or more hidden layers, where computations are performed using weights and biases.
* Activation Functions: Activation functions introduce non-linearity into the model, allowing it to learn complex patterns.
* Output Layer: The processed data is finally passed through the output layer, which produces the final prediction.
#### Backward Propagation
Backward propagation, also known as backpropagation, is the process of updating the weights and biases of the neural network based on the error calculated during feedforward propagation. It involves the following steps:

* Error Calculation: The error between the predicted output and the actual output is calculated using a loss function.
* Gradient Computation: The gradients of the loss function with respect to the weights and biases are computed using the chain rule of calculus.
* Parameter Update: The weights and biases are updated in the opposite direction of the gradient using an optimization algorithm such as gradient descent.
* Iteration: Steps 1-3 are repeated iteratively for multiple epochs until the model converges to the desired accuracy.
