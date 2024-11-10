# Comprehensive PyTorch CNN for CIFAR-10

# 1. Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm  # For progress bars

# 2. Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors (C x H x W)
    transforms.Normalize((0.5, 0.5, 0.5),  # Normalize each channel (R, G, B) to mean=0.5
                         (0.5, 0.5, 0.5))  # and standard deviation=0.5
])

# 3. Load the CIFAR-10 training and testing datasets
trainset = torchvision.datasets.CIFAR10(
    root='./data',  # Directory to store data
    train=True,  # Indicates training set
    download=True,  # Download if not present
    transform=transform  # Apply transformations
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,  # Number of samples per batch
    shuffle=True,  # Shuffle data for training
    num_workers=2  # Number of subprocesses for data loading
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,  # Indicates testing set
    download=True,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,  # Consistent batch size for evaluation
    shuffle=False,  # No shuffling for evaluation
    num_workers=2
)


# 4. Define the CNN model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3,  # Input channels (RGB)
            out_channels=32,  # Number of filters/kernels
            kernel_size=3,  # 3x3 filters
            stride=1,  # Move 1 pixel at a time
            padding=1  # Add 1 pixel padding to preserve dimensions
        )
        # Pooling Layer 1: Max Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Reduces dimensions by half

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32,  # Input channels from conv1
            out_channels=64,  # Number of filters
            kernel_size=3,  # 3x3 filters
            stride=1,  # Move 1 pixel at a time
            padding=1  # Preserve dimensions
        )
        # Pooling Layer 2: Average Pooling
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # Further reduces dimensions by half

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=64,  # Input channels from conv2
            out_channels=64,  # Number of filters
            kernel_size=3,  # 3x3 filters
            stride=1,  # Move 1 pixel at a time
            padding=1  # Preserve dimensions
        )
        # Pooling Layer 3: Max Pooling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Final downsampling

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # 64 channels * 4 height * 4 width = 1024 input features

        # Output Layer
        self.fc2 = nn.Linear(64, 10)  # 64 input features to 10 classes

        # Activation Function
        self.relu = nn.ReLU()  # ReLU activation

    def forward(self, x):
        # Pass through Convolutional Layer 1 -> ReLU -> Pooling Layer 1
        x = self.pool1(self.relu(self.conv1(x)))  # Output shape: [batch_size, 32, 16, 16]

        # Pass through Convolutional Layer 2 -> ReLU -> Pooling Layer 2
        x = self.pool2(self.relu(self.conv2(x)))  # Output shape: [batch_size, 64, 8, 8]

        # Pass through Convolutional Layer 3 -> ReLU -> Pooling Layer 3
        x = self.pool3(self.relu(self.conv3(x)))  # Output shape: [batch_size, 64, 4, 4]

        # Flatten the tensor into a vector for Fully Connected Layers
        x = x.view(-1, 64 * 4 * 4)  # Reshape to [batch_size, 1024]

        # Pass through Fully Connected Layer 1 -> ReLU
        x = self.relu(self.fc1(x))  # Output shape: [batch_size, 64]

        # Pass through Output Layer
        x = self.fc2(x)  # Output shape: [batch_size, 10]

        return x


# 5. Instantiate the model, define loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available
model = CNN().to(device)  # Move model to the selected device

criterion = nn.CrossEntropyLoss()  # Define loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define optimizer with learning rate

# 6. Training Loop
num_epochs = 10  # Number of times to iterate over the entire dataset

for epoch in range(num_epochs):
    running_loss = 0.0  # Initialize running loss for the epoch
    model.train()  # Set model to training mode (enables dropout, batch norm, etc.)

    # Use tqdm to create a progress bar for the training loop
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

        optimizer.zero_grad()  # Zero the gradients (clears old gradients)

        outputs = model(inputs)  # Forward pass: compute predicted outputs
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass: compute gradient of the loss w.r.t. model parameters
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss for the epoch

    # Calculate and print average loss for the epoch
    avg_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

print('Finished Training')  # Indicate end of training

# 7. Evaluation on Test Data
model.eval()  # Set model to evaluation mode (disables dropout, batch norm, etc.)
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for evaluation
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # Move data to device

        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability

        total += labels.size(0)  # Increment total samples
        correct += (predicted == labels).sum().item()  # Increment correct predictions

test_accuracy = 100 * correct / total  # Calculate accuracy
print(f"Test Accuracy: {test_accuracy:.2f}%")  # Print test accuracy

# 8. Saving the Model Checkpoint
checkpoint_path = 'cnn_checkpoint.pth'  # Define checkpoint path
torch.save(model.state_dict(), checkpoint_path)  # Save model's state dictionary
print("Model checkpoint saved.")  # Confirmation message

# 9. Loading the Model Checkpoint (Optional)
# To load the model later:
# model = CNN()  # Initialize the model architecture
# model.load_state_dict(torch.load(checkpoint_path))  # Load saved parameters
# model.to(device)  # Move to device
# model.eval()  # Set to evaluation mode
# print("Model checkpoint loaded.")  # Confirmation message
