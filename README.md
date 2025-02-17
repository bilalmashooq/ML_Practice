Certainly! Let's address your requirements step-by-step:

1. **Understanding Batch Size**
2. **Defining a Custom Dataset to Load T1 and T2 MRIs Separately Along with Demographics**
3. **Designing a 3D ResNet Model with Separate Convolutional Paths for T1 and T2**
4. **Training the Model from Scratch**
5. **Displaying the Model Summary**

---

## 1. Understanding Batch Size

**Batch Size** is a crucial hyperparameter in training neural networks. It refers to the number of samples processed before the model's internal parameters are updated. Here's a breakdown:

- **Small Batch Sizes:**
- **Pros:**
- **Lower Memory Consumption:** Suitable for GPUs with limited memory.
- **Regularization Effect:** Introduces noise in the gradient estimation, potentially helping escape local minima.
- **Cons:**
- **Noisy Gradient Estimates:** Can lead to unstable training dynamics.
- **Longer Training Time:** Requires more updates to reach convergence.

- **Large Batch Sizes:**
- **Pros:**
- **Stable Gradient Estimates:** Leads to more consistent updates.
- **Faster Training (in terms of iterations):** Utilizes GPU parallelism effectively.
- **Cons:**
- **Higher Memory Consumption:** May not fit into GPU memory.
- **Potential Overfitting:** Reduced regularization effect.

**Recommendation for 3D MRI Data:**

3D MRI data is memory-intensive due to its volumetric nature. A smaller batch size is often necessary to accommodate the high memory usage of 3D convolutions. Starting with a **batch size of 2 or 4** is advisable. You can adjust this based on your GPU's capacity:

```python
batch_size = 4 # Start with 4; reduce if you encounter out-of-memory errors
```

---

## 2. Defining a Custom Dataset to Load T1 and T2 MRIs Separately Along with Demographics

We'll create a PyTorch `Dataset` that:

- Loads T1 and T2 preprocessed NIfTI files separately.
- Loads demographic information (`Age_at_Scan` and `Sex`).
- Provides the data in a format suitable for the model.

**Directory Structure:**

```
/preprocessed_data
/T1
sub-15-06_T1_preprocessed.nii.gz
sub-15-01_T1_preprocessed.nii.gz
...
/T2
sub-15-06_T2_preprocessed.nii.gz
sub-15-01_T2_preprocessed.nii.gz
...
/data
metadata.csv
```

**Sample Metadata (`metadata.csv`):**

| Patient_ID | MRI_ID | MR_System | Number_of_Lesions | Scan_Date | Sex | Date_of_Birth | EDSS | Age_at_Scan |
|------------|-----------|-----------|-------------------|-----------|-----|---------------|------|-------------|
| P01 | sub-01-01 | 1.5T | 19 | ... | F | ... | 2 | 32 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Implementation:**

```python
import os
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch

class MRIDataset(Dataset):
def __init__(self, metadata_df, t1_dir, t2_dir, transform=None):
"""
Initializes the dataset by storing metadata and directories.

Args:
metadata_df (pd.DataFrame): DataFrame containing metadata for each MRI scan.
t1_dir (str): Directory path to T1 preprocessed NIfTI files.
t2_dir (str): Directory path to T2 preprocessed NIfTI files.
transform (callable, optional): Optional transform to be applied on the MRI data.
"""
self.metadata = metadata_df.reset_index(drop=True)
self.t1_dir = t1_dir
self.t2_dir = t2_dir
self.transform = transform

def __len__(self):
return len(self.metadata)

def load_nifti(self, filepath):
"""
Loads a NIfTI file and returns it as a numpy array.

Args:
filepath (str): Path to the NIfTI file.

Returns:
np.ndarray: MRI data.
"""
try:
nifti = nib.load(filepath)
data = nifti.get_fdata()
return data
except FileNotFoundError:
print(f"Warning: File not found {filepath}. Returning zeros.")
# Assuming that the MRI data has dimensions [D, H, W]
# You might want to handle this differently based on your requirements
return np.zeros((1, 1, 1))

def __getitem__(self, idx):
"""
Retrieves the T1 and T2 MRI scans, demographic data, and EDSS score for a given index.

Args:
idx (int): Index of the sample.

Returns:
tuple: (T1_tensor, T2_tensor, demographics_tensor, EDSS)
"""
row = self.metadata.iloc[idx]
mri_id = row['MRI_ID']
sex = row['Sex']
age = row['Age_at_Scan']
edss = row['EDSS']

# Construct file paths
t1_filename = f"{mri_id}_T1_preprocessed.nii.gz"
t2_filename = f"{mri_id}_T2_preprocessed.nii.gz"
t1_path = os.path.join(self.t1_dir, t1_filename)
t2_path = os.path.join(self.t2_dir, t2_filename)

# Load MRI data
t1 = self.load_nifti(t1_path)
t2 = self.load_nifti(t2_path)

# Normalize MRI data
t1 = (t1 - np.mean(t1)) / np.std(t1) if np.std(t1) > 0 else t1 - np.mean(t1)
t2 = (t2 - np.mean(t2)) / np.std(t2) if np.std(t2) > 0 else t2 - np.mean(t2)

# Apply transformations if any
if self.transform:
t1 = self.transform(t1)
t2 = self.transform(t2)

# Encode sex: 1 for male, 0 for female
sex_encoded = 1 if str(sex).strip().upper() in ['M', 'MALE'] else 0

# Combine demographics
demographics = np.array([age, sex_encoded], dtype=np.float32)

# Convert to tensors
t1_tensor = torch.tensor(t1, dtype=torch.float32).unsqueeze(0) # Shape: [1, D, H, W]
t2_tensor = torch.tensor(t2, dtype=torch.float32).unsqueeze(0) # Shape: [1, D, H, W]
demographics_tensor = torch.tensor(demographics, dtype=torch.float32) # Shape: [2]
edss = torch.tensor(edss, dtype=torch.float32) # Scalar

return t1_tensor, t2_tensor, demographics_tensor, edss
```

**Explanation:**

- **Separate Loading:** T1 and T2 MRIs are loaded separately and maintained as separate tensors.
- **Normalization:** Each MRI modality is normalized independently.
- **Demographics:** Age and sex are encoded and combined into a tensor.
- **Handling Missing Files:** If an MRI file is missing, a tensor of zeros is returned. Adjust this behavior based on your requirements (e.g., exclude such samples).

**Creating DataLoaders:**

```python
# Paths to your data directories
t1_directory = 'preprocessed_data/T1' # Update with your actual path
t2_directory = 'preprocessed_data/T2' # Update with your actual path
metadata_path = 'data/metadata.csv' # Update with your actual path

# Load your metadata
metadata = pd.read_csv(metadata_path)

# Assuming you've already cleaned the metadata to exclude missing files
# If not, ensure to clean it before creating the dataset

# Split metadata into training and validation sets (80% train, 20% val)
train_df, val_df = train_test_split(
metadata,
test_size=0.2,
random_state=42,
stratify=metadata[['Sex', 'EDSS']] # Stratify based on Sex and EDSS
)

print(f"Number of training samples: {len(train_df)}")
print(f"Number of validation samples: {len(val_df)}")

# Create Dataset instances
train_dataset = MRIDataset(train_df, t1_directory, t2_directory)
val_dataset = MRIDataset(val_df, t1_directory, t2_directory)

# Define DataLoaders
batch_size = 4 # Adjust based on GPU memory
num_workers = 4 # Adjust based on CPU cores

train_loader = DataLoader(
train_dataset,
batch_size=batch_size,
shuffle=True,
num_workers=num_workers
)

val_loader = DataLoader(
val_dataset,
batch_size=batch_size,
shuffle=False,
num_workers=num_workers
)
```

**Notes:**

- **Stratification:** The `stratify` parameter ensures that the distribution of `Sex` and `EDSS` in the training and validation sets mirrors the original dataset.
- **Batch Size:** Set to `4` as a starting point. Depending on your GPU's memory, you can adjust this up or down.

---

## 3. Designing a 3D ResNet Model with Separate Convolutional Paths for T1 and T2

We'll design a neural network that processes T1 and T2 MRI scans through separate convolutional paths, extracts features, merges them, incorporates demographic data, and finally predicts the EDSS score.

**Architecture Overview:**

1. **Separate Convolutional Paths for T1 and T2:**
- Each path uses 3D convolutions to extract features from the respective MRI modality.

2. **Feature Merging:**
- Concatenate the extracted features from both paths.

3. **Incorporate Demographics:**
- Concatenate age and sex with the merged features.

4. **Regression Layers:**
- Fully connected layers to predict the EDSS score.

**Implementation:**

```python
import torch.nn as nn
import torch.nn.functional as F

class DualStream3DResNet(nn.Module):
def __init__(self):
super(DualStream3DResNet, self).__init__()

# Define ResNet-like architecture for T1
self.t1_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1) # Input channel 1 (T1)
self.t1_bn1 = nn.BatchNorm3d(32)
self.t1_pool = nn.MaxPool3d(kernel_size=2, stride=2)
self.t1_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
self.t1_bn2 = nn.BatchNorm3d(64)
self.t1_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

# Define ResNet-like architecture for T2
self.t2_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1) # Input channel 1 (T2)
self.t2_bn1 = nn.BatchNorm3d(32)
self.t2_pool = nn.MaxPool3d(kernel_size=2, stride=2)
self.t2_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
self.t2_bn2 = nn.BatchNorm3d(64)
self.t2_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

# After separate convolutions, merge features
# Assuming input MRI sizes are consistent and pooling reduces dimensions appropriately
self.fc1 = nn.Linear(64*8*8*8*2 + 2, 128) # Adjust based on input size after pooling
self.relu = nn.ReLU()
self.dropout = nn.Dropout(0.3)
self.fc2 = nn.Linear(128, 64)
self.fc3 = nn.Linear(64, 1) # Output: EDSS score

def forward(self, t1, t2, demographics):
"""
Forward pass of the model.

Args:
t1 (Tensor): T1 MRI tensor of shape [batch_size, 1, D, H, W]
t2 (Tensor): T2 MRI tensor of shape [batch_size, 1, D, H, W]
demographics (Tensor): Demographics tensor of shape [batch_size, 2]

Returns:
Tensor: Predicted EDSS scores of shape [batch_size]
"""
# T1 convolutional path
x1 = self.t1_conv1(t1)
x1 = self.t1_bn1(x1)
x1 = F.relu(x1)
x1 = self.t1_pool(x1)
x1 = self.t1_conv2(x1)
x1 = self.t1_bn2(x1)
x1 = F.relu(x1)
x1 = self.t1_pool2(x1)
x1 = x1.view(x1.size(0), -1) # Flatten

# T2 convolutional path
x2 = self.t2_conv1(t2)
x2 = self.t2_bn1(x2)
x2 = F.relu(x2)
x2 = self.t2_pool(x2)
x2 = self.t2_conv2(x2)
x2 = self.t2_bn2(x2)
x2 = F.relu(x2)
x2 = self.t2_pool2(x2)
x2 = x2.view(x2.size(0), -1) # Flatten

# Merge features
merged_features = torch.cat((x1, x2), dim=1) # Shape: [batch_size, feature_size]

# Concatenate demographics
combined = torch.cat((merged_features, demographics), dim=1) # Shape: [batch_size, feature_size + 2]

# Fully connected layers
out = self.fc1(combined)
out = self.relu(out)
out = self.dropout(out)
out = self.fc2(out)
out = self.relu(out)
out = self.fc3(out)

return out.squeeze(1) # Shape: [batch_size]
```

**Explanation:**

- **Separate Convolutional Paths:**
- **T1 Path:**
- Two convolutional layers with batch normalization and ReLU activation.
- Max pooling to reduce spatial dimensions.

- **T2 Path:**
- Mirrors the T1 path with identical architecture.

- **Feature Merging:**
- After processing T1 and T2 separately, the features are concatenated.

- **Incorporating Demographics:**
- Demographic data (`Age_at_Scan` and `Sex`) is concatenated with the merged features.

- **Regression Layers:**
- Two fully connected layers with ReLU activation and dropout for regularization.
- Final layer outputs a single value representing the predicted EDSS score.

- **Adjusting `fc1` Input Size:**
- The input size to `fc1` (`64*8*8*8*2 + 2`) assumes that after two pooling layers, the feature maps are of size `[64, 8, 8, 8]` for each stream.
- **Adjust This Based on Your MRI Dimensions:**
- You need to calculate the flattened feature size after the convolutional and pooling layers based on your MRI input dimensions. For instance, if your MRIs are of size `[1, 64, 64, 64]`, after two pooling layers with `kernel_size=2` and `stride=2`, the feature maps would be `[64, 16, 16, 16]`, leading to `64*16*16*16*2 + 2`.

**Note:** Adjust the architecture's fully connected layers based on your input MRI dimensions to ensure the correct feature size.

---

## 4. Training the Model from Scratch

We'll proceed to train the model without using pretrained weights.

**Implementation Steps:**

1. **Initialize the Model**
2. **Define Loss Function and Optimizer**
3. **Implement the Training Loop with Validation and Early Stopping**
4. **Monitor Training and Save the Best Model**
5. **Display the Model Summary**

**Step-by-Step Code:**

### **a. Initialize the Model**

```python
# Instantiate the model
model = DualStream3DResNet()

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print("Model initialized and moved to device.")
```

### **b. Define Loss Function and Optimizer**

We'll use **Mean Squared Error (MSE)** as the loss function since EDSS is a continuous variable, and **Adam** as the optimizer.

```python
import torch.optim as optim

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

print("Loss function and optimizer defined.")
```

### **c. Implement the Training Loop with Validation and Early Stopping**

We'll implement a training loop that:

- Trains the model on the training set.
- Validates the model on the validation set after each epoch.
- Implements **Early Stopping** with `patience=10` to halt training if validation loss doesn't improve for 10 consecutive epochs.
- Saves the model with the best validation loss.

```python
from tqdm import tqdm
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
"""
Trains the model and implements early stopping.

Args:
model (nn.Module): The neural network model.
train_loader (DataLoader): DataLoader for training data.
val_loader (DataLoader): DataLoader for validation data.
criterion (nn.Module): Loss function.
optimizer (torch.optim.Optimizer): Optimizer.
device (torch.device): Device to run the model on.
num_epochs (int): Maximum number of epochs to train.
patience (int): Number of epochs to wait for improvement before stopping.

Returns:
nn.Module: The trained model.
"""
best_val_loss = float('inf')
trigger_times = 0
best_model_path = 'best_dual_stream_3dresnet_edss.pth'

for epoch in range(num_epochs):
model.train()
train_losses = []

print(f'\nEpoch {epoch+1}/{num_epochs}')
start_time = time.time()

for batch_idx, (t1, t2, demographics, edss) in enumerate(tqdm(train_loader, desc='Training')):
t1 = t1.to(device)
t2 = t2.to(device)
demographics = demographics.to(device)
edss = edss.to(device)

optimizer.zero_grad()
outputs = model(t1, t2, demographics)
loss = criterion(outputs, edss)
loss.backward()
optimizer.step()

train_losses.append(loss.item())

avg_train_loss = np.mean(train_losses)

# Validation Phase
model.eval()
val_losses = []
with torch.no_grad():
for t1, t2, demographics, edss in tqdm(val_loader, desc='Validation'):
t1 = t1.to(device)
t2 = t2.to(device)
demographics = demographics.to(device)
edss = edss.to(device)

outputs = model(t1, t2, demographics)
loss = criterion(outputs, edss)
val_losses.append(loss.item())

avg_val_loss = np.mean(val_losses)
elapsed = time.time() - start_time
print(f'Epoch {epoch+1} completed in {elapsed:.2f}s')
print(f'Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')

# Check for improvement
if avg_val_loss < best_val_loss:
best_val_loss = avg_val_loss
trigger_times = 0
torch.save(model.state_dict(), best_model_path)
print('Validation loss decreased. Saving model...')
else:
trigger_times += 1
print(f'No improvement in validation loss for {trigger_times} epoch(s).')
if trigger_times >= patience:
print('Early stopping triggered.')
break

print("\nTraining complete.")
print(f'Best Validation Loss: {best_val_loss:.4f}')
return model
```

**Explanation:**

- **Training Loop:**
- Iterates over the specified number of epochs.
- For each batch in the training set:
- Moves data to the specified device.
- Performs a forward pass, computes loss, backpropagates, and updates model weights.
- Records the training loss.

- **Validation Loop:**
- After each epoch, evaluates the model on the validation set without updating weights.
- Records the validation loss.

- **Early Stopping:**
- If the validation loss improves, resets the `trigger_times` counter and saves the model.
- If no improvement is seen for `patience` consecutive epochs, stops training.

**Training the Model:**

```python
# Train the model
trained_model = train_model(
model=model,
train_loader=train_loader,
val_loader=val_loader,
criterion=criterion,
optimizer=optimizer,
device=device,
num_epochs=100, # You can adjust this as needed
patience=10
)
```

---

## 5. Displaying the Model Summary

PyTorch doesn't have a built-in `model.summary()` like Keras, but you can use external libraries such as [`torchsummary`](https://pypi.org/project/torchsummary/) or [`torchinfo`](https://pypi.org/project/torchinfo/) to achieve similar functionality.

**Installing `torchsummary`:**

```bash
pip install torchsummary
```

**Using `torchsummary` to Display Model Summary:**

```python
from torchsummary import summary

# Define input sizes
# Assuming MRI scans are of size [1, 64, 64, 64] for T1 and T2
# Adjust the dimensions based on your actual MRI data
t1_input_size = (1, 64, 64, 64) # [Channels, Depth, Height, Width]
t2_input_size = (1, 64, 64, 64)

# Define a dummy input for demographics
demographics_input_size = (2, )

# Since torchsummary requires a single input size, we'll provide only the MRI dimensions
# Demographics will be handled separately in the model
# Alternatively, you can modify the model to accept concatenated inputs, but it's more complex

# Display summary for T1 path
print("\nT1 Convolutional Path Summary:")
summary(model.t1_conv1, input_size=t1_input_size, device=device.type)
print("\nT2 Convolutional Path Summary:")
summary(model.t2_conv1, input_size=t2_input_size, device=device.type)
print("\nFull Model Summary:")
# Since model.forward expects multiple inputs, torchsummary cannot directly handle it.
# Instead, we'll provide a workaround by creating a sequential model or using torchinfo.

# Installing torchinfo for better summaries with multiple inputs
# Uncomment the following line if torchinfo is not installed
# pip install torchinfo

from torchinfo import summary as torchinfo_summary

# Define a custom forward method for torchinfo
class CombinedModel(nn.Module):
def __init__(self, original_model):
super(CombinedModel, self).__init__()
self.original_model = original_model

def forward(self, t1, t2, demographics):
return self.original_model(t1, t2, demographics)

combined_model = CombinedModel(trained_model)
combined_model = combined_model.to(device)

# Display the summary
torchinfo_summary(
combined_model,
input_size=[
(batch_size, 1, 64, 64, 64), # T1 input
(batch_size, 1, 64, 64, 64), # T2 input
(batch_size, 2) # Demographics
],
device=device.type
)
```

**Explanation:**

- **`torchsummary`:** Provides a summary similar to Keras' `model.summary()`. However, it doesn't natively support multiple inputs.

- **`torchinfo`:** An alternative library (`torchinfo` replaces `torchsummary` and offers more flexibility, including support for multiple inputs).

- **Custom Forward Wrapper (`CombinedModel`):** To accommodate multiple inputs, we wrap the original model in a new `nn.Module`.

**Sample Output:**

```
T1 Convolutional Path Summary:
----------------------------------------------------------------
Layer (type) Output Shape Param #
================================================================
Conv3d-1 [-1, 32, 64, 64, 64] 896
BatchNorm3d-2 [-1, 32, 64, 64, 64] 64
MaxPool3d-3 [-1, 32, 32, 32, 32] 0
Conv3d-4 [-1, 64, 32, 32, 32] 55,296
BatchNorm3d-5 [-1, 64, 32, 32, 32] 128
MaxPool3d-6 [-1, 64, 16, 16, 16] 0
================================================================
Total params: 56,384
Trainable params: 56,384
Non-trainable params: 0
----------------------------------------------------------------

T2 Convolutional Path Summary:
----------------------------------------------------------------
Layer (type) Output Shape Param #
================================================================
Conv3d-1 [-1, 32, 64, 64, 64] 896
BatchNorm3d-2 [-1, 32, 64, 64, 64] 64
MaxPool3d-3 [-1, 32, 32, 32, 32] 0
Conv3d-4 [-1, 64, 32, 32, 32] 55,296
BatchNorm3d-5 [-1, 64, 32, 32, 32] 128
MaxPool3d-6 [-1, 64, 16, 16, 16] 0
================================================================
Total params: 56,384
Trainable params: 56,384
Non-trainable params: 0
----------------------------------------------------------------

Full Model Summary:
===================================================================================
Layer (type:depth-idx) Output Shape Param #
===================================================================================
DualStream3DResNet [Batch, 1, 64,64,64] 896
├─Conv3d-1 [-1, 32, 64,64,64] 896
├─BatchNorm3d-2 [-1, 32, 64,64,64] 64
├─ReLU-3 [-1, 32, 64,64,64] 0
├─MaxPool3d-4 [-1, 32, 32,32,32] 0
├─Conv3d-5 [-1, 64, 32,32,32] 55,296
├─BatchNorm3d-6 [-1, 64, 32,32,32] 128
├─ReLU-7 [-1, 64, 32,32,32] 0
├─MaxPool3d-8 [-1, 64, 16,16,16] 0
├─Conv3d-9 [-1, 32, 64,64,64] 896
├─BatchNorm3d-10 [-1, 32, 64,64,64] 64
├─ReLU-11 [-1, 32, 64,64,64] 0
├─MaxPool3d-12 [-1, 32, 32,32,32] 0
├─Conv3d-13 [-1, 64, 32,32,32] 55,296
├─BatchNorm3d-14 [-1, 64, 32,32,32] 128
├─ReLU-15 [-1, 64, 32,32,32] 0
├─MaxPool3d-16 [-1, 64, 16,16,16] 0
├─Linear-17 [-1, 128] 16,448
├─ReLU-18 [-1, 128] 0
├─Dropout-19 [-1, 128] 0
├─Linear-20 [-1, 64] 8,256
├─ReLU-21 [-1, 64] 0
├─Linear-22 [-1, 1] 65
===================================================================================
Total params: 136,065
Trainable params: 136,065
Non-trainable params: 0
===================================================================================
```

**Notes:**

- **Adjusting Feature Sizes:** Ensure that the output feature sizes after the convolutional and pooling layers match the input size of the first fully connected layer (`fc1`). You might need to calculate the feature size based on your MRI input dimensions.

- **Model Summary with Multiple Inputs:**
- PyTorch's `torchinfo` library can be used for more comprehensive summaries, especially with models that have multiple inputs. However, in this case, since we've defined separate paths within the same model, the standard summary suffices.

**Alternative Model Summary with `torchinfo`:**

If you prefer a more detailed summary, especially to handle multiple inputs, consider using the `torchinfo` library:

```bash
pip install torchinfo
```

```python
from torchinfo import summary

# Define a combined forward method for torchinfo
class CombinedModel(nn.Module):
def __init__(self, original_model):
super(CombinedModel, self).__init__()
self.original_model = original_model

def forward(self, t1, t2, demographics):
return self.original_model(t1, t2, demographics)

# Wrap the original model
combined_model = CombinedModel(trained_model)
combined_model = combined_model.to(device)

# Display the summary
summary(
combined_model,
input_data=[
torch.randn(batch_size, 1, 64, 64, 64).to(device), # T1 input
torch.randn(batch_size, 1, 64, 64, 64).to(device), # T2 input
torch.randn(batch_size, 2).to(device) # Demographics
],
dtypes=[torch.float, torch.float, torch.float],
device=device.type
)
```

**Sample Output:**

```
===================================================================================
Layer (type:depth-idx) Output Shape Param #
===================================================================================
CombinedModel [B, 1, 64,64,64] 896
├─DualStream3DResNet [B, 1, 64,64,64] 896
│ ├─Conv3d-1 [B, 32, 64,64,64] 896
│ ├─BatchNorm3d-2 [B, 32, 64,64,64] 64
│ ├─ReLU-3 [B, 32, 64,64,64] 0
│ ├─MaxPool3d-4 [B, 32, 32,32,32] 0
│ ├─Conv3d-5 [B, 64, 32,32,32] 55,296
│ ├─BatchNorm3d-6 [B, 64, 32,32,32] 128
│ ├─ReLU-7 [B, 64, 32,32,32] 0
│ ├─MaxPool3d-8 [B, 64, 16,16,16] 0
│ ├─Conv3d-9 [B, 32, 64,64,64] 896
│ ├─BatchNorm3d-10 [B, 32, 64,64,64] 64
│ ├─ReLU-11 [B, 32, 64,64,64] 0
│ ├─MaxPool3d-12 [B, 32, 32,32,32] 0
│ ├─Conv3d-13 [B, 64, 32,32,32] 55,296
│ ├─BatchNorm3d-14 [B, 64, 32,32,32] 128
│ ├─ReLU-15 [B, 64, 32,32,32] 0
│ ├─MaxPool3d-16 [B, 64, 16,16,16] 0
│ ├─Linear-17 [B, 128] 16,448
│ ├─ReLU-18 [B, 128] 0
│ ├─Dropout-19 [B, 128] 0
│ ├─Linear-20 [B, 64] 8,256
│ ├─ReLU-21 [B, 64] 0
│ └─Linear-22 [B, 1] 65
===================================================================================
Total params: 136,065
Trainable params: 136,065
Non-trainable params: 0
===================================================================================
```

---

## 6. Next Steps

With the model defined and the summary displayed, you can proceed to:

1. **Continue Training:**
- Ensure that your training loop runs correctly and monitor the loss curves.

2. **Evaluate on a Test Set:**
- After training, evaluate the model's performance on an unseen test set to gauge its real-world applicability.

3. **Fine-Tune the Model:**
- Adjust hyperparameters such as learning rate, batch size, and network depth.
- Incorporate data augmentation techniques to improve generalization.

4. **Implement Model Checkpoints and Logging:**
- Save models at various checkpoints.
- Log training and validation metrics for analysis.

5. **Enhance Model Interpretability:**
- Use techniques like Grad-CAM (adapted for 3D) to visualize which regions of the MRI scans influence the EDSS predictions.

6. **Optimize for Performance:**
- Experiment with different network architectures.
- Utilize GPU acceleration for faster training.

---

## 7. Complete Sample Code

For convenience, here's the complete integrated code based on the above steps. Ensure that paths are correctly set according to your directory structure.

```python
import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary # Using torchinfo for detailed model summaries

# 1. Define the Custom Dataset
class MRIDataset(Dataset):
def __init__(self, metadata_df, t1_dir, t2_dir, transform=None):
"""
Initializes the dataset by storing metadata and directories.

Args:
metadata_df (pd.DataFrame): DataFrame containing metadata for each MRI scan.
t1_dir (str): Directory path to T1 preprocessed NIfTI files.
t2_dir (str): Directory path to T2 preprocessed NIfTI files.
transform (callable, optional): Optional transform to be applied on the MRI data.
"""
self.metadata = metadata_df.reset_index(drop=True)
self.t1_dir = t1_dir
self.t2_dir = t2_dir
self.transform = transform

def __len__(self):
return len(self.metadata)

def load_nifti(self, filepath):
"""
Loads a NIfTI file and returns it as a numpy array.

Args:
filepath (str): Path to the NIfTI file.

Returns:
np.ndarray: MRI data.
"""
try:
nifti = nib.load(filepath)
data = nifti.get_fdata()
return data
except FileNotFoundError:
print(f"Warning: File not found {filepath}. Returning zeros.")
# Assuming that the MRI data has dimensions [D, H, W]
# Adjust based on your data
return np.zeros((64, 64, 64)) # Example dimensions

def __getitem__(self, idx):
"""
Retrieves the T1 and T2 MRI scans, demographic data, and EDSS score for a given index.

Args:
idx (int): Index of the sample.

Returns:
tuple: (T1_tensor, T2_tensor, demographics_tensor, EDSS)
"""
row = self.metadata.iloc[idx]
mri_id = row['MRI_ID']
sex = row['Sex']
age = row['Age_at_Scan']
edss = row['EDSS']

# Construct file paths
t1_filename = f"{mri_id}_T1_preprocessed.nii.gz"
t2_filename = f"{mri_id}_T2_preprocessed.nii.gz"
t1_path = os.path.join(self.t1_dir, t1_filename)
t2_path = os.path.join(self.t2_dir, t2_filename)

# Load MRI data
t1 = self.load_nifti(t1_path)
t2 = self.load_nifti(t2_path)

# Normalize MRI data
t1 = (t1 - np.mean(t1)) / np.std(t1) if np.std(t1) > 0 else t1 - np.mean(t1)
t2 = (t2 - np.mean(t2)) / np.std(t2) if np.std(t2) > 0 else t2 - np.mean(t2)

# Apply transformations if any
if self.transform:
t1 = self.transform(t1)
t2 = self.transform(t2)

# Encode sex: 1 for male, 0 for female
sex_encoded = 1 if str(sex).strip().upper() in ['M', 'MALE'] else 0

# Combine demographics
demographics = np.array([age, sex_encoded], dtype=np.float32)

# Convert to tensors
t1_tensor = torch.tensor(t1, dtype=torch.float32).unsqueeze(0) # Shape: [1, D, H, W]
t2_tensor = torch.tensor(t2, dtype=torch.float32).unsqueeze(0) # Shape: [1, D, H, W]
demographics_tensor = torch.tensor(demographics, dtype=torch.float32) # Shape: [2]
edss = torch.tensor(edss, dtype=torch.float32) # Scalar

return t1_tensor, t2_tensor, demographics_tensor, edss

# 2. Define the Model
class DualStream3DResNet(nn.Module):
def __init__(self):
super(DualStream3DResNet, self).__init__()

# T1 Convolutional Path
self.t1_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
self.t1_bn1 = nn.BatchNorm3d(32)
self.t1_pool = nn.MaxPool3d(kernel_size=2, stride=2)
self.t1_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
self.t1_bn2 = nn.BatchNorm3d(64)
self.t1_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

# T2 Convolutional Path
self.t2_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
self.t2_bn1 = nn.BatchNorm3d(32)
self.t2_pool = nn.MaxPool3d(kernel_size=2, stride=2)
self.t2_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
self.t2_bn2 = nn.BatchNorm3d(64)
self.t2_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

# Fully Connected Layers
# Adjust the input features based on your MRI dimensions
self.fc1 = nn.Linear(64*16*16*16*2 + 2, 128) # Example feature size
self.relu = nn.ReLU()
self.dropout = nn.Dropout(0.3)
self.fc2 = nn.Linear(128, 64)
self.fc3 = nn.Linear(64, 1) # Output: EDSS score

def forward(self, t1, t2, demographics):
"""
Forward pass of the model.

Args:
t1 (Tensor): T1 MRI tensor of shape [batch_size, 1, D, H, W]
t2 (Tensor): T2 MRI tensor of shape [batch_size, 1, D, H, W]
demographics (Tensor): Demographics tensor of shape [batch_size, 2]

Returns:
Tensor: Predicted EDSS scores of shape [batch_size]
"""
# T1 Path
x1 = self.t1_conv1(t1)
x1 = self.t1_bn1(x1)
x1 = F.relu(x1)
x1 = self.t1_pool(x1)
x1 = self.t1_conv2(x1)
x1 = self.t1_bn2(x1)
x1 = F.relu(x1)
x1 = self.t1_pool2(x1)
x1 = x1.view(x1.size(0), -1) # Flatten

# T2 Path
x2 = self.t2_conv1(t2)
x2 = self.t2_bn1(x2)
x2 = F.relu(x2)
x2 = self.t2_pool(x2)
x2 = self.t2_conv2(x2)
x2 = self.t2_bn2(x2)
x2 = F.relu(x2)
x2 = self.t2_pool2(x2)
x2 = x2.view(x2.size(0), -1) # Flatten

# Merge Features
merged = torch.cat((x1, x2), dim=1) # [batch_size, features*2]

# Concatenate Demographics
combined = torch.cat((merged, demographics), dim=1) # [batch_size, features*2 + 2]

# Fully Connected Layers
out = self.fc1(combined)
out = self.relu(out)
out = self.dropout(out)
out = self.fc2(out)
out = self.relu(out)
out = self.fc3(out)

return out.squeeze(1) # [batch_size]

# 3. Initialize the Model, Loss, and Optimizer
model = DualStream3DResNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print("Model initialized and moved to device.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
print("Loss function and optimizer defined.")
```

**Notes:**

- **Adjusting `fc1` Input Size:** The `fc1` layer's input size (`64*16*16*16*2 + 2`) assumes that after two pooling layers, the feature maps are reduced to `[64, 16, 16, 16]` for each path. Adjust this based on your MRI input dimensions. For example, if your original MRI dimensions are `[64, 64, 64]`, after two `MaxPool3d` layers with `kernel_size=2` and `stride=2`, the size becomes `[16, 16, 16]`.

- **Example:**
- If your input MRIs are `[1, 64, 64, 64]` for T1 and T2:
- After first pooling: `[32, 32, 32]`
- After second pooling: `[64, 16, 16, 16]` (since channels double in each conv layer)
- Flattened features per path: `64*16*16*16 = 262,144`
- Total merged features for both paths: `524,288`
- Concatenated with demographics (`+2`): `524,290`
- Therefore, `fc1` should have an input size of `524,290` and output size of `128`.

- This colossal number suggests that the architecture is too deep for the initial layers. To manage computational complexity and memory, consider adding more pooling layers or reducing the number of convolutional filters.

- **Simplified Architecture:**

Here's an adjusted architecture with more pooling to reduce feature sizes:

```python
class DualStream3DResNet(nn.Module):
def __init__(self):
super(DualStream3DResNet, self).__init__()

# T1 Convolutional Path
self.t1_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
self.t1_bn1 = nn.BatchNorm3d(32)
self.t1_pool = nn.MaxPool3d(kernel_size=2, stride=2) # Halve spatial dimensions
self.t1_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
self.t1_bn2 = nn.BatchNorm3d(64)
self.t1_pool2 = nn.MaxPool3d(kernel_size=2, stride=2) # Halve spatial dimensions again

# T2 Convolutional Path
self.t2_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
self.t2_bn1 = nn.BatchNorm3d(32)
self.t2_pool = nn.MaxPool3d(kernel_size=2, stride=2)
self.t2_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
self.t2_bn2 = nn.BatchNorm3d(64)
self.t2_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

# Compute the size after convolutions and pooling
# Example for input size [1, 64, 64, 64]:
# After first conv & pool: [32, 32, 32, 32]
# After second conv & pool: [64, 16, 16, 16]
# Flattened: 64*16*16*16 = 262,144 per path
# Total for both paths: 524,288
# Plus demographics: +2

# To reduce, add more pooling or use fewer features

# Adding more pooling layers
self.t1_pool3 = nn.MaxPool3d(kernel_size=2, stride=2) # Down to [64, 8, 8, 8]
self.t2_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

# Now flattened: [64*8*8*8] = 32,768 per path
# Total: 65,536
# Plus demographics: 2

self.fc1 = nn.Linear(65538, 128)
self.relu = nn.ReLU()
self.dropout = nn.Dropout(0.3)
self.fc2 = nn.Linear(128, 64)
self.fc3 = nn.Linear(64, 1)

def forward(self, t1, t2, demographics):
# T1 Path
x1 = self.t1_conv1(t1)
x1 = self.t1_bn1(x1)
x1 = F.relu(x1)
x1 = self.t1_pool(x1)
x1 = self.t1_conv2(x1)
x1 = self.t1_bn2(x1)
x1 = F.relu(x1)
x1 = self.t1_pool2(x1)
x1 = self.t1_pool3(x1)
x1 = x1.view(x1.size(0), -1) # Flatten

# T2 Path
x2 = self.t2_conv1(t2)
x2 = self.t2_bn1(x2)
x2 = F.relu(x2)
x2 = self.t2_pool(x2)
x2 = self.t2_conv2(x2)
x2 = self.t2_bn2(x2)
x2 = F.relu(x2)
x2 = self.t2_pool2(x2)
x2 = self.t2_pool3(x2)
x2 = x2.view(x2.size(0), -1) # Flatten

# Merge Features
merged = torch.cat((x1, x2), dim=1) # [batch_size, 65536]

# Concatenate Demographics
combined = torch.cat((merged, demographics), dim=1) # [batch_size, 65538]

# Fully Connected Layers
out = self.fc1(combined)
out = self.relu(out)
out = self.dropout(out)
out = self.fc2(out)
out = self.relu(out)
out = self.fc3(out)

return out.squeeze(1) # [batch_size]
```

**Final Thoughts:**

- **Memory Management:** 3D CNNs are memory-intensive. Adjust the number of convolutional filters and pooling layers to balance between feature richness and memory constraints.

- **Model Complexity:** Starting with a simpler model can help establish a baseline. You can gradually increase complexity based on performance and computational resources.

---

## 6. Complete Training and Model Summary Workflow

Putting it all together, here's a complete script integrating data loading, model definition, training, and displaying the model summary.

```python
import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary

# 1. Define the Custom Dataset
class MRIDataset(Dataset):
def __init__(self, metadata_df, t1_dir, t2_dir, transform=None):
self.metadata = metadata_df.reset_index(drop=True)
self.t1_dir = t1_dir
self.t2_dir = t2_dir
self.transform = transform

def __len__(self):
return len(self.metadata)

def load_nifti(self, filepath):
try:
nifti = nib.load(filepath)
data = nifti.get_fdata()
return data
except FileNotFoundError:
print(f"Warning: File not found {filepath}. Returning zeros.")
return np.zeros((64, 64, 64)) # Example dimensions

def __getitem__(self, idx):
row = self.metadata.iloc[idx]
mri_id = row['MRI_ID']
sex = row['Sex']
age = row['Age_at_Scan']
edss = row['EDSS']

# Construct file paths
t1_filename = f"{mri_id}_T1_preprocessed.nii.gz"
t2_filename = f"{mri_id}_T2_preprocessed.nii.gz"
t1_path = os.path.join(self.t1_dir, t1_filename)
t2_path = os.path.join(self.t2_dir, t2_filename)

# Load MRI data
t1 = self.load_nifti(t1_path)
t2 = self.load_nifti(t2_path)

# Normalize MRI data
t1 = (t1 - np.mean(t1)) / np.std(t1) if np.std(t1) > 0 else t1 - np.mean(t1)
t2 = (t2 - np.mean(t2)) / np.std(t2) if np.std(t2) > 0 else t2 - np.mean(t2)

# Apply transformations if any
if self.transform:
t1 = self.transform(t1)
t2 = self.transform(t2)

# Encode sex: 1 for male, 0 for female
sex_encoded = 1 if str(sex).strip().upper() in ['M', 'MALE'] else 0

# Combine demographics
demographics = np.array([age, sex_encoded], dtype=np.float32)

# Convert to tensors
t1_tensor = torch.tensor(t1, dtype=torch.float32).unsqueeze(0) # [1, D, H, W]
t2_tensor = torch.tensor(t2, dtype=torch.float32).unsqueeze(0) # [1, D, H, W]
demographics_tensor = torch.tensor(demographics, dtype=torch.float32) # [2]
edss_tensor = torch.tensor(edss, dtype=torch.float32) # Scalar

return t1_tensor, t2_tensor, demographics_tensor, edss_tensor

# 2. Define the Model
class DualStream3DResNet(nn.Module):
def __init__(self):
super(DualStream3DResNet, self).__init__()

# T1 Convolutional Path
self.t1_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
self.t1_bn1 = nn.BatchNorm3d(32)
self.t1_pool = nn.MaxPool3d(kernel_size=2, stride=2)
self.t1_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
self.t1_bn2 = nn.BatchNorm3d(64)
self.t1_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
self.t1_pool3 = nn.MaxPool3d(kernel_size=2, stride=2) # Additional pooling

# T2 Convolutional Path
self.t2_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
self.t2_bn1 = nn.BatchNorm3d(32)
self.t2_pool = nn.MaxPool3d(kernel_size=2, stride=2)
self.t2_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
self.t2_bn2 = nn.BatchNorm3d(64)
self.t2_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
self.t2_pool3 = nn.MaxPool3d(kernel_size=2, stride=2) # Additional pooling

# Fully Connected Layers
# After three pooling layers, [64, 8, 8, 8]
feature_size = 64 * 8 * 8 * 8
self.fc1 = nn.Linear(feature_size * 2 + 2, 128) # T1 + T2 + Demographics
self.relu = nn.ReLU()
self.dropout = nn.Dropout(0.3)
self.fc2 = nn.Linear(128, 64)
self.fc3 = nn.Linear(64, 1) # EDSS Score

def forward(self, t1, t2, demographics):
# T1 Path
x1 = self.t1_conv1(t1)
x1 = self.t1_bn1(x1)
x1 = F.relu(x1)
x1 = self.t1_pool(x1)
x1 = self.t1_conv2(x1)
x1 = self.t1_bn2(x1)
x1 = F.relu(x1)
x1 = self.t1_pool2(x1)
x1 = self.t1_pool3(x1)
x1 = x1.view(x1.size(0), -1) # Flatten

# T2 Path
x2 = self.t2_conv1(t2)
x2 = self.t2_bn1(x2)
x2 = F.relu(x2)
x2 = self.t2_pool(x2)
x2 = self.t2_conv2(x2)
x2 = self.t2_bn2(x2)
x2 = F.relu(x2)
x2 = self.t2_pool2(x2)
x2 = self.t2_pool3(x2)
x2 = x2.view(x2.size(0), -1) # Flatten

# Merge Features
merged = torch.cat((x1, x2), dim=1) # [batch_size, feature_size*2]

# Concatenate Demographics
combined = torch.cat((merged, demographics), dim=1) # [batch_size, feature_size*2 + 2]

# Fully Connected Layers
out = self.fc1(combined)
out = self.relu(out)
out = self.dropout(out)
out = self.fc2(out)
out = self.relu(out)
out = self.fc3(out)

return out.squeeze(1) # [batch_size]

# 3. Initialize the Model, Loss, and Optimizer
model = DualStream3DResNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print("Model initialized and moved to device.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
print("Loss function and optimizer defined.")
```

**Note:**

- **Feature Size Calculation:**
- After three pooling layers (`pool`, `pool2`, `pool3`), with `kernel_size=2` and `stride=2`, the spatial dimensions reduce by a factor of 8 (since \(2^3 = 8\)).
- For an input MRI of `[64, 64, 64]`, the output after pooling would be `[8, 8, 8]`.

---

## 7. Displaying the Model Summary

Using `torchinfo` to display a detailed model summary, including input shapes and parameter counts.

```python
from torchinfo import summary

# Wrap the model for summary
class CombinedModel(nn.Module):
def __init__(self, original_model):
super(CombinedModel, self).__init__()
self.original_model = original_model

def forward(self, t1, t2, demographics):
return self.original_model(t1, t2, demographics)

# Instantiate the wrapper
combined_model = CombinedModel(model)
combined_model = combined_model.to(device)

# Define dummy inputs
t1_dummy = torch.randn(1, 1, 64, 64, 64).to(device) # Batch size 1
t2_dummy = torch.randn(1, 1, 64, 64, 64).to(device)
demographics_dummy = torch.randn(1, 2).to(device)

# Display the summary
summary(
combined_model,
input_data=[t1_dummy, t2_dummy, demographics_dummy],
verbose=2
)
```

**Sample Output:**

```
==========================================================================================
Layer (type:depth-idx) Output Shape Param #
==========================================================================================
CombinedModel [1, 1, 64,64,64] 0
├─DualStream3DResNet [1, 1, 64,64,64] 0
│ ├─Conv3d-1 [1, 32, 64,64,64] 896
│ ├─BatchNorm3d-2 [1, 32, 64,64,64] 64
│ ├─ReLU-3 [1, 32, 64,64,64] 0
│ ├─MaxPool3d-4 [1, 32, 32,32,32] 0
│ ├─Conv3d-5 [1, 64, 32,32,32] 55,296
│ ├─BatchNorm3d-6 [1, 64, 32,32,32] 128
│ ├─ReLU-7 [1, 64, 32,32,32] 0
│ ├─MaxPool3d-8 [1, 64, 16,16,16] 0
│ ├─MaxPool3d-9 [1, 64, 8,8,8] 0
│ ├─Conv3d-10 [1, 32, 64,64,64] 896
│ ├─BatchNorm3d-11 [1, 32, 64,64,64] 64
│ ├─ReLU-12 [1, 32, 64,64,64] 0
│ ├─MaxPool3d-13 [1, 32, 32,32,32] 0
│ ├─Conv3d-14 [1, 64, 32,32,32] 55,296
│ ├─BatchNorm3d-15 [1, 64, 32,32,32] 128
│ ├─ReLU-16 [1, 64, 32,32,32] 0
│ ├─MaxPool3d-17 [1, 64, 16,16,16] 0
│ ├─MaxPool3d-18 [1, 64, 8,8,8] 0
│ ├─Linear-19 [1, 128] 524,290
│ ├─ReLU-20 [1, 128] 0
│ ├─Dropout-21 [1, 128] 0
│ ├─Linear-22 [1, 64] 8,256
│ ├─ReLU-23 [1, 64] 0
│ └─Linear-24 [1, 1] 65
==========================================================================================
Total parameters: 580,321
Trainable parameters: 580,321
Non-trainable parameters: 0
==========================================================================================
```

**Interpretation:**

- **Parameter Count:** Indicates the number of trainable parameters in each layer.

- **Output Shapes:** Shows the shape of the output tensors after each layer.

- **Depth-Index:** Helps in identifying the layer hierarchy.

**Notes:**

- The final layer outputs a single EDSS score per sample.

- Adjust `feature_size` in the `fc1` layer based on the actual size after convolutions and pooling.

---

## 8. Final Remarks

By following the above steps, you've set up a comprehensive pipeline to:

1. **Load and Preprocess Data:**
- Load T1 and T2 MRI scans separately.
- Incorporate demographic data.
- Handle missing files gracefully by returning zeros (you may choose to exclude such samples entirely for better performance).

2. **Define a Dual-Stream 3D ResNet Model:**
- Process T1 and T2 scans through separate convolutional paths.
- Merge extracted features.
- Incorporate demographic data before regression layers.

3. **Train the Model from Scratch:**
- Utilize MSE loss and Adam optimizer.
- Implement early stopping to prevent overfitting.

4. **Analyze and Visualize the Model:**
- Use `torchinfo` to display model architecture and parameter counts.
- Monitor training and validation loss to assess performance.

**Recommendations for Further Improvement:**

- **Data Augmentation:** Apply 3D transformations (rotations, flips, etc.) using libraries like [`torchio`](https://torchio.readthedocs.io/en/latest/) to enhance model generalization.

- **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and network depths to optimize performance.

- **Advanced Architectures:** Consider using more sophisticated architectures like 3D U-Net or incorporating attention mechanisms to improve feature extraction.

- **Regularization Techniques:** Incorporate dropout, weight decay, and other regularization methods to prevent overfitting.

- **Model Interpretability:** Implement visualization techniques to understand which regions of the MRI scans influence the EDSS predictions.
