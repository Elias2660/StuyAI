import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
from scipy import signal
from scipy.io import wavfile
import time

print("Data preprocessing...")

dog_train_path = "./cats_dogs/train/dog/"
dog_test_path = "./cats_dogs/test/dogs/"

cat_train_path = "./cats_dogs/train/cat/"
cat_test_path = "./cats_dogs/test/cats/"

labels_to_idx = {"cat": [1, 0], "dog": [0, 1]}

def get_waveforms_labels(path, label):
    features = []
    labels = []
    for filename in os.listdir(path):
        sample_rate, samples = wavfile.read(os.path.join(path, filename))
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        features.append(spectrogram)
        labels.append(labels_to_idx[label])
    return features, labels

def pad_spec(spectrogram, max_length):
    return np.pad(spectrogram, ((0,0), (0, max_length - spectrogram.shape[1])), mode='mean')

dog_train_features, dog_train_labels = get_waveforms_labels(dog_train_path, "dog")
cat_train_features, cat_train_labels = get_waveforms_labels(cat_train_path, "cat")

dog_test_features, dog_test_labels = get_waveforms_labels(dog_test_path, "dog")
cat_test_features, cat_test_labels = get_waveforms_labels(cat_test_path, "cat")

max_length = max(max(s.shape[1] for s in dog_train_features + cat_train_features), 
                 max(s.shape[1] for s in dog_test_features + cat_test_features))

X_train = np.array([pad_spec(s, max_length) for s in dog_train_features + cat_train_features])
y_train = np.array(dog_train_labels + cat_train_labels)

X_test = np.array([pad_spec(s, max_length) for s in dog_test_features + cat_test_features])
y_test = np.array(dog_test_labels + cat_test_labels)

# Convert to tensors and create DataLoaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1)  # Conv2d layer
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(self.calculate_linear_input_size(), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the linear layer
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

    def calculate_linear_input_size(self):
        # Temporary input for size calculation
        temp_input = torch.zeros(1, 1, 129, max_length)
        temp_out = self.conv1(temp_input)
        temp_out = self.pool(temp_out)
        temp_out = self.conv2(temp_out)
        temp_out = self.pool(temp_out)
        temp_out = self.conv3(temp_out)
        temp_out = self.pool(temp_out)
        return temp_out.view(temp_out.size(0), -1).shape[1]
    
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("Training...")

# Initialize best loss to a high value
best_loss = float('inf')
best_model = None

# Training loop
epoch_losses = {}
for epoch in range(250):
    start_time = time.time()
    model.train()
    losses = []
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, torch.max(y, 1)[1])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_loss = np.mean(losses)
    epoch_losses[epoch] = avg_loss

    # Check if the current model is better (i.e., has lower loss)
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model = model.state_dict()

    print(f"Epoch:{epoch+1:3d}, Loss-train:{avg_loss:1.3f}, Time:{time.time() - start_time:2.2f}s")

# Save the best model
torch.save(best_model, 'best_model.pth')

# Plotting and saving to a file
plt.title("Loss")
plt.plot(list(epoch_losses.values()))
plt.savefig('training_loss_plot.png')
