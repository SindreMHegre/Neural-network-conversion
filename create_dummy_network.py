import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 32)  # Input layer (10 inputs)
        self.fc2 = nn.Linear(32, 32)  # Hidden layer 1
        self.fc3 = nn.Linear(32, 2)   # Output layer (2 outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Generate random data
def generate_data(num_samples=1000):
    X = np.random.rand(num_samples, 10).astype(np.float32)  # 10 inputs
    # Define a simple relationship for the output, for example, summing first 5 values and the last 5 values
    y = np.column_stack([X[:, :5].sum(axis=1), X[:, 5:].sum(axis=1)]).astype(np.float32)
    return X, y

# Initialize the model
model = SimpleNet()

# Create training data
X_train, y_train = generate_data()

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model with some input
model.eval()
test_input = torch.rand(1, 10)  # Example input with 10 features
output = model(test_input)
print("Input:", test_input)
print("Output:", output)

test_input = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])  # Example input with 10 features
output = model(test_input)
print("Input:", test_input)
print("Output should be [1.5, 4.0]:", output)


# Export the trained model to ONNX
# dummy_input = torch.randn(1, 10)
# torch.onnx.export(model, dummy_input, "simple_net.onnx", input_names=["input"], output_names=["output"])
# Save the model
torch.save(model.state_dict(), "simple_net.pt")