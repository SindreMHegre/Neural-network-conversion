# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torchvision
import ai_edge_torch
import numpy
#from google.colab import files

# Define the network TODO: change to your network, if its a standard network, you can jump to initialization
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

# Initialize the model TODO: change to your model
model = SimpleNet()
# TODO: change to the path of your model
model.load_state_dict(torch.load('simple_net.pt'))
model.eval()

sample_input = (torch.rand(1, 10),)
torch_output = model(*sample_input)

# Convert the model
tfLite_model = ai_edge_torch.convert(model, sample_input)
tfLite_output = tfLite_model(*sample_input)

# Compare the outputs
print("Input:" + str(sample_input))
print("PyTorch output:" + str(torch_output))
print("TFLite output:" + str(tfLite_output))

# Save the TFLite model
#ai_edge_torch.save(tfLite_model, 'simple_net.tflite1')
# TODO change to the path where you want to save your model
tfLite_model.export('simple_net.tflite')