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

# Define the network TODO: change to your network
#from networks.create_dummy_network import SimpleNet
from networks.quadbug.networks import controlNetwork

# Initialize the model TODO: change to your model
model = controlNetwork()
# TODO: change to the path of your model
model.load_state_dict(torch.load('networks/quadbug/control_net.pt'))
model.eval()

sample_input = (torch.rand(1, 15),)
input_data = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]], dtype=torch.float32)
torch_output = model(*input_data) # [-0.4581,  0.0559, -0.2546, -0.2886, -0.2909, -0.2912]

# Convert the model
tfLite_model = ai_edge_torch.convert(model, sample_input)
tfLite_output = tfLite_model(*input_data)

# Compare the outputs
print("Input:" + str(input_data))
print("PyTorch output:" + str(torch_output))
print("TFLite output:" + str(tfLite_output))

# Save the TFLite model
#ai_edge_torch.save(tfLite_model, 'simple_net.tflite1')
# TODO change to the path where you want to save your model
tfLite_model.export('networks/quadbug/control_net.tflite')