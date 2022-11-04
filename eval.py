import torch
from torch import nn
import valohai
from torch.utils.data import DataLoader

my_inputs = {
    "model": "",
    "test": ""
}

my_parameters = {
    "batch_size": 25
}

# Valohai: Define a step with parameters and inputs
valohai.prepare(step="evaluate",
                image="python:3.9",
                default_parameters=my_parameters,
                default_inputs=my_inputs)

# Valohai: Get the parameter values
# We're using valohai.parameters to get the actual values during runtime
# The values defined in my_parameters are just default values valohai.yaml
batch_size = valohai.parameters("batch_size").value

# Valohai: Get the path to our input files
test_data_path = valohai.inputs("test").path()
model_path = valohai.inputs("model").path()

# Valohai: Load the previously generated test data
test_data = torch.load(test_data_path)

# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Define model
# ref: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#creating-models
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load(model_path))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')