import torch
from torchvision import datasets
from torch import nn
from torch.utils.data import DataLoader
import zipfile
import valohai

# Valohai: Define parameters
my_parameters = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 5
}

# Valohai: Define inputs
my_inputs = {
    "train": "",
    "test": ""
}

# Valohai: Define a step with parameters and inputs
valohai.prepare(step="train-model",
                image="python:3.9",
                default_parameters=my_parameters,
                default_inputs=my_inputs)

# Valohai: Get the parameter values
# We're using valohai.parameters to get the actual values during runtime
# The values defined in my_parameters are just default values valohai.yaml
batch_size = valohai.parameters("batch_size").value
epochs = valohai.parameters("epochs").value
learning_rate = valohai.parameters("learning_rate").value

# Valohai: Get the path to our input files
train_data_path = valohai.inputs("train").path()
test_data_path = valohai.inputs("test").path()

# Valohai: Load the previously generated train and test data
train_data = torch.load(train_data_path)
test_data = torch.load(test_data_path)

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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

model = NeuralNetwork().to(device)

# Optimize the model
# Valohai: We're passing the parameter value learnign_rate to the optimizer
# ref: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# Define a single training loop
# In a single training loop, the model makes predictions on the training dataset (fed to it in batches),
# and backpropagates the prediction error to adjust the model’s parameters.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

# We also check the model’s performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    # Valohai: Print out the accuracy and test_loss after each epoch
    with valohai.metadata.logger() as logger :
        logger.log("epoch", epoch)
        logger.log("accuracy", f"{correct:>2f}")
        logger.log("test_loss", f"{test_loss:>8f}")


# The training process is conducted over several iterations (epochs).
# During each epoch, the model learns parameters to make better predictions.
# We print the model’s accuracy and loss at each epoch as Valohai metadata
for epoch in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, epoch)

model_path = valohai.outputs().path("model.pth")
torch.save(model.state_dict(), model_path)
