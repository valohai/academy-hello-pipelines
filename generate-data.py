import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import valohai

# Define a Valohai step "generate-data"
valohai.prepare(step="generate-data", image="python:3.9")

training_data = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor(),
)

train_data_path = valohai.outputs().path('train_data.pth')
test_data_path = valohai.outputs().path('test_data.pth')

torch.save(training_data, train_data_path)
torch.save(test_data, test_data_path)
