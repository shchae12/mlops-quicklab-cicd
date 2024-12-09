import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from typing import Tuple
from trainer import NeuralNet, train_loader, save_model


@pytest.fixture
def mock_data() -> Tuple[DataLoader, DataLoader]:
    """
    Fixture to create mock DataLoader objects for training and testing.
    """
    input_size = 784
    num_classes = 10
    batch_size = 100

    # Generate random data
    x_train = torch.rand(500, input_size)
    y_train = torch.randint(0, num_classes, (500,))
    x_test = torch.rand(100, input_size)
    y_test = torch.randint(0, num_classes, (100,))

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


@pytest.fixture
def model() -> NeuralNet:
    """
    Fixture to provide a NeuralNet model instance.
    """
    return NeuralNet()


def test_model_initialization(model: NeuralNet) -> None:
    """
    Test the model initialization and structure.
    """
    assert isinstance(model, NeuralNet)
    assert model.fc1.in_features == 784
    assert model.fc2.out_features == 10


def test_forward_pass(
    model: NeuralNet, mock_data: Tuple[DataLoader, DataLoader]
) -> None:
    """
    Test the forward pass of the model.
    """
    train_loader, _ = mock_data
    model.eval()
    for images, _ in train_loader:
        output = model(images)
        assert output.shape[0] == images.shape[0]
        assert output.shape[1] == 10
        break


def test_training_step(
    model: NeuralNet, mock_data: Tuple[DataLoader, DataLoader]
) -> None:
    """
    Test a single training step.
    """
    train_loader, _ = mock_data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0
        break


def test_save_model(model: NeuralNet, tmp_path) -> None:
    """
    Test saving the model to a temporary directory.
    """
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    assert model_path.exists()
    assert model_path.stat().st_size > 0


def test_data_loading(mock_data: Tuple[DataLoader, DataLoader]) -> None:
    """
    Test the DataLoader functionality.
    """
    train_loader, test_loader = mock_data
    assert len(train_loader.dataset) == 500
    assert len(test_loader.dataset) == 100
