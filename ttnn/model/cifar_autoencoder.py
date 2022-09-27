# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pathlib
# Torchvision
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class PretrainAutoencoder(torch.nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        # Create model
        autoencoder = create_model()
        # Load data
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_transforms_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
        ]

        train_transform = transforms.Compose(train_transforms_list)
        trainset = torchvision.datasets.CIFAR10(
            root='../.data', train=True, download=True,
            transform=train_transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=16, shuffle=True, num_workers=2)

        # Define an optimizer and criterion
        criterion = nn.BCELoss()
        optimizer = optim.Adam(autoencoder.parameters())

        for epoch in range(100):
            running_loss = 0.0
            for i, (inputs, _) in tqdm(enumerate(trainloader, 0)):
                inputs = get_torch_vars(inputs)

                # ============ Forward ============
                encoded, outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                # ============ Backward ============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ============ Logging ============
                running_loss += loss.data
                if i % 2000 == 1999:
                    print(
                        '[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        self.model = autoencoder

        for param in self.model.parameters():
            param.requires_grad = False

        pathlib.Path('ls.data/autoencoder').mkdir(parents=True, exist_ok=True)
        torch.save(self.model, '.data/autoencoder/model.pt')
        print('Successfully saved checkpoint.')


class PretrainedAutoencoder(nn.Module):
    def __init__(self, seed=42):
        super().__init__()

        try:
            'loading model'
            self.enc = torch.load('.data/autoencoder/model.pt')
        except:
            'creating model'
            self.enc = PretrainAutoencoder(seed=seed)

        for param in self.enc.parameters():
            param.requires_grad = False

        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()

    def forward(self, x):
        encoded, decoded = self.enc.forward(x)
        return encoded


model = PretrainedAutoencoder()
