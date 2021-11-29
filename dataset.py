import torchvision
import torchvision.transforms as transforms
import torch

class dataset():
    def __init__(self, dataset, batch_size):
        if dataset == 'mnist':
            self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    ]))
            self.test_dataset = torchvision.datasets.MNIST(root='./data',
                train=False,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    ]))
            
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size = batch_size, shuffle=False)
        elif dataset == 'cifar10':
            # transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transform = transforms.Compose(
                [transforms.ToTensor()])
            
            self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=0)
            
            self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size,
                                                     shuffle=False, num_workers=0)
        else:
            assert False