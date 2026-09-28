import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def print_mean_std(name, dataset):
    imgs = [item[0] for item in dataset]
    imgs = torch.stack(imgs, dim=0).numpy()

    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean()
    mean_b = imgs[:, 2, :, :].mean()

    std_r = imgs[:, 0, :, :].std()
    std_g = imgs[:, 1, :, :].std()
    std_b = imgs[:, 2, :, :].std()

    print(name)
    print("mean = [{:.8f}, {:.8f}, {:.8f}]".format(mean_r, mean_g, mean_b))
    print("std  = [{:.8f}, {:.8f}, {:.8f}]".format(std_r, std_g, std_b))


def main():
    cifar_trainset = dset.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    print_mean_std("CIFAR100 train", cifar_trainset)

    svhn_trainset = dset.SVHN(
        root='./data',
        split='train',
        download=True,
        transform=transforms.ToTensor()
    )
    print_mean_std("SVHN train", svhn_trainset)


if __name__ == '__main__':
    main()
