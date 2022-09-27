import torchvision as tv
from os.path import join as pjoin  # pylint: disable=g-importing-member


# Preprocessing code from BiT (Google)
# https://github.com/google-research/big_transfer/blob/master/bit_pytorch/train.py

def get_resolution(original_resolution):
    """Takes (H,W) and returns (precrop, crop)."""
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96 * 96 else (512, 480)


known_dataset_sizes = {
    'cifar10': (32, 32),
    'cifar100': (32, 32),
    'oxford_iiit_pet': (224, 224),
    'oxford_flowers102': (224, 224),
    'imagenet2012': (224, 224),
}


def get_resolution_from_dataset(dataset):
    if dataset not in known_dataset_sizes:
        raise ValueError(
            f"Unsupported dataset {dataset}. Add your own here :)")
    return get_resolution(known_dataset_sizes[dataset])


def get_mixup(dataset_size):
    return 0.0 if dataset_size < 20_000 else 0.1


def mktrainval(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = get_resolution_from_dataset(args.dataset)
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize((precrop, precrop)),
        tv.transforms.RandomCrop((crop, crop)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.dataset == "cifar10":
        train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx,
                                        train=True, download=True)
        valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx,
                                        train=False, download=True)
    elif args.dataset == "cifar100":
        train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx,
                                         train=True, download=True)
        valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx,
                                         train=False, download=True)
    elif args.dataset == "imagenet2012":
        train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"),
                                            train_tx)
        valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
    else:
        raise ValueError(f"Sorry, we have not spent time implementing the "
                         f"{args.dataset} dataset in the PyTorch codebase. "
                         f"In principle, it should be easy to add :)")

    if args.examples_per_class is not None:
        logger.info(
            f"Looking for {args.examples_per_class} images per class...")
        indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
        train_set = torch.utils.data.Subset(train_set, indices=indices)

    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")


def load_image_dataloaders(c):
    batch_size = c.exp_batch_size
    from ttnn.utils.image_loading_utils import get_dataloaders

    if c.data_set in ['cifar10', 'cifar100']:
        # For CIFAR, let's just use 10% of the training set for validation.
        # That is, 10% of 50,000 rows = 5,000 rows
        val_perc = 0.10
    else:
        raise NotImplementedError

    _, trainloader, validloader, testloader = get_dataloaders(
        c.data_set, batch=batch_size, dataroot=f'{c.data_path}/{c.data_set}',
        c=c, split=val_perc, split_idx=0)
    data_dict = {
        'trainloader': trainloader,
        'validloader': validloader,
        'testloader': testloader}

    return data_dict
