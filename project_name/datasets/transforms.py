import torchvision.transforms as transforms


def train_transforms():
    return transforms.ToTensor()


def eval_transforms():
    return transforms.ToTensor()
