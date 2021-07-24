from PIL import Image
from torchvision import transforms


def build_transforms(isTrain=True):
    cur_tfs = []

    if isTrain:
        cur_tfs = [
            lambda x: Image.fromarray(x.astype('uint8')).convert('RGB'),
            transforms.RandomCrop(32, padding=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    else:
        cur_tfs = [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    
    return transforms.Compose(cur_tfs)
