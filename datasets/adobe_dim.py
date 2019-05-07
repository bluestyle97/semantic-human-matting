import cv2
import torch
import torch.utils.data as data

from utils.adobe_dim_utils import make_adobe_dim_dataset
import utils.adobe_dim_utils as extended_transforms

import warnings
warnings.filterwarnings('ignore')


class AdobeDIMDataset(data.Dataset):
    def __init__(self, root, mode, transforms=None):
        self.items = make_adobe_dim_dataset(root, mode)
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_name, trimap_name, alpha_name = self.items[index]
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        trimap = cv2.imread(trimap_name, cv2.IMREAD_GRAYSCALE)
        alpha = cv2.imread(alpha_name, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            for transform in self.transforms:
                image, trimap, alpha = transform(image, trimap, alpha)

        if self.mode == 'test':
            return image_name.split('/')[-1], image, trimap, alpha
        return image, trimap, alpha


class AdobeDIMDataLoader(object):
    def __init__(self, root, mode, batch_size):
        assert mode in ['pretrain_tnet', 'pretrain_mnet', 'end_to_end', 'test']

        if mode == 'pretrain_tnet':
            transforms = [
                extended_transforms.RandomPatch(400),
                extended_transforms.RandomFlip(),
                extended_transforms.Normalize(),
                extended_transforms.NumpyToTensor()
            ]
            train_set = AdobeDIMDataset(root, mode, transforms)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = len(train_set) // batch_size + 1
        elif mode == 'pretrain_mnet':
            transforms = [
                extended_transforms.RandomPatch(320),
                extended_transforms.RandomFlip(),
                extended_transforms.Normalize(),
                extended_transforms.TrimapToCategorical(),
                extended_transforms.NumpyToTensor()
            ]
            train_set = AdobeDIMDataset(root, mode, transforms)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = len(train_set) // batch_size + 1
        elif mode == 'end_to_end':
            transforms = [
                extended_transforms.RandomPatch(320),
                extended_transforms.RandomFlip(),
                extended_transforms.Normalize(),
                extended_transforms.NumpyToTensor()
            ]
            train_set = AdobeDIMDataset(root, mode, transforms)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = len(train_set) // batch_size + 1
        elif mode == 'test':
            transforms = [
                extended_transforms.Normalize(),
                extended_transforms.NumpyToTensor()
            ]
            test_set = AdobeDIMDataset(root, mode, transforms)
            self.test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.test_iterations = len(test_set) // batch_size + 1
        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass


if __name__ == '__main__':
    data_loader = AdobeDIMDataLoader('/root/datasets/Matting/Adobe_DIM', 'pretrain_mnet', 4).train_loader
    image, trimap, alpha = next(iter(data_loader))
    print(image.size(), trimap.size(), alpha.size())
    print(image)
    print(trimap)
    print(alpha)