import cv2
import os
import random
import numpy as np
import shutil
import torch
import warnings
warnings.filterwarnings('ignore')


def erode_dilate(mask, size=(10, 10), smooth=True):
    if smooth:
        size = (size[0]-4, size[1]-4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    dilated = cv2.dilate(mask, kernel, iterations=1)
    if smooth:  # if it is .jpg, prevent output to be jagged
        dilated[(dilated>5)] = 255
        dilated[(dilated <= 5)] = 0
    else:
        dilated[(dilated>0)] = 255

    eroded = cv2.erode(mask, kernel, iterations=1)
    if smooth:
        eroded[(eroded<250)] = 0
        eroded[(eroded >= 250)] = 255
    else:
        eroded[(eroded < 255)] = 0

    res = dilated.copy()
    res[((dilated == 255) & (eroded == 0))] = 128

    """# make sure there are only 3 values in trimap
    cnt0 = len(np.where(res >= 0)[0])
    cnt1 = len(np.where(res == 0)[0])
    cnt2 = len(np.where(res == 128)[0])
    cnt3 = len(np.where(res == 255)[0])
    assert cnt0 == cnt1 + cnt2 + cnt3
    """

    return res


def rand_trimap(mask, smooth=False):
    h, w = mask.shape
    scale_up, scale_down = 0.022, 0.006   # hyper parameter
    dmin = 0        # hyper parameter
    emax = 255 - dmin   # hyper parameter

    #  .jpg (or low quality .png) tend to be jagged, smoothing tricks need to be applied
    if smooth:
        # give thrshold for dilation and erode results
        scale_up, scale_down = 0.02, 0.006
        dmin = 5
        emax = 255 - dmin

        # apply gussian smooth
        if h < 1000:
            gau_ker = round(h*0.01)  # we restrict the kernel size to 5-9
            gau_ker = gau_ker if gau_ker % 2 ==1 else gau_ker-1  # make sure it's odd
            if h<500:
                gau_ker = max(3, gau_ker)
            mask = cv2.GaussianBlur(mask, (gau_ker, gau_ker), 0)

    kernel_size_high = max(10, round((h + w) / 2 * scale_up))
    kernel_size_low  = max(1, round((h + w) /2 * scale_down))
    erode_kernel_size  = np.random.randint(kernel_size_low, kernel_size_high)
    dilate_kernel_size = np.random.randint(kernel_size_low, kernel_size_high)

    erode_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    eroded_alpha = cv2.erode(mask, erode_kernel)
    dilated_alpha = cv2.dilate(mask, dilate_kernel)

    dilated_alpha = np.where(dilated_alpha > dmin, 255, 0)
    eroded_alpha = np.where(eroded_alpha < emax, 0, 255)

    res = dilated_alpha.copy()
    res[((dilated_alpha == 255) & (eroded_alpha == 0))] = 128

    return res


def generate_trimaps(mask_dir, trimap_dir, kernel_size):
    if not os.path.exists(trimap_dir):
        os.makedirs(trimap_dir)

    for filename in os.listdir(mask_dir):
        mask_name = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        trimap = erode_dilate(mask, (kernel_size, kernel_size), smooth=True)
        trimap_name = os.path.join(trimap_dir, filename.replace('jpg', 'png'))
        cv2.imwrite(trimap_name, trimap)


def copy_background_images(list_file, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    image_list = [l.strip('\n') for l in open(list_file, 'r').readlines()]
    for image in image_list:
        shutil.copyfile(os.path.join(source_dir, image), os.path.join(target_dir, image))


def make_adobe_dim_dataset(root, mode):
    assert mode in ['pretrain_tnet', 'pretrain_mnet', 'end_to_end', 'test']

    items = []
    # if mode == 'pretrain_tnet':
    #     fg_list = [l.strip('\n') for l in
    #                open(os.path.join(root, 'train_set', 'training_fg_names.txt'), 'r').readlines()]
    #     for fg in fg_list:
    #         trimap_name = os.path.join(root, 'train_set', 'trimap', fg.replace('jpg', 'png'))
    #         for j in range(100):
    #             img_name = os.path.join(root, 'train_set', 'merged', fg[:-4] + '_{}.png'.format(j))
    #             items.append((img_name, trimap_name))
    #
    # elif mode == 'pretrain_mnet' or mode == 'train':
    #     fg_list = [l.strip('\n') for l in
    #                open(os.path.join(root, 'train_set', 'training_fg_names.txt'), 'r').readlines()]
    #     bg_list = [l.strip('\n') for l in
    #                open(os.path.join(root, 'train_set', 'training_bg_names.txt'), 'r').readlines()]
    #     for i in range(len(fg_list)):
    #         fg = fg_list[i]
    #         fg_name = os.path.join(root, 'train_set', 'fg', fg)
    #         alpha_name = os.path.join(root, 'train_set', 'alpha', fg)
    #         trimap_name = os.path.join(root, 'train_set', 'trimap', fg.replace('jpg', 'png'))
    #         for j in range(100):
    #             bg_name = os.path.join(root, 'train_set', 'bg', bg_list[i * 100 + j])
    #             img_name = os.path.join(root, 'train_set', 'merged', fg[:-4] + '_{}.png'.format(j))
    #             items.append((img_name, trimap_name, alpha_name, fg_name, bg_name))
    #
    # elif mode == 'test':
    #     fg_list = [l.strip('\n') for l in
    #                open(os.path.join(root, 'test_set', 'test_fg_names.txt'), 'r').readlines()]
    #     for fg in fg_list:
    #         alpha_name = os.path.join(root, 'test_set', 'alpha', fg)
    #         for j in range(20):
    #             img_name = os.path.join(root, 'test_set', 'merged', fg[:-4] + '_{}.png'.format(j))
    #             items.append((img_name, alpha_name))
    #
    # else:
    #     raise Exception('Please choose proper mode for data')

    if mode == 'pretrain_tnet' or mode == 'pretrain_mnet' or mode == 'end_to_end':
        fg_list = [l.strip('\n') for l in
                   open(os.path.join(root, 'train_set', 'training_fg_names.txt'), 'r').readlines()]
        for fg in fg_list:
            trimap_name = os.path.join(root, 'train_set', 'trimap', fg.replace('jpg', 'png'))
            alpha_name = os.path.join(root, 'train_set', 'alpha', fg)
            for j in range(100):
                img_name = os.path.join(root, 'train_set', 'merged', fg[:-4] + '_{}.png'.format(j))
                items.append((img_name, trimap_name, alpha_name))
    elif mode == 'test':
        fg_list = [l.strip('\n') for l in
                   open(os.path.join(root, 'test_set', 'test_fg_names.txt'), 'r').readlines()]
        for fg in fg_list:
            trimap_name = os.path.join(root, 'test_set', 'trimap', fg.replace('jpg', 'png'))
            alpha_name = os.path.join(root, 'test_set', 'alpha', fg)
            for j in range(20):
                img_name = os.path.join(root, 'test_set', 'merged', fg[:-4] + '_{}.png'.format(j))
                items.append((img_name, trimap_name, alpha_name))
    else:
        raise Exception('Please choose proper mode for data')

    return items


class RandomPatch(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, image, trimap, alpha):
        # random scale
        if random.random() < 0.5:
            h, w, c = image.shape
            scale = 0.75 + 0.5 * random.random()
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            trimap = cv2.resize(trimap, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
            alpha = cv2.resize(alpha, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        # creat patch
        if random.random() < 0.5:
            h, w, c = image.shape
            if h > self.patch_size and w > self.patch_size:
                x = random.randrange(0, w - self.patch_size)
                y = random.randrange(0, h - self.patch_size)
                image = image[y:y + self.patch_size, x:x + self.patch_size, :]
                trimap = trimap[y:y + self.patch_size, x:x + self.patch_size]
                alpha = alpha[y:y + self.patch_size, x:x + self.patch_size]
            else:
                image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
                trimap = cv2.resize(trimap, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                alpha = cv2.resize(alpha, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            trimap = cv2.resize(trimap, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            alpha = cv2.resize(alpha, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

        return image, trimap, alpha


class RandomFlip(object):
    def __call__(self, image, trimap, alpha):
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            trimap = cv2.flip(trimap, 0)
            alpha = cv2.flip(alpha, 0)

        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            trimap = cv2.flip(trimap, 1)
            alpha = cv2.flip(alpha, 1)
        return image, trimap, alpha


class Normalize(object):
    def __call__(self, image, trimap, alpha):
        image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
        trimap[trimap == 0] = 0
        trimap[trimap == 128] = 1
        trimap[trimap == 255] = 2
        alpha = alpha.astype(np.float32) / 255.0
        return image, trimap, alpha


class TrimapToCategorical(object):
    def __call__(self, image, trimap, alpha):
        trimap = np.array(trimap, dtype=np.int)
        input_shape = trimap.shape
        trimap = trimap.ravel()
        n = trimap.shape[0]
        categorical = np.zeros((3, n), dtype=np.long)
        categorical[trimap, np.arange(n)] = 1
        output_shape = (3,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return image, categorical, alpha

class NumpyToTensor(object):
    def __call__(self, image, trimap, alpha):
        h, w, c = image.shape
        image = torch.from_numpy(image.transpose((2, 0, 1))).view(c, h, w).float()
        trimap = torch.from_numpy(trimap).view(-1, h, w).long()
        alpha = torch.from_numpy(alpha).view(1, h, w).float()
        return image, trimap, alpha


if __name__ == '__main__':
    trimap = np.array([[0, 2],
                       [1, 0]])
    _, trimap_categorical, _ = TrimapToCategorical()(None, trimap, None)
    print(trimap_categorical)
