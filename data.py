import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import rawpy


class SonyDataset(Dataset):

    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps

        # files start with 0 are data for training
        fns = glob.glob(gt_dir + '0*.ARW')
        # find ids
        ids = [os.path.basename(fn)[0:5] for fn in fns]

        self.gt_images = []
        self.input_images = []

        for img_id in ids:
            gt_path = glob.glob(self.gt_dir + img_id + '*.ARW')
            gt_img = os.path.basename(gt_path[0])
            imgs = glob.glob(input_dir + img_id + '_0*_0.1s.ARW')
            # we only use 10 burst imgs
            if len(imgs) == 10:
                self.gt_images.append(gt_img)
                self.input_images.append([os.path.basename(img) for img in imgs])
            imgs = glob.glob(input_dir + img_id + '_0*_0.04s.ARW')
            if len(imgs) == 10:
                self.gt_images.append(gt_img)
                self.input_images.append([os.path.basename(img) for img in imgs])
            imgs = glob.glob(input_dir + img_id + '_0*_0.033s.ARW')
            if len(imgs) == 10:
                self.gt_images.append(gt_img)
                self.input_images.append([os.path.basename(img) for img in imgs])

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, ind):

        gt_fn = self.gt_images[ind]
        print(gt_fn)
        input_fns = self.input_images[ind]
        assert len(input_fns) == 10

        gt_raw = rawpy.imread(self.gt_dir + gt_fn)
        gt = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # normalize and crop patch
        gt = np.float32(gt / 65535.)
        H, W = gt.shape[0], gt.shape[1]
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        gt_patch = gt[yy:yy + self.ps, xx:xx + self.ps, :]

        # same operations on inputs
        input_patches = []
        for input_fn in input_fns:
            input_raw = rawpy.imread(self.input_dir + input_fn)
            input = input_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            input = np.float32(input / 65535.)
            input_patches.append(input[yy:yy + self.ps, xx:xx + self.ps, :])

        input_patches = np.stack(input_patches, axis=0)

        # todo: add data augmentation

        gt_patch = torch.from_numpy(gt_patch)
        gt_patch = gt_patch.permute(2, 0, 1)

        input_patches = torch.from_numpy(input_patches)
        input_patches = input_patches.permute(0, 3, 1, 2)

        return input_patches, gt_patch


class SonyTestDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir
        self.gt_dir = gt_dir

        self.fns = glob.glob(input_dir + '1*.ARW')  # file names, 1 for testing.

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        # input
        id = self.ids[ind]
        in_path = self.fns[ind]
        in_fn = os.path.basename(in_path)
        # ground truth
        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        # ratio
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        # load images
        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # clipping, convert to tensor
        input_full = np.minimum(input_full, 1.0)
        input_full = torch.from_numpy(input_full)
        input_full = torch.squeeze(input_full)
        input_full = input_full.permute(2, 0, 1)

        scale_full = torch.from_numpy(scale_full)
        scale_full = torch.squeeze(scale_full)

        gt_full = torch.from_numpy(gt_full)
        gt_full = torch.squeeze(gt_full)
        return input_full, scale_full, gt_full, id, ratio


if __name__ == "__main__":
    dataset = SonyDataset(input_dir='./testdata/short/', gt_dir='./testdata/long/')
    print("Total training pairs in dataset is ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    for inputs, gt in loader:
        print(inputs.shape)
        print(gt.shape)
        break
