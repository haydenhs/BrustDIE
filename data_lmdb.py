"""
SID dataset using lmdb format
"""

import os
import pickle
import numpy as np
import lmdb
import torch
import torch.utils.data as data


def get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = [int(s) for s in meta_info['resolution'].split('_')]
    return paths, sizes


def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


class SonyDataset(data.Dataset):
    def __init__(self, data_path, use_augment=False):
        self.paths, self.sizes = get_paths_from_lmdb(data_path)
        print("dataset contains {} pairs.".format(len(self.paths)))
        self.aug = use_augment
        if self.aug:
            print('Using data augmentation during training')
        self.env = lmdb.open(data_path, readonly=True, lock=False, readahead=False,
                            meminit=False)

    def __len__(self):
        if self.aug:
            return len(self.paths) * 8
        else:
            return len(self.paths)

    def __getitem__(self, index):
        if self.aug:
            opt = index % 8
            index = index // 8
        else:
            opt = 0
        key = self.paths[index]

        # read from lmdb
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        data_flat = np.frombuffer(buf, dtype=np.float32)
        N, H, W, C = self.sizes
        data = data_flat.reshape(N, H, W, C)
        gt = data[0]
        inputs = data[1:]

        # data augmentation
        gt = data_aug(gt, mode=opt)
        inputs = np.stack([data_aug(img, mode=opt) for img in inputs])

        # numpy to torch
        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2)
        gt = torch.from_numpy(gt).permute(2, 0, 1)

        return inputs, gt


if __name__ == "__main__":
    dataset = SonyDataset(data_path='/data/hsun/SIDSony/Sony_train.lmdb')
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    for inputs, gt in loader:
        print(inputs.shape)
        print(gt.shape)
        break
