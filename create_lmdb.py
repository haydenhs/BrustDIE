"""Create lmdb file for SID dataset"""

import sys
import os
import glob
import pickle
import numpy as np
import lmdb
import rawpy
from tqdm import tqdm


def load_raw(folder_path, file_name):
    """ Load a raw image and set wb using LibCam """
    raw = rawpy.imread(folder_path + file_name)
    img = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    img = np.float32(img / 65535.)
    return img


def main():
    dataset = 'Sony'
    input_dir = '/data/hsun/SIDSony/Sony/short/'
    gt_dir = '/data/hsun/SIDSony/Sony/long/'
    lmdb_save_path = '/data/hsun/SIDSony/Sony_train.lmdb'

    if dataset == 'Sony':
        Sony(input_dir, gt_dir, lmdb_save_path)
    elif dataset == 'Fuji':
        Fuji()
    elif dataset == 'test':
        test_lmdb('/data/hsun/SIDSony/Sony_train.lmdb', 'Sony')


def Sony(input_dir, gt_dir, lmdb_save_path):
    # possible options
    fns = glob.glob(gt_dir + '0*.ARW')
    ids = [os.path.basename(fn)[0:5] for fn in fns]

    # find corresponding file names
    gt_images = []
    input_images = []

    for img_id in ids:
        gt_path = glob.glob(self.gt_dir + img_id + '*.ARW')
        gt_img = os.path.basename(gt_path[0])
        # try all 3 exposure time options
        imgs = glob.glob(input_dir + img_id + '_0*_0.1s.ARW')
        # we only use 10 burst imgs
        if len(imgs) == 10:
            gt_images.append(gt_img)
            input_images.append([os.path.basename(img) for img in imgs])
        imgs = glob.glob(input_dir + img_id + '_0*_0.04s.ARW')
        if len(imgs) == 10:
            gt_images.append(gt_img)
            input_images.append([os.path.basename(img) for img in imgs])
        imgs = glob.glob(input_dir + img_id + '_0*_0.033s.ARW')
        if len(imgs) == 10:
            gt_images.append(gt_img)
            input_images.append([os.path.basename(img) for img in imgs])

    # save data pairs to lmdb
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    # create lmdb environment
    env = lmdb.open(lmdb_save_path, map_size=1099511627776)

    # write data to lmdb
    txn = env.begin(write=True)
    for idx, (in_fns, gt_fn) in enumerate(tqdm(zip(input_images, gt_images))):

        key = gt_fn[:-5]
        key_byte = key.encode('ascii')

        # read imgs
        gt = load_raw(gt_dir, gt_fn)
        inputs = []
        for fn in in_fns:
            inputs.append(load_raw(input_dir, fn))
        inputs = np.stack(inputs, axis=0)

        txn.put(key_byte, (inputs, gt))
        txn.commit()
        txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    # create meta information
    meta_info = {}
    meta_info['name'] = 'SIDSony_brust_{}_train'.format(10)
    meta_info['keys'] = gt_images
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def Fuji():
    pass


def test_lmdb(dataroot, dataset='Sony'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('# keys: ', len(meta_info['keys']))
    # read one image
    if dataset == 'Sony':
        key = '00001_00_10'
    else:
        key = '00000_00_00'
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    inputs, gt = np.frombuffer(buf, dtype=np.float32)
    print(inputs.shape)
    print(gt.shape)


if __name__ == "__main__":
    main()
