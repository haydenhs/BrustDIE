"""Create lmdb file for SID dataset"""
# Sony Raw image size 4256x2848
import sys
import os
import glob
import pickle
import numpy as np
import lmdb
import rawpy
import cv2
from tqdm import tqdm


def load_raw(folder_path, file_name):
    """ Load a raw image process using LibCam """
    raw = rawpy.imread(folder_path + file_name)
    img = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    img = np.float32(img / 65535.)
    return img


def crop_image(img, win, s):
    """
    Crop an image into patches

    :param img: an image of HxWxC
    :param win: crop patch size
    :param s: crop stride
    :return: patches (num_of_patches x patch_size x patch_size x C)
    """

    h, w, c = img.shape

    pat_row_num = list(range(0, h - win, s))
    pat_row_num.append(h - win)
    pat_col_num = list(range(0, w - win, s))
    pat_col_num.append(w - win)

    patches = np.zeros((len(pat_row_num)*len(pat_col_num), win, win, c), dtype='float32')

    num = 0

    for i in pat_row_num:
        for j in pat_col_num:
            up = i
            down = up + win
            left = j
            right = left + win
            patches[num, :, :, :] = img[up:down, left:right, :]
            num += 1

    return patches


def main():
    dataset = 'Sony'
    input_dir = '/data/hsun/SIDSony/Sony/short/'
    gt_dir = '/data/hsun/SIDSony/Sony/long/'
    save_path = '/data/hsun/SIDSony/Sony_train.lmdb'
    ps = 512

    if dataset == 'Sony':
        Sony(input_dir, gt_dir, patch_size=ps, stride=ps, lmdb_save_path=save_path)
    elif dataset == 'Fuji':
        Fuji()
    elif dataset == 'test':
        test_lmdb(save_path, 'Sony')


def Sony(input_dir, gt_dir, patch_size, stride, lmdb_save_path):
    """
    Create Sony SID dataset in lmdb format.

    :param input_dir: input sony short exporsure times
    :param gt_dir:  input sony long exposure times
    :param patch_size:  patch size to be cropped
    :param stride:  stride between patches
    :param lmdb_save_path: lmdb file save path
    :return:
    """
    # possible options
    fns = glob.glob(gt_dir + '0*.ARW')
    ids = [os.path.basename(fn)[0:5] for fn in fns]

    # find corresponding file names
    gt_images = []
    input_images = []

    for img_id in ids:
        gt_path = glob.glob(gt_dir + img_id + '*.ARW')
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
    dummy = load_raw(gt_dir, gt_images[0])
    H, W, C = dummy.shape
    data_size_per_img = dummy.nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(gt_images) * 11
    env = lmdb.open(lmdb_save_path, map_size=data_size*10)

    # write data to lmdb
    txn = env.begin(write=True)
    keys = []
    for idx, in_fns in enumerate(tqdm(input_images)):

        # read imgs and crop patches
        data = []
        gt_fn = gt_images[idx]
        gt = load_raw(gt_dir, gt_fn)
        gt_patches = crop_image(gt, win=patch_size, s=stride)
        data.append(gt_patches)
        for fn in in_fns:
            frame = load_raw(input_dir, fn)
            assert frame.shape[0] == H and frame.shape[1] == W
            data.append(crop_image(frame, win=patch_size, s=stride))
        assert len(data) == 11
        data = np.stack(data, axis=1)

        # save patches to lmdb
        num_of_patches = gt_patches.shape[0]
        print("Image id {} contains {} patches.".format(gt_fn[:-5], num_of_patches))
        for i in range(num_of_patches):

            key = gt_fn[:-5] + '_' + str(i).zfill(3)
            keys.append(key)
            key_byte = key.encode('ascii')
            txn.put(key_byte, data[i])
        # commit every when a full-size image
        txn.commit()
        txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb. Dataset contains {} pairs'.format(len(keys)))

    # create meta information
    meta_info = {}
    meta_info['name'] = 'SIDSony_brust{}_train'.format(10)
    meta_info['resolution'] = '{}_{}_{}_{}'.format(11, patch_size, patch_size, C)
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(os.path.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def Fuji():
    pass


def test_lmdb(dataroot, dataset='Sony'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))

    # read one image
    if dataset == 'Sony':
        key = '00001_00_10_' + str(0).zfill(3)
    else:
        key = '00000_00_00'
    print('Reading {} for test.'.format(key))

    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    data_flat = np.frombuffer(buf, dtype=np.float32)
    N, H, W, C = [int(s) for s in meta_info['resolution'].split('_')]
    data = data_flat.reshape(N, H, W, C)
    gt = data[0]
    inputs = data[1:]
    print('inputs shape:', inputs.shape)
    print('gt shape:', gt.shape)
    test = np.uint8(np.clip(gt*255, 0, 255))
    test = cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test.png', test)


if __name__ == "__main__":
    main()
