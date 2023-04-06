import random
import time
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.data_util import paths_from_folder

import cv2
import numpy as np

# @DATASET_REGISTRY.register()
class FFHQAugDataset(data.Dataset):
    """FFHQ dataset for StyleGAN.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.

    """

    def __init__(self, opt):
        super(FFHQAugDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', " f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # FFHQ has 70000 images in total
            # self.paths = [osp.join(self.gt_folder, f'{v:08d}.png') for v in range(70000)]
            self.paths = paths_from_folder(self.gt_folder)

        self.gray_prob = opt.get('gray_prob')

        self.exposure_prob = opt.get('exposure_prob', 0.)
        self.exposure_range = opt['exposure_range']

        self.shift_prob = opt.get('shift_prob', 0.)
        self.shift_unit = opt.get('shift_unit', 32)
        self.shift_max_num = opt.get('shift_max_num', 3)

        logger = get_root_logger()
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')

        if self.exposure_prob is not None:
            logger.info(f'Use random exposure. Prob: {self.exposure_prob}')
            logger.info(f'Use random exposure. Range: [{self.exposure_range[0]}, {self.exposure_range[1]}]')

        if self.shift_prob is not None:
            logger.info(f'Use random shift. Prob: {self.shift_prob}')
            logger.info(f'Use random shift. uint: {self.shift_unit}')
            logger.info(f'Use random shift. max_num: {self.shift_max_num}')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)
        h, w, _ = img_gt.shape

        if (self.exposure_prob is not None) and (np.random.uniform() < self.exposure_prob):
                exp_scale = np.random.uniform(self.exposure_range[0], self.exposure_range[1])
                img_gt *= exp_scale

        if (self.shift_prob is not None) and (np.random.uniform() < self.shift_prob ):
            # self.shift_unit = 32
            # import pdb
            # pdb.set_trace()
            shift_vertical_num = np.random.randint(0, self.shift_max_num*2+1)
            shift_horisontal_num = np.random.randint(0, self.shift_max_num*2+1)
            shift_v = self.shift_unit * shift_vertical_num
            shift_h = self.shift_unit * shift_horisontal_num
            img_gt_pad = np.pad(img_gt, ((self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit), (0,0)), 
                                mode='symmetric')
            img_gt = img_gt_pad[shift_v:shift_v + h, shift_h: shift_h + w,:]

        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        return {'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

import argparse
from omegaconf import OmegaConf
import pdb
from basicsr.utils import img2tensor, imwrite, tensor2img

if __name__=='__main__':
    # pdb.set_trace()
    base='configs/ROHQD.yaml'

    opt = OmegaConf.load(base)
    dataset = FFHQAugDataset(opt['data']['params']['train']['params'])

    for i in range(14):
        sample = dataset.getitem(i)