from glob import glob
import cv2
import numpy as np
from pycocotools.coco import COCO

import os

import torch
import torch.distributed as dist

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class GlobDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        pathname=None,
        img_size=(608, 1088),
        preproc=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)

        rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
        ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
        self.img_list = sorted(glob(pathname))[rank::ws]
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.img_list)

    def pull_item(self, index):
        img_file = self.img_list[index]
        img = cv2.imread(img_file)
        assert img is not None
        return img, {}, torch.tensor(img.shape[:2]), torch.tensor(index)

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id


def get_glob_loader(args, pathname, batch_size, is_distributed, testdev=False):
    from yolox.data import ValTransform

    valdataset = GlobDataset(
        pathname=pathname,
        img_size=args.input_size,
        preproc=ValTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    )

    if is_distributed:
        batch_size = batch_size // dist.get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            valdataset, shuffle=False
        )
    else:
        sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
        "num_workers": args.data_num_workers,
        "pin_memory": True,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = batch_size
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    return val_loader
