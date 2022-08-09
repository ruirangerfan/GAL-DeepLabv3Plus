import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
from data.base_dataset import BaseDataset


class potholedataset(BaseDataset):
    """dataloader for pothole dataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_labels = 2

        if opt.phase == "train":
            self.image_list = np.arange(1, 43)
        elif opt.phase == "val":
            self.image_list = np.arange(43, 56)
        else:
            self.image_list = np.arange(43, 56)

    def __getitem__(self, index):
        base_dir = "./datasets/pothole"
        name = str(self.image_list[index]).zfill(2) + ".png"

        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(base_dir, 'rgb', name)), cv2.COLOR_BGR2RGB)
        tdisp_image = cv2.imread(os.path.join(base_dir, 'tdisp', name), cv2.IMREAD_ANYDEPTH)
        label_image = cv2.cvtColor(cv2.imread(os.path.join(base_dir, 'label', name)), cv2.COLOR_BGR2RGB)

        label = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.uint8)
        label[label_image[:, :, 0] > 0] = 1

        rgb_image = rgb_image.astype(np.float32) / 255
        tdisp_image = tdisp_image.astype(np.float32) / 65535
        rgb_image = transforms.ToTensor()(rgb_image)
        tdisp_image = transforms.ToTensor()(tdisp_image)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, tdisp images, and labels for training;
        # 'path': image name for saving predictions
        return {'rgb_image': rgb_image, 'tdisp_image': tdisp_image, 'label': label,
                'path': name}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'pothole'
