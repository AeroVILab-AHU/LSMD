import os
import cv2
import numpy
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_root='data/', transform=None):
        # def __init__(self, dataset="mcd", split="train", file_root='data/', transform=None):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        # self.file_list = open(file_root + '/' + dataset + '/list/' + dataset + '.txt').read().splitlines()
        # self.pre_images = [file_root + '/' + dataset + '/RGB_A/' + x for x in self.file_list]
        # self.post_images = [file_root + '/' + dataset + '/RGB_B/' + x for x in self.file_list]
        # self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        # self.transform = transform
        # print("List file used:", file_root + '/' + dataset + '/list/' + dataset + '.txt')
        # print("First 5 pre_images:", self.pre_images[:5])
        # print("First 5 post_images:", self.post_images[:5])
        # print("First 5 gts:", self.gts[:5])

        list_file = os.path.join(file_root, dataset + '.txt')
        self.file_list = open(list_file).read().splitlines()
        self.pre_images = [os.path.join(file_root, 'A', x) for x in self.file_list]
        self.post_images = [os.path.join(file_root, 'B', x) for x in self.file_list]
        self.gts = [os.path.join(file_root, 'label', x) for x in self.file_list]

        # ---mask
        self.mask_A = [os.path.join(file_root, 'mask_A', x.replace('.tif', '.png')) for x in
                       self.file_list]
        self.mask_B = [os.path.join(file_root, 'mask_B', x.replace('.tif', '.png')) for x in
                       self.file_list]

        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]

        pre_image = cv2.imread(pre_image_name, cv2.IMREAD_UNCHANGED)  # 保留原始通道
        label = cv2.imread(label_name, 0)  # 单通道 mask
        post_image = cv2.imread(post_image_name, cv2.IMREAD_UNCHANGED)

        rgb_pre_image = pre_image[:, :, :3]  # 只取前三个通道（RGB）
        rgb_post_image = post_image[:, :, :3]

        # 取第4通道 (NIR)，用 3: 保留三维 (H, W, 1)
        nir_pre_image = pre_image[:, :, 3:]
        nir_post_image = post_image[:, :, 3:]

        # nir_pre_image = numpy.repeat(nir_pre_image[:, :, None], 3, axis=2)  # (H, W, 3)
        # nir_post_image = numpy.repeat(nir_post_image[:, :, None], 3, axis=2)  # (H, W, 3)

        nir_pre_image = numpy.repeat(nir_pre_image, 3, axis=2)  # (H, W, 3)
        nir_post_image = numpy.repeat(nir_post_image, 3, axis=2)  # (H, W, 3)

        rgb_img = numpy.concatenate((rgb_pre_image, rgb_post_image), axis=2)
        nir_img = numpy.concatenate((nir_pre_image, nir_post_image), axis=2)
        img = numpy.concatenate((rgb_img, nir_img), axis=2)
        # --- RemoteSAM mask 读取 ---
        mask_A = cv2.imread(self.mask_A[idx], 0)  # 单通道
        mask_B = cv2.imread(self.mask_B[idx], 0)  # 单通道

        # if self.transform:
        #     [pre_image, label, post_image] = self.transform(pre_image, label, post_image)
        #
        # return pre_image, label, post_image
        if self.transform:
            [img, label] = self.transform(img, label)
            # 保证 mask 和 label 尺寸一致（用 label 的 transform 结果来对齐）
            mask_A = cv2.resize(mask_A, (label.shape[-1], label.shape[-2]), interpolation=cv2.INTER_NEAREST)
            mask_B = cv2.resize(mask_B, (label.shape[-1], label.shape[-2]), interpolation=cv2.INTER_NEAREST)
        # --- 转 tensor ---
        mask_A = torch.from_numpy(mask_A).unsqueeze(0).float() / 255.0
        mask_B = torch.from_numpy(mask_B).unsqueeze(0).float() / 255.0
        # change_mask = ((mask_A > 0.5) ^ (mask_B > 0.5)).float()
        # return img, label, change_mask
        return img, label, mask_A, mask_B

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
