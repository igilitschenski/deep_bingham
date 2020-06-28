from .utils import *
from torch.utils.data import Dataset, random_split, Subset
import yaml
import os

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from PIL import Image
import numpy as np
from skimage import io
import torch
import quaternion
import cv2
import h5py

torch.manual_seed(0)


def make_hdf5_file(config, image_transform):
    dataset_path = config["dataset_path"]
    if config["dirs"]:
        dirs = config["dirs"]
        tless_full = TLessSplit(
            TLessFullDataset(dataset_path, dirs, image_transform), shuffle=config.get("shuffle", True))
        train = tless_full.train
        test = tless_full.test
    else:
        train_dirs = config["train_dirs"]
        test_dirs = config["test_dirs"]
        train = TLessFullDataset(dataset_path, train_dirs, image_transform)
        test = TLessFullDataset(dataset_path, test_dirs, image_transform)

    file_dataset = config["hdf5"]

    img_shape = train[0]["image"].shape
    label_shape = train[0]["pose"].shape[-1]
    f = h5py.File("datasets/{}".format(file_dataset), "w")

    train_img = f.create_dataset("train_img", (
        len(train), img_shape[0], img_shape[1], img_shape[2]))
    train_label = f.create_dataset("train_label", (len(train), label_shape))
    test_img = f.create_dataset("test_img", (
        len(test), img_shape[0], img_shape[1], img_shape[2]))
    test_label = f.create_dataset("test_label", (len(test), label_shape))

    print("Making HDF5")
    for i in range(len(train)):
        f["train_img"][i, :, :, :] = train[i]["image"]
        f["train_label"][i, :] = train[i]["pose"]

    for i in range(len(test)):
        f["test_img"][i, :, :, :] = test[i]["image"]
        f["test_label"][i, :] = test[i]["pose"]


class TLessTrainTest():
    """
        Stores a training and test set for the TLess Dataset

        Parameters:
        config_file: a yaml file or dictionary that contains data loading
            information ex. See configs/upna_train.yaml. The dataset_path stores
            the locsation of the originial downloaded dataset. The
            preprocess_path is where the processed images and poses will be
            stored.

        image_transforms: A list of of composed pytorch transforms to be applied
            to a PIL image
    """

    def __init__(self, config_file, image_transform=None):
        if type(config_file) == dict:
            config = config_file
        else:
            with open(config_file) as fp:
                config = yaml.load(fp)
        file_dataset = config["hdf5"]
        if not os.path.isfile("datasets/{}".format(file_dataset)):
            make_hdf5_file(config_file, image_transform)
        f = h5py.File("datasets/{}".format(file_dataset), 'r')

        biterion = config["biterion"]
        blur = config["blur"]
        self.train = TLessHDF5(f.get('train_img'), f.get('train_label'),
                               f.get("train_bb"),
                               biterion, blur)
        self.test = TLessHDF5(f.get('test_img'), f.get('test_label'),
                              f.get("test_bb"), biterion, blur)


class TLessHDF5(Dataset):
    """
        Loads TLess dataset from a HDF5 dataset and applies transformations to
        biterion or quaternion form and adds noise to the labels.
        biterion: format of the pose. if true, biterion. if false, quaternion.
    """
    def __init__(self, images, labels, bb, biterion, blur):
        self.images = images
        self.labels = labels
        self.bb = bb
        self.biterion = biterion
        self.blur = blur

    def __getitem__(self, idx):
        image = self.images[idx, :, :, :]
        pose = self.labels[idx, :]
        if self.blur:
            size = 10
            kernel = np.ones((size, size), np.float32) / size ** 2
            blurred_img = cv2.filter2D(image, -1, kernel)
            image = blurred_img
        if self.biterion:
            convert_to_rad = quaternion_to_euler(pose[0], pose[1], pose[2],
                                                 pose[3])

            sample = {'image': torch.from_numpy(image),
                      'pose': torch.Tensor([math.degrees(convert_to_rad[0]),
                                            math.degrees(convert_to_rad[1]),
                                            math.degrees(convert_to_rad[2])])}
        else:
            sample = {'image': torch.from_numpy(image),
                      'pose': torch.Tensor(pose)}
        return sample

    def __len__(self):
        return self.images.shape[0]


class TLessSplit(object):
    def __init__(self, dataset, shuffle=True):
        train_size = int(len(dataset) * 0.75)
        if shuffle:
            self.train, self.test = random_split(dataset, [train_size, len(
            dataset) - train_size])
        else:
            self.train = Subset(dataset, list(range(train_size)))
            self.test = Subset(dataset, list(range(train_size, len(dataset))))
        

class TLessFullDataset(Dataset):

    def __init__(self, path, dirs, image_transform):
        self.subdatasets = []
        self.size = [0]
        self.image_transform = image_transform
        for i in range(len(dirs)):
            self.subdatasets.append(
                TLessSingleDataset(path, dirs[i], self.image_transform))
            self.size.append(len(self.subdatasets[i]) + self.size[-1])

    def __getitem__(self, idx):
        data_bin = 0
        if not type(idx) == int:
            idx = idx.item()
        for i in range(1, len(self.size)):
            if self.size[i] > idx >= self.size[i - 1]:
                data_bin = i - 1
        new_index = idx - self.size[data_bin]
        return self.subdatasets[data_bin][new_index]

    def __len__(self):
        return self.size[-1]


class TLessSingleDataset(Dataset):
    def __init__(self, path, direc, image_transform):
        self.dir_to_gt = {}

        self.full_path = os.path.join(path, direc)
        with open(self.full_path + "/gt.yml") as fp:
            self.dir_to_gt = yaml.load(fp, Loader=Loader)
        self.size = len(self.dir_to_gt.keys())
        self.image_transform = image_transform

    def __getitem__(self, index):
        name_img = str(index).zfill(4)
        img_path = os.path.join(self.full_path, "rgb",
                                "{}.png".format(name_img))
        bb = np.array(self.dir_to_gt[index][0]["obj_bb"])
        image = Image.fromarray(
            io.imread(img_path)[int(bb[1]): int(bb[1] + bb[3]),
            int(bb[0]): int(bb[0] + bb[2]), :])
        pose = np.array(self.dir_to_gt[index][0]["cam_R_m2c"]).reshape(3, 3)
        if self.image_transform:
            image = self.image_transform(image).numpy()
        pose = rotation_matrix_to_quaternion(pose)
        assert (sum(np.array(self.dir_to_gt[index][0]["obj_bb"])) != 0)
        return {"image": image, "pose": torch.Tensor(pose)}

    def __len__(self):
        return self.size


def rotation_matrix_to_quaternion(rot_mat):
    quat = quaternion.as_float_array(quaternion.from_rotation_matrix(rot_mat))
    return quat
