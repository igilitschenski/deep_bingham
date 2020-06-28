"""
Data loading methods from matlab file from:
https://github.com/lucasb-eyer/BiternionNet
"""
import os
import h5py
import yaml
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from .utils import *
from bingham_distribution import BinghamDistribution

class IDIAPTrainTest(object):
    """
        Stores a training and test set for the IDIAP Head Pose Dataset

        Parameters:
        config_file: a yaml file or dictionary that contains data loading
            information ex. See configs/upna_train.yaml The dataset_path stores
            the locsation of the originial downloaded dataset. The
            preprocess_path is where the processed images and poses will be
            stored.

        image_transforms: A list of of composed pytorch transforms to be applied to a PIL image

    """

    def __init__(self, config_file, image_transform=None):
        if type(config_file) == dict:
            config = config_file
        else:
            with open(config_file) as fp:
                config = yaml.load(fp)
        self.dataset_path = config["dataset_path"]
        mat_file = self.dataset_path + "/or_label_full.mat"
        self.image_transform = image_transform
        pose_file = h5py.File(mat_file)

        train_table = load("train", pose_file)
        test_table = load("test", pose_file)
        euler_noise = config["euler_noise"]
        biterion = config["biterion"]
        quat_noise = config["quat_noise"]
        self.train = IDIAPSplitSet(train_table, image_transform,
                                   self.dataset_path + "/train", euler_noise,
                                   quat_noise,
                                   biterion)
        self.test = IDIAPSplitSet(test_table, image_transform,
                                  self.dataset_path + "/test", euler_noise,
                                  quat_noise, biterion)


def matlab_string(obj):
    """
    Return a string parsed from a matlab file
    """
    return ''.join(chr(c) for c in obj[:, 0])


def matlab_strings(mat, ref):
    """
    Returns an array of strings parsed from matlab file
    """
    return [matlab_string(mat[r]) for r in ref[:, 0]]


def matlab_array(mat, ref, dtype):
    """
    Parses the relevant information (ref) with type
    dtype from a matlab file (mat) and returns
    a numpy array.

    Parameter:
        mat: matlab file containing pose information
        ref: the column of the file of interest
        dtype: data type of data
    """
    N = len(ref)
    arr = np.empty(N, dtype=dtype)
    for i in range(N):
        arr[i] = mat[ref[i, 0]][0, 0]
    return arr


def load(traintest, mat_full):
    """
    Loads train or test data from mat lab file containing both train
    and test data and returns the relevant information in numpy arrays

    Parameters:
        traintest: a string that denotes "train" or "test"
        mat_full: the matlab file containing pose information

    Returns:
        pan: a numpy array containing pan angles from the dataset
        tilt: a numpy array containing tilt angles from the dataset
        roll: a numpy array containing roll angles from the dataset
        names: a numpy array containing image names from the dataset
    """
    container = mat_full['or_label_' + traintest]
    pan = matlab_array(mat_full, container['pan'], np.float32)
    tilt = matlab_array(mat_full, container['tilt'], np.float32)
    roll = matlab_array(mat_full, container['roll'], np.float32)
    names = matlab_strings(mat_full, container['name'])
    return pan, tilt, roll, names


class IDIAPSplitSet(Dataset):
    """
        Stores a training or test set for the UPNA Head Pose Dataset

        Parameters:
            dataset_path: the location of where processed images and poses will
                be stored.
            image_transforms: A list of of composed pytorch transforms to be
                applied to a PIL image
            euler_noise: the standard deviation of the Gaussian distribution
                that we sample noise from
            quat_noise: the Z of the bingham distribution that we sample noise
                from
    """

    def __init__(self, table, image_transform, dataset_path, euler_noise,
                 quat_noise, biterion):
        self.image_transform = image_transform
        self.pan, self.tilt, self.roll, self.names = table
        self.dataset_path = dataset_path
        if euler_noise:
            s = np.random.normal(0, euler_noise, 3 * len(self.names))
            self.euler_noise = []
            for i in range(len(self.names)):
                self.euler_noise.append([s[i * 3], s[i * 3 + 1], s[i * 3 + 2]])
        else:
            self.euler_noise = None

        if quat_noise:
            bd = BinghamDistribution(np.eye(4), np.array(quat_noise))
            self.quat_noise = quaternion.as_quat_array(
                bd.random_samples(len(self.pan)))
        else:
            self.quat_noise = []
        self.biterion = biterion

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        pan = self.pan[idx]
        tilt = self.tilt[idx]
        roll = self.roll[idx]
        name = self.names[idx]
        img_name = os.path.join(self.dataset_path, name)
        image = Image.fromarray(io.imread(img_name))

        if self.image_transform:
            image = self.image_transform(image)
        if self.euler_noise:
            pan = math.degrees(pan) + self.euler_noise[idx][0]
            tilt = math.degrees(tilt) + self.euler_noise[idx][1]
            roll = math.degrees(roll) + self.euler_noise[idx][2]
        else:
            pan = math.degrees(pan)
            tilt = math.degrees(tilt)
            roll = math.degrees(roll)
        if len(self.quat_noise) != 0:
            w, x, y, z = convert_euler_to_quaternion(pan, tilt, roll)
            quat_pose = quaternion.quaternion(w, x, y, z)
            res = quaternion.as_float_array(quat_pose * self.quat_noise[idx])
            euler_res = quaternion_to_euler(res[0], res[1], res[2], res[3])
            pan = math.degrees(euler_res[0])
            tilt = math.degrees(euler_res[1])
            roll = math.degrees(euler_res[2])

        if self.biterion:
            sample = {'image': image,
                      'pose': torch.Tensor([pan, tilt, roll])}

        else:
            sample = {'image': image,
                      'pose': torch.from_numpy(
                          convert_euler_to_quaternion(pan,
                                                      tilt,
                                                      roll))}
        return sample
