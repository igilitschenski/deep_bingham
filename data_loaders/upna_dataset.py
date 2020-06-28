import os
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
import h5py
from .upna_preprocess import *
from .utils import *
from bingham_distribution import BinghamDistribution

def make_hdf5_file(config, image_transform):
    dataset_path = config["preprocess_path"]
    csv_train = dataset_path + "/train/input.csv"
    csv_test = dataset_path + "/test/input.csv"
    biterion = config["biterion"]
    if os.path.isfile(csv_train) and os.path.isfile(csv_test):
        test_frame = pd.read_csv(csv_test)
        train_frame = pd.read_csv(csv_train)

    else:
        preprocess = UpnaHeadPoseDataPreprocess(config)
        test_frame = preprocess.frame_test
        train_frame = preprocess.frame_train

    train = UpnaHeadPoseSplitSet(dataset_path + "/train",
                                 train_frame, image_transform)
    test = UpnaHeadPoseSplitSet(dataset_path + "/test",
                                test_frame, image_transform)

    img_shape = train[0]["image"].shape
    label_shape = train[0]["pose"].shape[-1]

    f = h5py.File(dataset_path + "/dataset.hdf5", "w")

    f.create_dataset("train_img", (
        len(train), img_shape[0], img_shape[1], img_shape[2]))
    f.create_dataset("train_label", (len(train), label_shape))
    f.create_dataset("test_img", (
        len(test), img_shape[0], img_shape[1], img_shape[2]))
    f.create_dataset("test_label", (len(test), label_shape))

    for i, data in enumerate(train):
        f["train_img"][i, :, :, :] = train[i]["image"]
        f["train_label"][i, :] = train[i]["pose"]
        print("train", i)

    for i, data in enumerate(test):
        f["test_img"][i, :, :, :] = test[i]["image"]
        f["test_label"][i, :] = test[i]["pose"]
        print("test", i)


class UpnaHeadPoseTrainTest():
    """
        Stores a training and test set for the UPNA Head Pose Dataset

        Parameters:
        config_file: a yaml file or dictionary that contains data loading
            information ex. See configs/upna_train.yaml. The dataset_path stores
            the locsation of the originial downloaded dataset. The
            preprocess_path is where the processed images and poses will be stored.

        image_transforms: A list of of composed pytorch transforms to be applied
            to a PIL image
    """

    def __init__(self, config_file, image_transform=None):
        if type(config_file) == dict:
            config = config_file
        else:
            with open(config_file) as fp:
                config = yaml.load(fp)

        if not os.path.isfile(config["preprocess_path"] + "/dataset.hdf5"):
            make_hdf5_file(config_file, image_transform)
        f = h5py.File(config["preprocess_path"] + "/dataset.hdf5", 'r')
        noise = config["euler_noise"]
        quat_noise = config["quat_noise"]
        biterion = config["biterion"]
        self.train = UpnaHDF5(f.get('train_img'), f.get('train_label'),
                              biterion, noise, quat_noise)
        self.test = UpnaHDF5(f.get('test_img'), f.get('test_label'), biterion,
                             noise, quat_noise)


class UpnaHDF5(Dataset):
    """
        Loads UPNA dataset from a HDF5 dataset and applies transformations to
        biterion or quaternion form and adds noise to the labels.
        biterion: format of the pose. if true, biterion. if false, quaternion.
        euler_noise: the standard deviation of the Gaussian distribution that we
            sample noise from
        quat_noise: the Z of a bingham distribution that we sample noise from
    """

    def __init__(self, images, labels, biterion, euler_noise, quat_noise):
        self.images = images
        self.labels = labels
        self.biterion = biterion
        if euler_noise:
            s = np.random.normal(0, euler_noise, 3 * len(self.labels))
            self.euler_noise = []
            for i in range(len(self.labels)):
                self.euler_noise.append([s[i * 3], s[i * 3 + 1], s[i * 3 + 2]])
        else:
            self.euler_noise = None

        if quat_noise:
            quat_noise = [float(quat_noise[0]), float(quat_noise[1]),
                          float(quat_noise[2]), 0]
            bd = BinghamDistribution(np.identity(4), np.array(quat_noise))
            samples = bd.random_samples(len(labels))
            perm = [3, 0, 1, 2]
            re_samples = samples[:, perm]
            self.quat_noise = quaternion.as_quat_array(re_samples)
        else:
            self.quat_noise = []

    def __getitem__(self, idx):

        image = torch.from_numpy(self.images[idx, :, :, :]).float()
        if self.euler_noise:
            pose = np.array([self.labels[idx][0] + self.euler_noise[idx][0],
                             self.labels[idx][1] + self.euler_noise[idx][1],
                             self.labels[idx][2] + self.euler_noise[idx][2]])
        else:
            pose = self.labels[idx, :]
        if len(self.quat_noise) != 0:
            w, x, y, z = convert_euler_to_quaternion(pose[0], pose[1], pose[2])
            quat_pose = quaternion.quaternion(w, x, y, z)
            res = quaternion.as_float_array(quat_pose * self.quat_noise[idx])
            roll, pitch, yaw = quaternion_to_euler(res[0], res[1], res[2],
                                                   res[3])
            pose = np.array(
                [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])

        if self.biterion:
            sample = {'image': image,
                      'pose': torch.from_numpy(pose)}
        else:
            sample = {'image': image,
                      'pose': convert_euler_to_quaternion(pose[0],
                                                          pose[1],
                                                          pose[2])}

        return sample

    def __len__(self):
        return self.images.shape[0]


class UpnaHeadPoseSplitSet(Dataset):
    def __init__(self, dataset_path, frame, image_transform):
        """
        Stores a training or test set for the UPNA Head Pose Dataset

        Parameters:
            dataset_path: the location of where processed images and poses will be stored.
            frame: the the csv frame that stores the posesi
            image_transforms: A list of of composed pytorch transforms to be applied to a PIL image
    
        """
        self.frame = frame
        self.image_transform = image_transform
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        name = self.frame.iloc[idx, 0]
        frame_index = idx
        img_name = os.path.join(self.dataset_path, name)
        image = Image.fromarray(io.imread(img_name))
        head_pose = self.frame.iloc[frame_index, 1:4].as_matrix()
        head_pose = head_pose.astype('float').reshape(-1, 3)[0]

        if self.image_transform:
            image = self.image_transform(image)

        sample = {'image': image,
                  'pose': head_pose}

        return sample


# TODO: GET RID OF THIS- REDUNDANT. except for the images field. need to incorporate that elsewhere...
class UpnaHeadPoseDataset(Dataset):
    """
        Stores a test set for the UPNA Head Pose Dataset

        Parameters:
        config_file: a yaml file or dictionary that contains data loading
            information ex. See configs/upna_train.yaml. The dataset_path stores
            the location of the originial downloaded dataset. The
            preprocess_path is where the processed images and poses will be stored.

        image_transforms: (optional) A list of of composed pytorch transforms to
            be applied to a PIL image
        images: (optional) Can provide a list of image names and a dataset will
            be constructed with those images
    """

    def __init__(self, config_file, image_transform=None):
        if type(config_file) == dict:
            config = config_file
        else:
            with open(config_file) as fp:
                config = yaml.load(fp)
        self.dataset_path = config["preprocess_path"] + "/test"
        self.csv_path = self.dataset_path + "/input.csv"
        self.user = config["user"]
        self.video = config["video"]
        if os.path.isfile(self.csv_path):
            self.frame = pd.read_csv(self.csv_path)

        else:
            self.frame = UpnaHeadPoseDataPreprocess(config_file).frame_test

        self.image_transform = image_transform
        self.images = self._generate_file_names()

    def __len__(self):
        return len(self.images)

    def _generate_file_names(self):
        """
        From user number and video number, generate a list of corresponding frames.

        Parameters:
            user_num: string user number ex. "07"
            video_num: string video number ex. "03"

        Returns:
            names: a list of file names.
        """
        names = []
        for i in range(1, 300):
            string_name = "User_{}/user_{}_video_{}_frame{}.jpg".format(
                self.user, self.user, self.video, i)
            names.append(string_name)
        return names

    def __getitem__(self, idx):
        name = self.images[idx]
        frame_index = get_frame_index(name, self.frame)
        img_name = os.path.join(self.dataset_path, name)
        image = Image.fromarray(io.imread(img_name))
        head_pose = self.frame.iloc[frame_index, 1:4].as_matrix()
        head_pose = head_pose.astype('float').reshape(-1, 3)[0]

        if self.image_transform:
            image = self.image_transform(image)

        sample = {'image': image,
                  'pose': torch.from_numpy(
                      convert_euler_to_quaternion(head_pose[0], head_pose[1],
                                                  head_pose[2]))}
        return sample
