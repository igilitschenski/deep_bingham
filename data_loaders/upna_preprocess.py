from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import yaml
# Ignore warnings
import warnings
import csv

warnings.filterwarnings("ignore")
import cv2

TRAIN_SET = set(
    ["User_01", "User_02", "User_03", "User_04", "User_05", "User_06"])
TEST_SET = set(["User_07", "User_08", "User_09", "User_10"])


class UpnaHeadPoseDataPreprocess:
    def __init__(self, config_file):
        if type(config_file) != dict:
            with open(config_file) as fp:
                config = yaml.load(fp)

        else:
            config = config_file

        print(os.path.abspath(config["dataset_path"]))
        assert os.path.isdir(os.path.abspath(config["dataset_path"]))
        self.dataset_path = config["dataset_path"]
        self.processed_path = config["preprocess_path"]

        if not os.path.isdir(config["preprocess_path"]):
            os.mkdir(config["preprocess_path"])

            print("preprocessing dataset")
            self._preprocess_data()

        # Create data frames for training and validation sets. Maps image names
        # to quaternions.
        train_data_frame, test_data_frame = self._make_data_frame()
        self.frame_train = pd.read_csv(train_data_frame)
        self.frame_test = pd.read_csv(test_data_frame)
        print("dataset preprocessing finished")

    def _extract_frames(self, videofile, videofolder):
        """Extracts frames from video and stores them as jpg image."""

        # Create folder for processed video if necessary.
        if videofolder in TRAIN_SET:
            videopath = self.processed_path + "/train/" + videofolder
        else:
            videopath = self.processed_path + "/test/" + videofolder
        if not os.path.isdir(videopath):
            os.makedirs(videopath)

        file_prefix = videopath + "/" + os.path.splitext(videofile)[
            0] + "_frame"

        # Open and process video.
        filepath = self.dataset_path + "/" + videofolder + "/" + videofile
        video_capture = cv2.VideoCapture(filepath)
        success, image = video_capture.read()
        count = 1
        while success:
            # save frame as jpg file.
            cv2.imwrite(file_prefix + ("%d.jpg" % (count)), image)
            success, image = video_capture.read()
            count += 1

    def _transform_labels(self, labelfile, labelfolder):
        """ Transforms labels from Euler angles into dual quaternions. """
        # Create folder for processed labels if necessary.
        if labelfolder in TRAIN_SET:
            labeldir = self.processed_path + "/train/" + labelfolder
        else:
            labeldir = self.processed_path + "/test/" + labelfolder
        if not os.path.isdir(labeldir):
            os.makedirs(labeldir)

        # Open and transform label files replacing euler angles by quaternions.
        # The quaternion a + b I + c J + d K is stored as a b c d.
        file_path = self.dataset_path + "/" + labelfolder + "/" + labelfile
        quaternion_list = []
        with open(file_path, 'rt') as groundtruth_file:
            csv_reader = csv.reader(groundtruth_file, delimiter='\t',
                                    lineterminator='\r\n\r\n\t')
            for row in csv_reader:
                orientation_quat = np.array(
                    [float(row[3]), float(row[4]), float(row[5])])
                quaternion_list.append(orientation_quat)

            processed_file_path = labeldir + "/" + labelfile

        with open(processed_file_path, "wt") as processed_file:
            csv_writer = csv.writer(processed_file, delimiter="\t")
            for row in quaternion_list:
                row = np.round(row, 4).tolist()
                csv_writer.writerow(row)

    def _preprocess_data(self):
        """Transforms video frames into image frames."""

        print("Transforming video frames into image frames")
        # Iterate over subfolders.
        iter_dirs = iter(os.walk(self.dataset_path))
        next(iter_dirs)
        # first = True
        for cur_dir in iter_dirs:
            cur_dir_basename = os.path.basename(os.path.normpath(cur_dir[0]))
            for cur_file in cur_dir[2]:
                if cur_file.endswith('groundtruth3D_zeroed.txt'):  # and first:
                    # Transform orientation labels into quaternions.
                    print("Processing " + cur_file)
                    self._transform_labels(cur_file, cur_dir_basename)
                elif cur_file.endswith('mp4'):  # and first:
                    # Extract frames from videos.
                    self._extract_frames(cur_file, cur_dir_basename)

    def _make_data_frame(self):
        print("Creating a csv mapping image file names to quaternion poses")
        train_csv_file = self.processed_path + "/train/input.csv"
        test_csv_file = self.processed_path + "/test/input.csv"
        iter_dirs = iter(os.walk(self.processed_path))
        next(iter_dirs)

        for cur_dir in iter_dirs:
            cur_dir_basename = os.path.basename(os.path.normpath(cur_dir[0]))
            for cur_file in cur_dir[2]:
                if cur_file.endswith('groundtruth3D_zeroed.txt'):
                    if cur_dir_basename in TRAIN_SET:
                        self._add_images_poses_to_csv(cur_file,
                                                      cur_dir_basename,
                                                      train_csv_file)
                    else:
                        self._add_images_poses_to_csv(cur_file,
                                                      cur_dir_basename,
                                                      test_csv_file)

        return train_csv_file, test_csv_file

    def _add_images_poses_to_csv(self, cur_file, cur_dir_basename, csv_file):
        image_name_list = []
        if cur_dir_basename in TRAIN_SET:
            quat_path = self.processed_path + "/train/" + cur_dir_basename \
                        + "/" + cur_file
        else:
            quat_path = self.processed_path + "/test/" + cur_dir_basename \
                        + "/" + cur_file
        with open(quat_path, 'rt') as quaternion_file:
            csv_reader = csv.reader(quaternion_file, delimiter='\t')
            data = list(csv_reader)
            row_count = len(data)

        for i in range(1, row_count + 1):
            words = os.path.splitext(cur_file)[0].split("_")
            sub_name = "_".join(words[:4])
            name = cur_dir_basename + "/" + sub_name + "_frame{}.jpg".format(i)
            image_name_list.append(name)

        with open(csv_file, "a") as fp:
            field_names = ["image_name", "q0", "q1", "q2", "q3"]
            csv_writer = csv.DictWriter(fp, fieldnames=field_names)
            print("writing to csv file", image_name_list[0])
            for i in range(row_count):
                quat = data[i]
                csv_writer.writerow(
                    {'image_name': image_name_list[i], "q0": quat[0],
                     "q1": quat[1], "q2": quat[2]})
