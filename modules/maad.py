import torch
from modules.gram_schmidt import gram_schmidt, gram_schmidt_batched
from modules.quaternion_matrix import quaternion_matrix
from utils.utils import \
    convert_euler_to_quaternion
from modules.vm_operations import *
import math

def angular_loss_single_sample(target, predicted):
    """ Returns the angle between two quaternions.
        Note that for a quaternion q, -q = q so the
        angle of rotation must be less than 180 degrees.

        Inputs:
          target = target quaternion
          predicted = predicted quaternion
    """
    quat_ang = torch.clamp(torch.abs(torch.dot(target, predicted)), min=0,
                           max=1)
    acos_val = torch.acos(quat_ang)
    diff_ang = acos_val * 2
    return diff_ang


def maad_mse(target, predicted):
    """
        Computes the MAAD  over a batch of
        target, predicted quaternion pairs

        Inputs
          target = batch of target quaternions
          predicted = batch of predicted quaternions
    """
    angular_loss = 0
    for i in range(target.shape[0]):
        angular_loss += angular_loss_single_sample(target[i], predicted[i])

    return angular_loss / target.shape[0]

def maad_cosine(target, predicted):
    angular_dev = 0
    for i in range(target.shape[0]):
        angles = output_to_angles(predicted[i])
        pan = torch.atan2(angles[1], angles[0])
        tilt = torch.atan2(angles[3], angles[2])
        roll = torch.atan2(angles[5], angles[4])

        pan_target = target[i][0]
        tilt_target = target[i][1]
        roll_target = target[i][2]

        target_quat = convert_euler_to_quaternion(pan_target, tilt_target,
                                                      roll_target)
        predicted_quat = convert_euler_to_quaternion(math.degrees(pan),
                                                         math.degrees(tilt),
                                                         math.degrees(roll))
        angular_dev += angular_loss_single_sample(torch.from_numpy(target_quat),
                                                  torch.from_numpy(
                                                      predicted_quat))

    return angular_dev / target.shape[0]



def maad_biternion(target, predicted):
    angular_dev = 0
    for i in range(target.shape[0]):
        angles, kappas = output_to_angles_and_kappas(predicted[i])
        pan = torch.atan2(angles[1], angles[0])
        tilt = torch.atan2(angles[3], angles[2])
        roll = torch.atan2(angles[5], angles[4])

        pan_target = target[i][0]
        tilt_target = target[i][1]
        roll_target = target[i][2]

        target_quat = convert_euler_to_quaternion(pan_target, tilt_target,
                                                      roll_target)
        predicted_quat = convert_euler_to_quaternion(math.degrees(pan),
                                                         math.degrees(tilt),
                                                         math.degrees(roll))
        angular_dev += angular_loss_single_sample(torch.from_numpy(target_quat),
                                                  torch.from_numpy(
                                                      predicted_quat))

    return angular_dev / target.shape[0]


def maad_bingham(target, predicted, orthogonalization="gram_schmidt"):
    """ Computes mean absolute angular deviation between a pair of quaternions

    Parameters:
        predicted (torch.Tensor): Output from network of shape (N, 16) if
            orthogonalization is "gram_schmidt" and (N, 4) if it is
            "quaternion_matrix".
       target (torch.Tensor): Ground truth of shape N x 4
       orthogonalization (str): Orthogonalization method to use. Can be
            "gram_schmidt" for usage of the classical gram-schmidt method.
            "modified_gram_schmidt" for a more robust variant, or
            "quaternion_matrix" for usage of a orthogonal matrix representation
            of an output quaternion.
    """
    angular_dev = 0
    if orthogonalization == "gram_schmidt":
        batch_size = target.shape[0]
        reshaped_output = predicted.reshape(batch_size, 4, 4)
        param_m = gram_schmidt_batched(reshaped_output)

        for i in range(batch_size):
            angular_dev += angular_loss_single_sample(
                target[i], param_m[i, :, 3])
    else:
        for i in range(target.shape[0]):
            if orthogonalization == "modified_gram_schmidt":
                reshaped_output = predicted[i][-16:].reshape(4, 4)
                param_m = gram_schmidt(reshaped_output, modified=True)
            elif orthogonalization == "quaternion_matrix":
                param_m = quaternion_matrix(predicted[i])
            else:
                raise ValueError("Invalid orthogonalization method.")
            angular_dev += angular_loss_single_sample(target[i], param_m[:, 3])

    return angular_dev / target.shape[0]



