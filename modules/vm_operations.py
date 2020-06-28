import torch

def output_to_kappas(output):
    zero_vec = torch.zeros(len(output), 3)
    if output.is_cuda:
        device = output.get_device()
        zero_vec = torch.zeros(len(output), 3).to(device)
    kappas = torch.where(output[:, :3] > 0, output[:, :3], zero_vec)
    return kappas


def output_to_angles(output):
    pan = normalize_cosine_sine(output[:2])
    tilt = normalize_cosine_sine(output[2:4])
    roll = normalize_cosine_sine(output[4:])

    angles = torch.cat((pan, tilt, roll), 0)
    return angles

def output_to_angles_and_kappas(output):
    pan = normalize_cosine_sine(output[3:5])
    tilt = normalize_cosine_sine(output[5:7])
    roll = normalize_cosine_sine(output[7:])

    angles = torch.cat((pan, tilt, roll), 0)

    zero_vec = torch.zeros(3)
    if output.is_cuda:
        device = output.get_device()
        zero_vec = torch.zeros(3).to(device)
    kappas = torch.where(output[:3] > 0, output[:3], zero_vec)
    return angles, kappas


def normalize_cosine_sine(angle_tensor):
    return angle_tensor / torch.sqrt(torch.sum(torch.pow(angle_tensor, 2)))


