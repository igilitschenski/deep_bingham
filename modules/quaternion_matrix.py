import torch


def quaternion_matrix(quat):
    """ Computes an orthogonal matrix from a quaternion.

    We use the representation from the NeurIPS 2018 paper "Bayesian Pose
    Graph Optimization via Bingham Distributions and Tempred Geodesic MCMC" by
    Birdal et al. There, the presentation is given above eq. (6). In practice
    any similar scheme will do.

    Parameters:
        quat (torch.tensor): Tensor of shape 4 representing a quaternion.

    """
    # This cumbersome way is necessary because copy constructors seem not to
    # preserve gradients.
    indices = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ], device=quat.device)

    sign_mask = torch.tensor([
        [1, -1, -1,  1],
        [1,  1,  1,  1],
        [1, -1,  1, -1],
        [1,  1, -1, -1]
    ], device=quat.device, dtype=quat.dtype)

    quat_normalized = quat / torch.norm(quat)
    quat_mat = torch.take(quat_normalized, indices)
    quat_mat = sign_mask * quat_mat

    return quat_mat
