from modules.maad import output_to_angles, maad_cosine 
from utils import radians
import torch

class CosineLoss():
    """
    Class for calculating Cosine Loss assuming biternion representation of pose. 
    """
    def __init__(self):
        self.stats = 0
    def __call__(self, target, output):
        """
        Calculates the cosine loss on a batch of target-output values.

        Arguments:
            target: Target values at which loss is evaluated of shape (N, 3)
            output: Output values, shape (N, 6) from predicted from network 
                   prior to normalization of each sin/cos pair.

        Result:
            loss: The loss of the current batch
            log_likelihood: 0. This loss function does not calculate log likelihood so
                           so 0 is returned.
        """
        loss = 0
        for i in range(output.shape[0]):
             loss += self._cosine_single_sample(target[i], output[i])
        return loss, torch.Tensor([0])
 
    def statistics(self, target, output, cur_epoch):
         stats = {"maad": float(maad_cosine(target, output))}
         self.stats = stats
         return stats

    def _cosine_single_sample(self, target, output):
        """
        Calculates cosine loss for a single sample.

        Arguments:
           target: Target value at which loss is evaluated of shape (1, 3)
           output: Output value, shape (1, 6) from predicted from network 
                   prior to normalization of each sin/cos pair.

        Returns:
            loss: The loss of a single sample.
        """
        radian_target = radians(target)
        radian_target_cos = torch.cos(radian_target)
        radian_target_sin = torch.sin(radian_target)
        target_biternion = []
        for i in range(3):
            target_biternion.append(radian_target_cos[i])
            target_biternion.append(radian_target_sin[i])
        target = torch.tensor(target_biternion) 

        if output.is_cuda:
            device = output.get_device()
            target = target.to(device)

        angles  = output_to_angles(output)
        return 3 - torch.dot(angles, target)

