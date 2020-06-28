import torch
from modules import maad
from utils import AverageMeter, eaad_bingham, eaad_von_mises
import numpy as np


def run_evaluation(model, dataset, loss_function, device, floating_point_type="float"):
    model.eval()
    losses = AverageMeter()
    log_likelihoods = AverageMeter()
    maads = AverageMeter()
    averaged_stats = AverageMeter()
    eaads = AverageMeter()
    min_eaads = AverageMeter()
    min_maads = AverageMeter() 
    val_load_iter = iter(dataset)
    eval_fraction = 0.1
    for i in range(int(len(dataset)*eval_fraction)):
        data = val_load_iter.next()
        if floating_point_type == "double":
            target_var = data["pose"].double().to(device)
            input_var = data["image"].double().to(device)
        else:
            target_var = data["pose"].float().to(device)
            input_var = data["image"].float().to(device)

        if torch.sum(torch.isnan(target_var)) > 0:
            continue
            # compute output
        output = model(input_var)
        if loss_function.__class__.__name__ == "MSELoss":
            # norm over the last dimension (ie. orientations)
            norms = torch.norm(output, dim=-1, keepdim=True).to(device)
            output = output / norms
        if loss_function.__class__.__name__ == "BinghamMixtureLoss":
            loss, log_likelihood = loss_function(target_var, output, 49)
        else:
            loss, log_likelihood = loss_function(target_var, output)

        if floating_point_type == "double":
            loss = loss.double() / data["image"].shape[0]
        else:
            loss = loss.float() / data["image"].shape[0]
        # measure accuracy and record loss
        losses.update(loss.item(), data["image"].size(0))
        log_likelihoods.update(log_likelihood.item(), data["image"].size(0))

        if loss_function.__class__.__name__ == "VonMisesLoss":
            angular_deviation = maad(loss_function, target_var, output, None)
            maads.update(angular_deviation)
            min_maads.update(angular_deviation)
            eaad, min_eaad = kappas_to_eaad(output)
            eaads.update(eaad, data["image"].size(0))
            min_eaads.update(min_eaad, data["image"].size(0))
        else:
            stats = loss_function.statistics(target_var, output, 31)
            averaged_stats.update(stats, data["image"].size(0))
            maads.update(stats["maad"])
            if loss_function.__class__.__name__ == "BinghamMixtureLoss":
                min_maads.update(stats["mmaad"])
            else:
                min_maads.update(stats["maad"])

        if "Bingham" in loss_function.__class__.__name__:
            eaad, min_eaad = bingham_z_to_eaad(
                stats, loss_function
            ) 
            eaads.update(eaad, data["image"].size(0))
            min_eaads.update(min_eaad, data["image"].size(0))

    if "Bingham" or "VonMises" in loss_function.__class__.__name__:
        print("Loss: {}, Log Likelihood: {}, MAAD: {}, Min MAAD: {}, EAAD: {}, Min EAAD: {}".format(
                losses.avg, log_likelihoods.avg, maads.avg, min_maads.avg, eaads.avg, min_eaads.avg))
    else:
        print("Loss: {}, Log Likelhood: {}, MAAD: {}".format(losses.avg, log_likelihoods.avg, maads.avg))


def kappas_to_eaad(output):
    kappas = torch.mean(output[:, :3], 0).detach().cpu().numpy()
    eaad = eaad_von_mises(kappas)
    return eaad, eaad
    
def bingham_z_to_eaad(stats, loss_function):
    eaads = []

    if loss_function.__class__.__name__ == "BinghamLoss":
        z_0, z_1, z_2 = stats["z_0"], stats["z_1"], stats["z_2"]
        bingham_z = np.array([z_0, z_1, z_2, 0])
        eaad = eaad_bingham(bingham_z)
        eaads.append(eaad)

    elif loss_function.__class__.__name__ == "BinghamMixtureLoss":
        for j in range(loss_function._num_components):
            bingham_z = [stats["mode_" + str(j) + "_z_{}".format(i)] for i in range(3)]
            bingham_z.append(0)
            bingham_z = np.array(bingham_z)
            eaad = eaad_bingham(bingham_z)
            eaads.append(eaad)
    return sum(eaads)/len(eaads), min(eaads)
