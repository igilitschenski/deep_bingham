import argparse
import os
import torch
import torchvision.transforms as transforms
import yaml

import data_loaders
import modules.network

from modules import angular_loss, BinghamFixedDispersionLoss, \
    BinghamHybridLoss, BinghamLoss, BinghamMixtureLoss, \
    CosineLoss, MSELoss, VonMisesLoss, VonMisesFixedKappaLoss
from utils.evaluation import run_evaluation

DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/upna_train.yaml"

LOSS_FUNCTIONS = {'mse': MSELoss,
                  'bingham': BinghamLoss,
                  'bingham_mdn': BinghamMixtureLoss,
                  'von_mises': VonMisesLoss,
                  'cosine': CosineLoss}


def get_dataset(config):
    """Returns the test data using the provided configuration"""

    data_loader = config["data_loader"]
    size = data_loader["input_size"]

    data_transforms = transforms.Compose([transforms.CenterCrop(600),
                                          transforms.Resize((size, size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
    data_transforms_idiap = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if data_loader["name"] == "UPNAHeadPose":
        dataset = data_loaders.UpnaHeadPoseTrainTest(
            data_loader["config"], data_transforms)
        test_dataset = dataset.test
    elif data_loader["name"] == "T_Less":
        dataset = data_loaders.TLessTrainTest(data_loader["config"],
                  data_transforms_idiap)
        test_dataset = dataset.test
    else:
        dataset = data_loaders.IDIAPTrainTest(
            data_loader["config"], data_transforms_idiap)
        test_dataset = dataset.test

    return test_dataset


def get_data_loader(dataset, batch_size):
    """Return a data loader"""
    dataset = get_dataset(dataset)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def main():
    """Loads arguments and starts testing."""

    parser = argparse.ArgumentParser(
        description="Deep Orientation Estimation")

    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)

    args = parser.parse_args()
    config_file = args.config

    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(
        args.config)

    with open(config_file) as fp:
        config = yaml.load(fp)

    if "loss_parameters" in config["test"]:
        loss_parameters = config["test"]["loss_parameters"]
    else:
        loss_parameters = None

    device = torch.device(config["test"][
            "device"] if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    num_classes = config["test"]["num_outputs"]
    # Build model architecture
    num_channels = config["test"]["num_channels"]
    model_name = config["test"]["model"]
    model = modules.network.get_model(name=model_name,
                              pretrained=True,
                              num_channels=num_channels,
                              num_classes=num_classes)
    model.to(device)
    print("Model name: {}".format(model_name))

    model_path = config["test"]["model_path"]
    
    if os.path.isfile(model_path):
        print("Loading model {}".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        
    else:
        assert "model not found"

    # Get data loader
    batch_size = 32
    test_loader = get_data_loader(config, batch_size)

    loss_function_name  = config["test"]["loss_function"]
    dataset_name = config["data_loader"]["name"]

    if loss_parameters:
        criterion = LOSS_FUNCTIONS[loss_function_name](**loss_parameters)
    else:
        criterion = LOSS_FUNCTIONS[loss_function_name]()

    if "floating_point_type" in config["test"]:
        floating_point_type = config["test"]["floating_point_type"]
    else:
        floating_point_type = "float"
    if floating_point_type == "double":
        model.double()

    run_evaluation(
        model, test_loader, criterion, 
        device, floating_point_type
    )
    

if __name__ == '__main__':
    main()
