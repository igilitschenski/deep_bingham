"""
Deep Orientation Estimation Training
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from tensorboardX import SummaryWriter

import data_loaders
import modules.network

from modules import BinghamLoss, BinghamMixtureLoss, \
    VonMisesLoss, MSELoss, CosineLoss
from training import Trainer


torch.manual_seed(0)
DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/upna_train.yaml"

LOSS_FUNCTIONS = {'mse': MSELoss,
                  'bingham': BinghamLoss,
                  'bingham_mdn': BinghamMixtureLoss,
                  'von_mises': VonMisesLoss,
                  'cosine': CosineLoss}


def get_dataset(config):
    """ Returns the training data using the provided configuration."""

    data_loader = config["data_loader"]
    size = data_loader["input_size"]
    data_transforms = transforms.Compose([
        transforms.CenterCrop(600),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data_transforms_idiap = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_transforms_depth = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    if data_loader["name"] == "UPNAHeadPose":
        dataset = data_loaders.UpnaHeadPoseTrainTest(
            data_loader["config"], data_transforms)
        train_dataset = dataset.train
    elif data_loader["name"] == "IDIAP":
        dataset = data_loaders.IDIAPTrainTest(data_loader["config"], 
        data_transforms_idiap)
        train_dataset = dataset.train
    elif data_loader["name"] == "T_Less":
        dataset = data_loaders.TLessTrainTest(
            data_loader["config"], data_transforms_idiap)
        train_dataset = dataset.train
    else:
        sys.exit("Unknown data loader " + config['data_loader']["name"] + ".")

    training_size = int(len(train_dataset) * 0.90)
    val_size = len(train_dataset) - training_size

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [training_size, val_size]) 
    return train_dataset, val_dataset


def main():
    """ Loads arguments and starts training."""
    parser = argparse.ArgumentParser(description="Deep Orientation Estimation")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)

    args = parser.parse_args()
    config_file = args.config

    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(
        args.config)

    with open(config_file) as fp:
        config = yaml.load(fp)

    if not os.path.exists(config["train"]["save_dir"]):
        os.makedirs(config["train"]["save_dir"])

    device = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Build model architecture
    num_channels = config["train"]["num_channels"] or 3
    model_name = config["train"]["model"] or 'vgg11'
    num_classes = config["train"].get("num_outputs", None)

    model = modules.network.get_model(name=model_name,
                                      pretrained=True,
                                      num_channels=num_channels,
                                      num_classes=num_classes)
    model.to(device)
    print("Model name: {}".format(model_name))

    # optionally resume from checkpoint

    resume = config["train"]["resume"]
    if resume:
        if os.path.isfile(resume):
            print("Loading checkpoint {}".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint["epoch"]

            model.load_state_dict(checkpoint["state_dict"])

        else:
            start_epoch = 0
            print("No checkpoint found at {}".format(resume))

    else:
        start_epoch = 0

    # Get dataset
    train_dataset, test_dataset = get_dataset(config)
    b_size = config["train"]["batch_size"] or 4

    # This should not be necessary but it surprisingly is. In the presence of a
    # GPU, PyTorch tries to allocate GPU memory when pin_memory is set to true
    # in the data loader. This happens even if training is to happen on CPU and
    # all objects are on CPU.
    if config["train"]["device"] != "cpu":
        use_memory_pinning = True
    else:
        use_memory_pinning = False

    validationloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=b_size, shuffle=True, num_workers=1,
        pin_memory=use_memory_pinning)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=b_size, shuffle=True, num_workers=1,
        pin_memory=use_memory_pinning)

    print("batch size: {}".format(b_size))
    # Define loss function (criterion) and optimizer
    learning_rate = config["train"]["learning_rate"] or 0.0001
    loss_function_name = config["train"]["loss_function"]

    if "loss_parameters" in config["train"]:
        loss_parameters = config["train"]["loss_parameters"]
    else:
        loss_parameters = None

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(optimizer)
    # Set up tensorboard writer

    writer_train = SummaryWriter(
        "runs/{}/training".format(config["train"]["save_as"]))
    writer_val = SummaryWriter(
        "runs/{}/validation".format(config["train"]["save_as"]))

    # Train the network
    num_epochs = config["train"]["num_epochs"] or 2
    print("Number of epochs: {}".format(num_epochs))
    if loss_parameters is not None:
        loss_function = LOSS_FUNCTIONS[loss_function_name](**loss_parameters)
    else:
        loss_function = LOSS_FUNCTIONS[loss_function_name]()

    if "floating_point_type" in config["train"]:
        floating_point_type = config["train"]["floating_point_type"]
    else:
        floating_point_type = "float"
    trainer = Trainer(device, floating_point_type)

    for epoch in range(start_epoch, num_epochs):
        trainer.train_epoch(
            trainloader, model, loss_function, optimizer,
            epoch, writer_train, writer_val, validationloader)
        save_checkpoint(
            {'epoch': epoch + 1, 'state_dict': model.state_dict()},
            filename=os.path.join(config["train"]["save_dir"],
                                  'checkpoint_{}_{}.tar'.format(
                                      model_name, epoch))
        )

    print('Finished training')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()
