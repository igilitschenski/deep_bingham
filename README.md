# Deep Orientaton Uncertainty Learning based on a Bingham Loss
![deep_bingham_teaser](https://user-images.githubusercontent.com/11874191/80402132-71d77a00-888b-11ea-978d-777158ed40ea.gif)

## Installation
For a quick conda install simply run

```bash
conda env create --file=environment.yaml
conda activate deep_orientation
```

## Datasets
For the paper, we make use of the following datasets (some of them require a registration for downloading).

1. [UPNA Head Pose Dataset](http://www.unavarra.es/gi4e/databases/hpdb).<br />
   The dataset needs to be extracted into the folder `datasets/upna`.
2. [IDIAP Head Pose Dataset](https://www.idiap.ch/dataset/headpose/index_html)<br />
   The dataset needs to be extracted into the folder `datasets/IDIAPHeadPose`.
3. [T-Less Dataset](http://cmp.felk.cvut.cz/t-less/)<br />
   We use all objects from the kinect training set for our experiments on t-less. To simplify downloading, we provide the convenience script `datasets/download_tless.sh`.

## Training
Run the following command to generate the lookup table for Bingham Loss training.
```bash
$ python generate_lookup_table.py
```

Before starting training, we need to generate a lookup table for the Bingham loss.  This can be done by running the following script. 
```bash
$ python generate_lookup_table.py
```
You can perform training by selecting a configuration YAML file (or writing a new YAML file). Follow the structure in the [example UPNA training file](configs/baselines/upna/bd_cgs.yaml). After that, run `python train.py --config [config file]`.

In the training portion of the config file, specify the device/gpu number that you would like to use. Also, be sure to name the model you are training! For example if `save_as = practice`, then the `save_dir = /models/practice/` for consistency.

To run all the experiments in our paper,
```bash
$ ./run_training.py
```
## Tensorboard
A tensorboard providing training information can be started via
```bash
$ python -m tensorboard.main --logdir=runs/
```

## Evaluation
To run all the evaluations in our paper,

```bash
$./run_evaluations.py
```

## Citing

If you found this repository useful, please cite [our paper](http://www.gilitschenski.org/igor/publications/202004-iclr-deep_bingham/iclr20-deep_bingham.pdf) presented at ICLR 2020.

```
@inproceedings{Gilitschenski2020,
    title={Deep Orientation Uncertainty Learning based on a Bingham Loss},
    author={Igor Gilitschenski and Roshni Sahoo and Wilko Schwarting and Alexander Amini and Sertac Karaman and Daniela Rus},
    booktitle={International Conference on Learning Representations},
    year={2020}
}
```

                                                                                         
## Acknowledgment
This work was supported in part by NSF Grant 1723943, the Office of Naval Research (ONR) Grant
N00014-18-1-2830, and Toyota Research Institute (TRI). The work solely reflects the opinions
and conclusions of its authors and not TRI, Toyota, or any other Toyota entity. Their support is
gratefully acknowledged.  The structure of the directories and the code is partially based on the 
[Pytorch Template](https://github.com/victoresque/pytorch-template) by Victor Huang.
