from l5kit.geometry import transform_points
from l5kit.visualization import draw_trajectory
import matplotlib.pyplot as plt
import yaml
import datetime
import torch
import os


def load_config_data(path):
    with open(path) as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


class DotDict(dict):
    """dot.notation access to dictionary attributes
    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def modify_args(args):
    args.device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.checkpoints_dir = os.path.join("./checkpoints",
                                        '{}_{}_{}'.format(args.model_architecture, args.loss, current_time))
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    return args


def save_log(message, checkpoint_dir):
    log_name = os.path.join(checkpoint_dir, "log.txt")
    with open(log_name, "a") as log_file:
        log_file.write(message + '\n')


def visualize_data(dataset, idx):
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
    draw_trajectory(im, target_positions_pixels, rgb_color=(255, 0, 255), yaws=data["target_yaws"])
    plt.imshow(im)
    plt.show()