from l5kit.geometry import transform_points
from l5kit.visualization import draw_trajectory
import matplotlib.pyplot as plt
import yaml


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


def visualize_data(dataset, idx):
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
    draw_trajectory(im, target_positions_pixels, rgb_color=(255, 0, 255), yaws=data["target_yaws"])
    plt.imshow(im)
    plt.show()