from utils import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import create_chopped_dataset
from torch.utils.data import DataLoader
import os
import numpy as np


class lyft_loader:
    def __init__(self, name, data_path, cfg_path):
        os.environ["L5KIT_DATA_FOLDER"] = data_path
        cfg = load_config_data(cfg_path)
        dm = LocalDataManager(None)

        if name == 'train':
            loader_cfg = cfg["train_data_loader"]
            zarr = ChunkedDataset(dm.require(loader_cfg["key"])).open()
            agents_mask = None

        elif name == 'val':
            loader_cfg = cfg["val_data_loader"]
            zarr_path = dm.require(loader_cfg["key"])
            chop_path = os.path.splitext(zarr_path)[0] + "_chopped_{}".format(loader_cfg["num_frames_to_chop"])

            if not os.path.exists(chop_path):
                eval_base_path = create_chopped_dataset(zarr_path,
                                                        cfg["raster_params"]["filter_agents_threshold"],
                                                        loader_cfg["num_frames_to_chop"],
                                                        cfg["model_params"]["future_num_frames"],
                                                        min_frame_future=10)

            eval_zarr_path = os.path.join(eval_base_path, os.path.basename(zarr_path))
            eval_mask_path = os.path.join(eval_base_path, "mask.npz")
            zarr = ChunkedDataset(eval_zarr_path).open()
            agents_mask = np.load(eval_mask_path)["arr_0"]

        elif name == 'test':
            raise

        rasterizer = build_rasterizer(cfg, dm)
        dataset = AgentDataset(cfg, zarr, rasterizer, agents_mask=agents_mask)
        self.dataloader = DataLoader(dataset, shuffle=loader_cfg["shuffle"], batch_size=loader_cfg["batch_size"],
                                     num_workers=loader_cfg["num_workers"])
        self.loader_iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self.loader_iter)
        except:
            self.loader_iter = iter(self.dataloader)
            data = next(self.loader_iter)
        return data