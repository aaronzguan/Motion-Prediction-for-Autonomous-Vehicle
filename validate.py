import os
import torch
from utils import load_config_data, DotDict, modify_args
from tqdm import tqdm
from loader import lyft_loader
from model import CreateModel
import numpy as np
from l5kit.geometry import transform_points
from l5kit.data import LocalDataManager
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
import matplotlib.pyplot as plt


def run(model, dataloader):
    model.eval()
    progress_bar = tqdm(dataloader)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []
    running_loss = 0
    running_batch = 0
    with torch.no_grad():
        for data in progress_bar:
            loss, outputs = model.forward(data)

            # convert agent coordinates into world offsets
            agents_coords = outputs.cpu().numpy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = transform_points(agents_coords, world_from_agents) - centroids[:, None, :2]

            future_coords_offsets_pd.append(np.stack(coords_offset))
            timestamps.append(data["timestamp"].numpy().copy())
            agent_ids.append(data["track_id"].numpy().copy())

            running_loss += loss.item()
            running_batch += len(data["image"])

    loss = running_loss / running_batch
    agent_ids = np.concatenate(agent_ids)
    timestamps = np.concatenate(timestamps)
    coords = np.concatenate(future_coords_offsets_pd)

    return loss, coords, agent_ids, timestamps


def visualize(model, eval_gt_path, eval_dataset, eval_ego_dataset):
    """
    Visualize prediction results　from the ego (AV) point of view for those frames of interest
    """
    model.eval()

    save_folder = "./visualize"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # build a dict to retrieve future trajectories from GT
    gt_rows = {}
    for row in read_gt_csv(eval_gt_path):
        gt_rows[row["track_id"] + row["timestamp"]] = row["coord"]

    # randomly pick some frames for visualization
    num_frames = 1000
    random_frames = np.random.randint(99, len(eval_ego_dataset) - 1, num_frames)

    for frame_number in random_frames:
        agent_indices = eval_dataset.get_frame_indices(frame_number)
        if not len(agent_indices):
            continue
        print("Visualize frame {}".format(frame_number))
        # get AV point-of-view frame
        data_ego = eval_ego_dataset[frame_number]

        predicted_positions = []
        target_positions = []

        for v_index in agent_indices:
            data_agent = eval_dataset[v_index]

            out_net = model.get_output(torch.from_numpy(data_agent["image"]).unsqueeze(0))
            out_pos = out_net[0].reshape(-1, 2).detach().cpu().numpy()
            # store absolute world coordinates
            predicted_positions.append(transform_points(out_pos, data_agent["world_from_agent"]))
            # retrieve target positions from the GT and store as absolute coordinates
            track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]
            target_positions.append(gt_rows[str(track_id) + str(timestamp)] + data_agent["centroid"][:2])

        # convert coordinates to AV point-of-view so we can draw them
        predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["raster_from_world"])
        target_positions = transform_points(np.concatenate(target_positions), data_ego["raster_from_world"])

        im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
        draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)
        plt.imshow(im_ego)
        plt.savefig(os.path.join(save_folder, "frame_" + str(frame_number) + "_pred.png"))

        im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
        draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
        plt.imshow(im_ego)
        plt.savefig(os.path.join(save_folder, "frame_" + str(frame_number) + "_gt.png"))


if __name__ == "__main__":
    # TODO: modify the checkpoint path
    checkpoint_path = "./checkpoints/resnet28_NLL_20210423-011023/resnet28_step288300_epoch1.pth"
    run_validation = False
    run_visualization = True

    L5KIT_DATA_FOLDER = os.path.abspath("/home/ubuntu/lyft-data")
    os.environ["L5KIT_DATA_FOLDER"] = L5KIT_DATA_FOLDER
    cfg = load_config_data("./agent_motion_config.yaml")

    dataloader = lyft_loader(name="val", cfg=cfg)
    val_loader = dataloader.get_loader()

    model_params = DotDict(cfg["model_params"])
    model_params = modify_args(model_params)
    print(model_params)

    model = CreateModel(model_params)
    model.load_model(checkpoint_path)

    # Run the validation though the evaluation dataset
    if run_validation:
        loss, coords, agent_ids, timestamps = run(model, val_loader)

        print("Evaluation loss: {:.4f}".format(loss))

        # Store the evaluation results into a csv file
        pred_path = "prediction_{}.csv".format(model_params.model_architecture)
        write_pred_csv(pred_path,
                       timestamps=timestamps,
                       track_ids=agent_ids,
                       coords=coords)

    # Get the ground truth for evaluation dataset
    dm = LocalDataManager(None)
    loader_cfg = cfg["val_data_loader"]
    zarr_path = dm.require(loader_cfg["key"])
    eval_base_path = os.path.splitext(zarr_path)[0] + "_chopped_{}".format(loader_cfg["num_frames_to_chop"])
    eval_gt_path = os.path.join(eval_base_path, "gt.csv")

    # Get evaluation metrics
    if run_validation:
        metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
        for metric_name, metric_mean in metrics.items():
            print(metric_name, metric_mean)

    if run_visualization:
        # Build the ego dataset
        rasterizer = build_rasterizer(cfg, dm)
        eval_dataset = dataloader.get_dataset()
        eval_ego_dataset = EgoDataset(cfg, eval_dataset.dataset, rasterizer)

        # Visualization from ego point of view
        visualize(model, eval_gt_path, eval_dataset, eval_ego_dataset)

