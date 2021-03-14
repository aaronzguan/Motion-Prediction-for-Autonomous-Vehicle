import os
from loader import lyft_loader
from utils import load_config_data, DotDict, modify_args, save_log
from model import CreateModel
import tqdm

if __name__ == '__main__':
    L5KIT_DATA_FOLDER = os.path.abspath("data")
    os.environ["L5KIT_DATA_FOLDER"] = L5KIT_DATA_FOLDER
    cfg = load_config_data("./agent_motion_config.yaml")
    train_loader = lyft_loader(name="train", cfg=cfg).get_loader()

    model_params = DotDict(cfg["model_params"])
    model_params = modify_args(model_params)
    # print(model_params)

    model = CreateModel(model_params)
    train_params = DotDict(cfg["train_params"])
    model.train_setup(train_params)

    pbar = tqdm(range(1, train_params.epochs + 1), ncols=0)

    for epoch in pbar:
        model.train()
        for i, data in enumerate(train_loader):
            model.optimize_parameters(data)

        if epoch % train_params.check_freq == 0:
            states = model.get_current_states()
            description = '[{}|{}] '.format(epoch, train_params.epochs)
            for name, value in states.items():
                description += '{}: {:.4f} '.format(name, value)
            pbar.set_description(desc=description)
            save_log(description, checkpoint_dir=model_params.checkpoints_dir)

        model.scheduler_step()



