import os
from loader import lyft_loader
from utils import load_config_data, DotDict, modify_args, save_log
from model import CreateModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import validate

if __name__ == '__main__':
    L5KIT_DATA_FOLDER = os.path.abspath("/home/ubuntu/lyft-data")
    os.environ["L5KIT_DATA_FOLDER"] = L5KIT_DATA_FOLDER
    cfg = load_config_data("./agent_motion_config.yaml")
    train_loader = lyft_loader(name="train", cfg=cfg).get_loader()
    val_loader = lyft_loader(name="val", cfg=cfg).get_loader()

    model_params = DotDict(cfg["model_params"])
    model_params = modify_args(model_params)
    print(model_params)

    logger = SummaryWriter(log_dir=model_params.checkpoints_dir)

    model = CreateModel(model_params)
    train_params = DotDict(cfg["train_params"])
    model.train_setup(train_params)

    pbar = tqdm(range(1, train_params.epochs + 1), ncols=0)
    iters = 0
    for epoch in pbar:
        model.train()
        model.reset_running_states()

        data_iter = tqdm(train_loader, position=0, leave=True, ascii=True)
        for batch_idx, data in enumerate(data_iter):
            model.optimize(data)

            if iters % train_params.log_every_n_step == 0:
                states = model.get_current_states()
                logger.add_scalar('loss', states['loss'], iters)
                logger.add_scalar('lr', states['lr'], iters)

            if iters % train_params.check_freq == 0:
                states = model.get_current_states()
                description = 'Epoch {} Batch {}/{} ({:.0f}%) '.format(epoch, batch_idx, len(train_loader),
                                                                       100 * batch_idx / len(train_loader))
                for name, value in states.items():
                    description += '{}: {:.4f} '.format(name, value)
                data_iter.set_description(desc=description)
                save_log(description, checkpoint_dir=model_params.checkpoints_dir)
                model.save_model(iters, epoch)

            iters += 1

        if epoch % train_params.eval_freq == 0:
            eval_loss, _, _, _ = validate.run(model, dataloader=val_loader)
            description = 'Eval Epoch: {}/{} loss: {:.4f}'.format(epoch, train_params.epochs, eval_loss)
            pbar.set_description(desc=description)
            save_log(description, checkpoint_dir=model_params.checkpoints_dir)

        model.scheduler_step()

    logger.flush()



