import torch
import torch.nn as nn
import torch.optim as optim
import networks
import losses
import os
from collections import OrderedDict


def weights_init(m, type='xavier'):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif type == 'orthogonal':
            nn.init.orthogonal_(m.weight)
        elif type == 'gaussian':
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CreateModel:
    def __init__(self, model_params):
        self.model_params = model_params
        self.train_params = None
        self.device = model_params.device

        # Get the number of input channel
        in_channel = 3 + (model_params.history_num_frames + 1) * 2
        # Get the dimension of output features
        out_features = 2 * model_params.future_num_frames

        self.model = getattr(networks, model_params.model_architecture)
        self.model = self.model(in_channel, image_size=224,
                                out_features=out_features, use_pool=True,
                                use_dropout=False)
        if model_params.loss == "NLL":
            self.criterion = getattr(losses, "neg_multi_log_likelihood_single")

        # Attach to device
        self.model.to(self.device)

        self.lr = None
        self.loss = 0
        self.running_loss = 0
        self.running_batch = 0

        self.state_names = ['loss', 'lr']

    def train_setup(self, train_params):
        self.train_params = train_params
        self.lr = train_params.lr
        self.running_loss = 0
        self.running_batch = 0

        if self.train_params.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
        elif self.train_params.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-5, amsgrad=True)
        else:
            raise RuntimeError("Invalid optimizer: {}".formate(self.train_params.optimizer))

        if self.train_params.scheduler == "steps":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=self.train_params.optimizer_milestones,
                                                            gamma=self.train_params.step_gamma,
                                                            last_epoch=self.train_params.continue_epoch)
        elif self.train_params.scheduler == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  T_max=self.train_params.epochs,
                                                                  eta_min=self.lr / 1000,
                                                                  last_epoch=self.train_params.continue_epoch)
        elif self.train_params.scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                            T_0=self.train_params.scheduler_period,
                                                                            T_mult=self.train_params.get('scheduler_t_mult', 1),
                                                                            eta_min=self.lr / 1000.0,
                                                                            last_epoch=-1)
        else:
            raise RuntimeError("Invalid scheduler: {}".format(self.train_params.scheduler))

        # self.model.apply(weights_init)
        # Switch to training mode
        self.model.train()

    def scheduler_step(self):
        self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def get_output(self, image):
        image = image.to(self.device)
        outputs = self.model(image)
        return outputs

    def forward(self, data):
        inputs = data["image"].to(self.device)
        target_availabilities = data["target_availabilities"].to(self.device)
        targets = data["target_positions"].to(self.device)
        # Forward pass
        outputs = self.model(inputs).reshape(targets.shape)
        loss = self.criterion(targets.float(), outputs.float(), target_availabilities.float())
        return loss, outputs

    def optimize(self, data):
        loss, _ = self.forward(data)

        self.optimizer.zero_grad()
        loss.backward()

        self.running_loss += loss.item()
        self.running_batch += len(data["image"])
        self.loss = self.running_loss / self.running_batch

        # Apply gradient clipping avoid gradient exploding
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_params.get("grad_clip", 2))

        self.optimizer.step()

    def reset_running_states(self):
        self.running_loss = 0
        self.running_batch = 0

    def get_current_states(self):
        states_ret = OrderedDict()
        for name in self.state_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                states_ret[name] = float(getattr(self, name))
        return states_ret

    def save_model(self, which_step, which_epoch):
        save_filename = '{}_step{}_epoch{}.pth'.format(self.model_params.model_architecture, which_step, which_epoch)
        save_path = os.path.join(self.model_params.checkpoints_dir, save_filename)

        if self.model_params.use_cuda and torch.cuda.is_available():
            try:
                torch.save(self.model.module.cpu().state_dict(), save_path)
            except:
                torch.save(self.model.cpu().state_dict(), save_path)
        else:
            torch.save(self.model.cpu().state_dict(), save_path)

        self.model.to(self.device)

    def load_model(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

