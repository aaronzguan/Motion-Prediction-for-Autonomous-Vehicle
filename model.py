import torch
import torch.nn as nn
import torch.optim as optim
import networks
import losses
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace


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


class CreateModel(nn.Module):
    def __init__(self, args, class_num):
        super(CreateModel, self).__init__()
        self.args =
        self.device =

        model_name = args["model_params"]["model_architecture"]
        loss_name = args["model_params"]["loss"]
        self.model = getattr(networks, model_name)()
        self.criterion = getattr(losses, loss_name)
        self.train_params =

    def train_setup(self):
        self.lr =
        self.checkpoints_dir =
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=5e-5, amsgrad=True)
        if self.train_params.scheduler == "steps":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.train_params.optimiser_milestones,
                gamma=0.2,
                last_epoch=continue_epoch,
            )
        elif self.train_params.scheduler == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=nb_epochs,
                eta_min=initial_lr / 1000,
                last_epoch=continue_epoch,
            )
        elif self.train_params.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.train_params.scheduler_period,
                T_mult=self.train_params.get('scheduler_t_mult', 1),
                eta_min=initial_lr / 1000.0,
                last_epoch=-1
            )
        else:
            raise RuntimeError("Invalid scheduler name")

        self.model.apply(weights_init)
        # Switch to training mode
        self.model.train()

    def scheduler_step(self):
        self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def optimize_parameters(self, data):
        inputs = data["image"].to(self.device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(self.device)
        targets = data["target_positions"].to(self.device)
        # Forward pass
        outputs = self.model(inputs).reshape(targets.shape)
        loss = self.criterion(outputs, targets)
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, which_epoch):

