import conf
from torch.utils.data import DataLoader
from .conteda_learner import Learner_base

from utils import memory

from utils.loss_functions import *

device = torch.device(
    "cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu"
)

class CONTEDA(Learner_base):
    def __init__(self, *args, **kwargs):
        super(CONTEDA, self).__init__(*args, **kwargs)
        self.fix_net_train_BN()
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                # train the FC layer
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

    def train_online(self, train_set, current_num_sample):
        """
        Train the model
        """

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, "previous_train_loss"):
            self.previous_train_loss = 0

        if current_num_sample > len(train_set[0]):
            return FINISHED

        # Add a sample
        feats, cls, dls = train_set
        current_sample = (
            feats[current_num_sample - 1],
            cls[current_num_sample - 1],
            dls[current_num_sample - 1],
        )

        self.mems = []

        if
            memory.FIFO(capacity=conf.args.memory_size)

        with torch.no_grad():
            self.net.eval()

            if conf.args.memory_type in ["FIFO", "Reservoir"]:
                self.mem.add_instance(current_sample)

            elif conf.args.memory_type in ["PBRS"]:
                f, c, d = (
                    current_sample[0].to(device),
                    current_sample[1].to(device),
                    current_sample[2].to(device),
                )

                logit = self.net(f.unsqueeze(0))
                pseudo_cls = logit.max(1, keepdim=False)[1][0]
                self.mem.add_instance([f, pseudo_cls, d, c, 0])

        if conf.args.use_learned_stats:  # batch-free inference
            self.evaluation_online(
                current_num_sample,
                [[current_sample[0]], [current_sample[1]], [current_sample[2]]],
                train_set
            )

        if (
            current_num_sample % conf.args.update_every_x != 0
        ):  # train only when enough samples are collected
            if not (
                current_num_sample == len(train_set[0])
                and conf.args.update_every_x >= current_num_sample
            ):  # update with entire data

                self.log_loss_results(
                    "train_online",
                    epoch=current_num_sample,
                    loss_avg=self.previous_train_loss,
                )
                return SKIPPED

        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, _, _ = self.mem.get_memory()
        feats = torch.stack(feats)
        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(
            dataset,
            batch_size=conf.args.opt["batch_size"],
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )

        entropy_loss = HLoss(temp_factor=conf.args.temperature)

        for e in range(conf.args.epoch):

            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)
                preds_of_data = self.net(feats)  # update bn stats

                if conf.args.no_optim:
                    pass  # no optimization
                else:
                    loss = entropy_loss(preds_of_data)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        self.log_loss_results("train_online", epoch=current_num_sample, loss_avg=0)

        return TRAINED
