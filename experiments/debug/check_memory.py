import argparse
import time
import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        model = self.register_model(model)

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
