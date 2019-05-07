import cv2
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torchvision.utils import make_grid
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models.shm import SHM
from models.loss import PredictionL1Loss, ClassificationLoss
from datasets.adobe_dim import AdobeDIMDataLoader
from utils.metrics import AverageMeter
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class SHMAgent(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger("SHMAgent")
        self.logger.info("Creating SHM architecture and loading pretrained weights...")

        self.model = SHM()
        self.data_loader = AdobeDIMDataLoader(self.config.data_root, self.config.mode, self.config.batch_size)
        self.current_epoch = 0
        self.cuda = torch.cuda.is_available() & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.writer = SummaryWriter(log_dir=self.config.summary_dir, comment='SHM')

    def save_checkpoint(self):
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = 'checkpoint-epoch{}.pth.tar'.format(self.current_epoch)
        torch.save(state, os.path.join(self.config.checkpoint_dir, filename))

    def load_checkpoint(self):
        try:
            if self.config.mode == 'pretrain_tnet':
                if self.config.tnet_checkpoint is not None:
                    filename = os.path.join(self.config.checkpoint_dir, self.config.tnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.current_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif self.config.mode == 'pretrain_mnet':
                if self.config.mnet_checkpoint is not None:
                    filename = os.path.join(self.config.checkpoint_dir, self.config.mnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.current_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif self.config.mode == 'end_to_end' or self.config.mode == 'test':
                if self.config.shm_checkpoint is not None:
                    filename = os.path.join(self.config.checkpoint_dir, self.config.shm_checkpoint)
                    checkpoint = torch.load(filename)
                    self.current_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    return
                if self.config.tnet_checkpoint is not None:
                    filename = os.path.join('experiment', 'tnet', 'checkpoints', self.config.tnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.model.tnet.load_state_dict(checkpoint['state_dict'])
                if self.config.mnet_checkpoint is not None:
                    filename = os.path.join('experiment', 'mnet', 'checkpoints', self.config.mnet_checkpoint)
                    checkpoint = torch.load(filename)
                    self.model.mnet.load_state_dict(checkpoint['state_dict'])
        except OSError as e:
            self.logger.info("No checkpoint exists. Skipping...")
            self.logger.info("**First time to train**")

    def trimap_to_image(self, trimap):
        n, c, h, w = trimap.size()
        if c == 3:
            trimap = torch.argmax(trimap, dim=1, keepdim=False).numpy()
        else:
            trimap = trimap.numpy()
        trimap[trimap==0] = 0
        trimap[trimap==1] = 128
        trimap[trimap==2] = 255
        trimap =  torch.from_numpy(trimap.reshape(n, 1, h, w)).int()
        return trimap

    def alpha_to_image(self, alpha):
        alpha = alpha * 255
        return alpha.int()

    def run(self):
        assert self.config.mode in ['pretrain_tnet', 'pretrain_mnet', 'end_to_end', 'test']
        try:
            if self.config.mode == 'pretrain_tnet':
                self.train_tnet()
            elif self.config.mode == 'pretrain_mnet':
                self.train_mnet()
            elif self.config.mode == 'end_to_end':
                self.train_end_to_end()
            else:
                self.test()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train_tnet(self):
        self.model = self.model.tnet
        self.loss_t = ClassificationLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.config.lr, betas=(0.9, 0.999),
                                    weight_decay=self.config.weight_decay)
        self.load_checkpoint()

        self.model.train()
        self.model = self.model.to(self.device)
        self.loss_t = self.loss_t.to(self.device)
        if self.cuda and self.config.ngpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.config.ngpu)))

        sample_image, sample_trimap_gt, _ = next(iter(self.data_loader.train_loader))
        for epoch in range(self.current_epoch, self.config.max_epoch):
            loss_t_epoch = AverageMeter()

            tqdm_loader = tqdm(self.data_loader.train_loader,
                               total=self.data_loader.train_iterations,
                               desc="Epoch-{}-".format(self.current_epoch + 1))
            for image, trimap_gt, _ in tqdm_loader:
                image, trimap_gt = image.to(self.device), trimap_gt.to(self.device)
                trimap_pre = self.model(image)
                loss_t = self.loss_t(trimap_pre, trimap_gt)

                self.optimizer.zero_grad()
                loss_t.backward()
                self.optimizer.step()

                loss_t_epoch.update(loss_t.item())

            self.current_epoch += 1

            self.writer.add_scalar('pretrain_tnet/loss_classification', loss_t_epoch.val, self.current_epoch)
            if self.current_epoch % self.config.sample_period == 0:
                sample_image, sample_trimap_gt = sample_image.to(self.device), sample_trimap_gt.to(self.device)
                sample_trimap_pre = self.model(sample_image).cpu()
                sample_trimap_pre = self.trimap_to_image(sample_trimap_pre)
                sample_trimap_gt = self.trimap_to_image(sample_trimap_gt)
                self.writer.add_image('pretrain_tnet/sample_trimap_prediction',
                                      make_grid(sample_trimap_pre, nrow=2, normalize=True, scale_each=True),
                                      self.current_epoch)
                self.writer.add_image('pretrain_tnet/sample_trimap_ground_truth',
                                      make_grid(sample_trimap_gt, nrow=2, normalize=True, scale_each=True),
                                      self.current_epoch)
            if self.current_epoch % self.config.checkpoint_period == 0:
                self.save_checkpoint()
            print("Training Results at epoch-" + str(self.current_epoch) + " | " +
                  "loss_classification: " + str(loss_t_epoch.val))

    def train_mnet(self):
        self.model = self.model.mnet
        self.loss_p = PredictionL1Loss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.config.lr, betas=(0.9, 0.999),
                                    weight_decay=self.config.weght_decay)
        self.load_checkpoint()

        self.model.train()
        self.model = self.model.to(self.device)
        self.loss_p = self.loss_p.to(self.device)
        if self.cuda and self.config.ngpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.config.ngpu)))

        sample_image, sample_trimap_gt, sample_alpha_gt = next(iter(self.data_loader.train_loader))
        for epoch in range(self.current_epoch, self.config.max_epoch):
            loss_p_epoch = AverageMeter()
            loss_alpha_epoch = AverageMeter()
            loss_comps_epoch = AverageMeter()

            tqdm_loader = tqdm(self.data_loader.train_loader,
                               total=self.data_loader.train_iterations,
                               desc="Epoch-{}-".format(self.current_epoch + 1))
            for image, trimap_gt, alpha_gt in tqdm_loader:
                image, trimap_gt, alpha_gt = image.to(self.device), trimap_gt.to(self.device), alpha_gt.to(self.device)

                input = torch.cat([image, trimap_gt], dim=1)
                alpha_pre = self.model(input)
                loss_p, loss_alpha, loss_comps = self.loss_p(image, alpha_pre, alpha_gt)

                self.optimizer.zero_grad()
                loss_p.backward()
                self.optimizer.step()

                loss_p_epoch.update(loss_p.item())
                loss_alpha_epoch.update(loss_alpha.item())
                loss_comps_epoch.update(loss_comps.item())

            self.current_epoch += 1

            self.writer.add_scalar('pretrain_mnet/loss_prediction', loss_p_epoch.val, self.current_epoch)
            self.writer.add_scalar('pretrain_mnet/loss_alpha_prediction', loss_alpha_epoch.val, self.current_epoch)
            self.writer.add_scalar('pretrain_mnet/loss_composition', loss_comps_epoch.val, self.current_epoch)
            if self.current_epoch % self.config.sample_period == 0:
                sample_image, sample_trimap_gt, sample_alpha_gt = \
                    sample_image.to(self.device), sample_trimap_gt.to(self.device), sample_alpha_gt.to(self.device)
                sample_input = torch.cat((sample_image, sample_trimap_gt), dim=1)
                sample_alpha_pre = self.model(sample_input).cpu()
                sample_alpha_pre = self.alpha_to_image(sample_alpha_pre)
                sample_alpha_gt = self.alpha_to_image(sample_alpha_pre)
                self.writer.add_image('pretrain_mnet/sample_alpha_prediction',
                                      make_grid(sample_alpha_pre, nrow=2, normalize=True, scale_each=True),
                                      self.current_epoch)
                self.writer.add_image('pretrain_mnet/sample_alpha_ground_truth',
                                      make_grid(sample_alpha_gt, nrow=2, normalize = True, scale_each=True),
                                      self.current_epoch)
            if self.current_epoch % self.config.checkpoint_period == 0:
                self.save_checkpoint()
            print("Training Results at epoch-" + str(self.current_epoch) + " | " +
                  "loss_prediction: " + str(loss_p_epoch.val) + " loss_alpha_prediction: " +
                  str(loss_alpha_epoch.val) + " loss_composition: " + str(loss_comps_epoch.val))

    def train_end_to_end(self):
        self.loss_p = PredictionL1Loss()
        self.loss_t = ClassificationLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.config.lr, betas=(0.9, 0.999),
                                    weight_decay=self.config.weght_decay)
        self.load_checkpoint()

        self.model.train()
        self.model = self.model.to(self.device)
        self.loss_p = self.loss_p.to(self.device)
        self.loss_t = self.loss_t.to(self.device)
        if self.cuda and self.config.ngpu > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.config.ngpu)))

        for epoch in range(self.current_epoch, self.config.max_epoch):
            loss_epoch = AverageMeter()
            loss_p_epoch = AverageMeter()
            loss_alpha_epoch = AverageMeter()
            loss_comps_epoch = AverageMeter()
            loss_t_epoch = AverageMeter()

            tqdm_loader = tqdm(self.data_loader.train_loader,
                               total=self.data_loader.train_iterations,
                               desc="Epoch-{}-".format(self.current_epoch+1))
            for image, trimap_gt, alpha_gt in tqdm_loader:
                image, trimap_gt, alpha_gt = image.to(self.device), trimap_gt.to(self.device), alpha_gt.to(self.device)
                trimap_pre, alpha_pre = self.model(image)
                loss_p, loss_alpha, loss_comps = self.loss_p(image, alpha_pre, alpha_gt)
                loss_t = self.loss_t(trimap_pre, trimap_gt)
                loss = loss_p + self.config.loss_lambda * loss_t

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch.update(loss.item())
                loss_p_epoch.update(loss_p.item())
                loss_alpha_epoch.update(loss_alpha.item())
                loss_comps_epoch.update(loss_comps.item())
                loss_t_epoch.update(loss_t.item())

            self.current_epoch += 1

            self.writer.add_scalar('end_to_end/loss', loss_epoch.val, self.current_epoch)
            self.writer.add_scalar('end_to_end/loss_prediction', loss_p_epoch.val, self.current_epoch)
            self.writer.add_scalar('end_eo_end/loss_alpha_prediction', loss_alpha_epoch.val, self.current_epoch)
            self.writer.add_scalar('end_to_end/loss_composition', loss_comps_epoch.val, self.current_epoch)
            self.writer.add_scalar('end_to_end/loss_classification', loss_t_epoch.val, self.current_epoch)
            if self.current_epoch % self.config.checkpoint_period == 0:
                self.save_checkpoint()
            print("Training Results at epoch-" + str(self.current_epoch) + " | " +
                  "loss: " + str(loss_epoch.val) + " loss_prediction: " + str(loss_p_epoch.val) +
                  " loss_alpha_prediction: " + str(loss_alpha_epoch.val) + " loss_composition: " +
                  str(loss_comps_epoch.val) + " loss_classification: " + str(loss_t_epoch.val))

    def test(self):
        self.load_checkpoint()

        self.model.eval()
        self.model = self.model.to(self.device)

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc="Testing at checkpoint -{}-".format(self.config.checkpoint_file))

        for image_name, image, trimap_gt, alpha_gt in tqdm_loader:
            batch_size = len(image_name)
            trimap_pre, alpha_pre = self.model(image)
            for i in range(batch_size):
                cv2.imwrite(os.path.join(self.config.out_dir, '{}_trimap.png'.format(image_name[i][:-4])),
                            trimap_pre[i])
                cv2.imwrite(os.path.join(self.config.out_dir, '{}_alpha.png'.format(image_name[i][:-4])),
                            alpha_pre[i])
        print('Test finished')

    def finalize(self):
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.writer.close()
        self.data_loader.finalize()