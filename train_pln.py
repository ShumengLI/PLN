import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.vnet import VNet
from networks.unet import UNet
from networks.spatial import SpatialTransformer
from utils import ramps, losses
from utils.util_pln import *
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from dataloaders.lits import LiTS
from dataloaders.kits import KiTS

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/LA/processed_h5/', help='data load root path')
parser.add_argument('--exp', type=str, default='pln', help='name of experiment')
parser.add_argument('--dataset', type=str, default='la', help='dataset to use')
parser.add_argument('--label_num', type=int, default=16, help='number of labeled samples')

parser.add_argument('--pretrain_reg_epoch', type=int, default=200, help='pretrain epoch number for reg module')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled samples batch size')
parser.add_argument('--reg_iter', type=int, default=10, help='number of registration training per iter')

parser.add_argument('--seg_lr', type=float, default=0.01, help='seg learning rate')
parser.add_argument("--reg_lr", type=float, default=0.0001, help="reg learning rate")
parser.add_argument("--sim_loss", type=str, default='mse', help="similarity loss: mse or ncc")
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema decay')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency rampup')

parser.add_argument('--w_p_max', type=float, default=0.9, help='max weight of seg pred')
parser.add_argument('--alpha_p', type=float, default=0.2, help='final pseudo-label cal: seg net prediction power')
parser.add_argument('--alpha_d', type=float, default=0.8, help='slice-wise weight')
parser.add_argument('--save_img', type=int, default=6000, help='img saving iterations')
parser.add_argument('--w_dice', type=float, default=0.3, help='dice loss')

parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model_" + args.dataset + "/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.seg_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + 'saveimg')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.dataset == 'la':
        num_classes = 2
        patch_size = (112, 112, 80)
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                              RandomRotFlip(),
                              RandomCrop(patch_size),
                              ToTensor(),
                              ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 80))

    elif args.dataset == 'lits':
        num_classes = 2
        patch_size = (176, 176, 64)
        db_train = LiTS(base_dir=train_data_path,
                        split='train',
                        transform=transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                        ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 100))

    elif args.dataset == 'kits':
        num_classes = 2
        patch_size = (176, 176, 64)
        db_train = KiTS(base_dir=train_data_path,
                        split='train',
                        transform=transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                        ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 190))

    reg_size = (patch_size[0], patch_size[1])
    num_slice = patch_size[2]
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def create_regnet(vol_size):
        nf_enc = [16, 32, 32, 32]
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
        net = UNet(len(vol_size), nf_enc, nf_dec).cuda()
        return net

    def create_model(ema=False):
        model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # Network definition
    reg_unet = create_regnet(reg_size)
    reg_stn = SpatialTransformer(reg_size).cuda()
    reg_stn_label = SpatialTransformer(reg_size, mode="nearest").cuda()

    model = create_model()
    ema_model = create_model(ema=True)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    reg_unet.train()
    reg_stn.train()
    reg_stn_label.eval()
    model.train()
    ema_model.train()

    # Set optimizer and losses
    reg_optimizer = optim.Adam(reg_unet.parameters(), lr=args.reg_lr)
    seg_optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    consistency_criterion = losses.softmax_mse_loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = args.pretrain_reg_epoch + max_iterations // len(trainloader) + 1
    lr_ = base_lr

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, label_full_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label_full']

            ### Train Registration Module
            slice_volume_batch = volume_batch.cpu().detach().numpy()
            slice_volume_batch = np.squeeze(slice_volume_batch, 1)
            slice_volume_batch = np.transpose(slice_volume_batch, (0, 3, 1, 2))
            slice_volume_batch = slice_volume_batch.reshape((-1, slice_volume_batch.shape[2], slice_volume_batch.shape[3]))

            slice_label_batch = label_batch.cpu().detach().numpy()
            slice_label_batch = slice_label_batch[:labeled_bs]
            lbl_idx = label_index(slice_label_batch, labeled_bs, num_slice)         # get labeled slice index
            slice_label_batch = np.transpose(slice_label_batch, (0, 3, 1, 2))
            slice_label_batch = slice_label_batch.reshape((-1, slice_label_batch.shape[2], slice_label_batch.shape[3]))

            train_generator = vxm_data_generator(num_slice, x_data=slice_volume_batch, x_label=None, batch_size=32)
            for i in range(1, args.reg_iter + 1):
                input_sample = next(train_generator)
                input_moving, input_fixed = torch.from_numpy(input_sample[0]).cuda().float(), torch.from_numpy(input_sample[1]).cuda().float()

                flow_m2f = reg_unet(input_moving, input_fixed)
                m2f = reg_stn(input_moving, flow_m2f)

                # Calculate loss
                reg_loss = sim_loss_fn(m2f, input_fixed)

                reg_optimizer.zero_grad()
                reg_loss.backward()
                reg_optimizer.step()

            if epoch_num >= args.pretrain_reg_epoch:
                # Generate registration prediction
                reg_unet.eval()
                reg_pred = regnet_test(slice_volume_batch, slice_label_batch, reg_unet, reg_stn_label, label_batch[:labeled_bs].shape, num_slice, lbl_idx)
                reg_unet.train()

                ### Train Semi-supervised Segmentation Module
                volume_batch = volume_batch.cuda()
                noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = volume_batch + noise

                outputs = model(volume_batch)
                outputs_soft = F.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_outputs = ema_model(ema_inputs)
                    ema_outputs_soft = F.softmax(ema_outputs, dim=1)

                ### Parasitic-like Mechanism
                # Pseudo-labels Generation
                w_p = args.w_p_max * pow(iter_num / max_iterations, args.alpha_p)
                seg_pred, prediction = get_prediction(reg_pred, w_p, ema_outputs_soft, lbl_idx, labeled_bs)

                # Save imgs
                if iter_num % args.save_img == 0:
                    save_images(volume_batch, label_batch, label_full_batch, reg_pred, seg_pred, prediction, snapshot_path, iter_num)

                ### Guidance for Segmentation Module
                # Calculate loss
                label_batch[:labeled_bs] = torch.from_numpy(prediction).long()
                label_batch = label_batch.cuda()
                weight = get_ce_weight(label_batch, reg_pred, seg_pred, labeled_bs, iter_num, lbl_idx, num_slice, args.alpha_d)
                loss_seg_ce = losses.pixel_weighted_ce_loss(outputs[:labeled_bs], weight, label_batch[:labeled_bs], labeled_bs)
                loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                supervised_loss = 0.5 * (loss_seg_ce + loss_seg_dice * args.w_dice)

                consistency_weight = get_current_consistency_weight(iter_num//150)
                consistency_dist = consistency_criterion(outputs, ema_outputs)
                consistency_dist = torch.mean(consistency_dist)
                consistency_loss = consistency_weight * consistency_dist

                loss = supervised_loss + consistency_loss

                seg_optimizer.zero_grad()
                loss.backward()
                seg_optimizer.step()

                update_ema_variables(model, ema_model, args.ema_decay, iter_num)

                ### Guidance for Registration Module
                # by Dice loss between pseudo-labels and registration predictions
                slice_label_batch = np.transpose(prediction, (0, 3, 1, 2))
                slice_label_batch = slice_label_batch.reshape((-1, slice_label_batch.shape[2], slice_label_batch.shape[3]))

                train_generator = vxm_data_generator(num_slice, x_data=slice_volume_batch[:slice_label_batch.shape[0]], x_label=slice_label_batch, batch_size=32)
                for i in range(1, args.reg_iter + 1):
                    input_sample, input_label = next(train_generator)
                    input_moving, input_fixed = torch.from_numpy(input_sample[0]).cuda().float(), torch.from_numpy(input_sample[1]).cuda().float()
                    input_moving_label, input_fixed_label = torch.from_numpy(input_label[0]).cuda().float(), torch.from_numpy(input_label[1]).cuda().float()

                    flow_m2f = reg_unet(input_moving, input_fixed)
                    m2f = reg_stn(input_moving, flow_m2f)
                    m2f_label = reg_stn(input_moving_label, flow_m2f)

                    # Calculate loss
                    sim_loss = sim_loss_fn(m2f, input_fixed)
                    dice_loss = losses.dice_loss(m2f_label, input_fixed_label)
                    reg_loss = sim_loss + dice_loss

                    reg_optimizer.zero_grad()
                    reg_loss.backward()
                    reg_optimizer.step()

                iter_num = iter_num + 1
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/mt_loss', loss, iter_num)
                writer.add_scalar('loss/loss_seg_ce', loss_seg_ce, iter_num)
                writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
                writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

                logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                             (iter_num, loss.item(), consistency_dist.item(), consistency_weight))

                # change lr
                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in seg_optimizer.param_groups:
                        param_group['lr'] = lr_
                if iter_num % 2000 == 0:
                    save_mode_path_reg = os.path.join(snapshot_path, 'reg_iter_' + str(iter_num) + '.pth')
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(reg_unet.state_dict(), save_mode_path_reg)
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if iter_num >= max_iterations:
                    break
        if iter_num >= max_iterations:
            break
    save_mode_path_reg = os.path.join(snapshot_path, 'reg_iter_' + str(iter_num) + '.pth')
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(reg_unet.state_dict(), save_mode_path_reg)
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
