import math
import torch
import numpy as np
from torch.nn import functional as F


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def pixel_weighted_ce_loss(input, weights, target, bs):
    """
    input (B=2, C=2, 112, 112, 80)
    weights (B=2, 112, 112, 80)
    target (B=2, 112, 112, 80)
    """
    # Calculate log probabilities
    log_soft = F.log_softmax(input)
    # shape
    a, b, c = input.shape[-3], input.shape[-2], input.shape[-1]
    weights = weights.view(bs, 1, a, b, c)
    # Gather log probabilities with respect to target
    log_soft = log_soft.gather(1, target.view(bs, 1, a, b, c))
    # Multiply with weights
    weights = weights.float()
    weighted_log_soft = (log_soft * weights).view(bs, -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_log_soft.sum(1) / weights.view(bs, -1).sum(1)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def ncc_loss(I, J, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    # sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    # sum_filt = torch.ones([1, 1, *win]).to(device)
    sum_filt = torch.ones([1, 1, *win]).cuda()
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross
