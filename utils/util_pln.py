import cv2
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage

def get_largest_component_sitk(prediction):
    segmentation = sitk.GetImageFromArray(prediction)
    cc = sitk.ConnectedComponent(segmentation, True)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, segmentation)
    largestCClabel = 0
    largestCCsize = 0
    for l in stats.GetLabels():
        if int(stats.GetPhysicalSize(l)) >= largestCCsize:
            largestCCsize = int(stats.GetPhysicalSize(l))
            largestCClabel = l
    largestCC = cc == largestCClabel  # get the largest component
    return sitk.GetArrayFromImage(largestCC).astype(np.int32)


def cv_morphlogy(prediction, kernel_size, open_iteration, close_iteration=1, open=True, close=False):
    prediction = prediction.astype(np.uint8)
    if open:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel, iterations=open_iteration)
    if close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel, iterations=close_iteration)
    return prediction


def reg_postprocess(pred_lbl_slc, slice_distance, post_morph=True):
    if post_morph:
        pred_lbl_slc = pred_lbl_slc.astype(np.int)
        if sum(sum(pred_lbl_slc)) != 0:
            pred_lbl_slc = get_largest_component_sitk(pred_lbl_slc)
        pred_lbl_slc = pred_lbl_slc.astype(np.uint8)
        if slice_distance <= 10:
            pred_lbl_slc = cv_morphlogy(pred_lbl_slc, 5, 1)
        else:
            pred_lbl_slc = cv_morphlogy(pred_lbl_slc, 5, 5, close=True)
    return pred_lbl_slc


def regnet_test(slice_volume_batch, slice_label_batch, reg_unet, reg_stn_label, pred_shape, num_slice, lbl_idx):
    reg_pred = np.zeros(pred_shape)
    for i in range(pred_shape[0]):
        start_i, curr_lbl_idx, end_i = num_slice * i, lbl_idx[i], num_slice * (i + 1)
        lbl_idx_i = start_i + curr_lbl_idx
        # the labeled slice (gt)
        reg_pred[i, ..., curr_lbl_idx] = slice_label_batch[lbl_idx_i]

        # moving label
        input_label = torch.from_numpy(slice_label_batch[np.newaxis, np.newaxis, lbl_idx_i, ...]).cuda().float()
        for idx in range(lbl_idx_i - 1, start_i - 1, -1):
            input_moving = torch.from_numpy(slice_volume_batch[np.newaxis, np.newaxis, idx + 1, ...]).cuda().float()
            input_fixed = torch.from_numpy(slice_volume_batch[np.newaxis, np.newaxis, idx, ...]).cuda().float()
            pred_flow = reg_unet(input_moving, input_fixed)
            pred_label = reg_stn_label(input_label, pred_flow)
            # post process
            pred_lbl_slc = reg_postprocess(pred_label[0, 0, ...].cpu().detach().numpy(), lbl_idx_i - idx, post_morph=True)
            reg_pred[i, ..., idx - start_i] = pred_lbl_slc
            # next moving label
            input_label = torch.from_numpy(pred_lbl_slc[np.newaxis, np.newaxis, ...]).cuda().float()

        # moving label
        input_label = torch.from_numpy(slice_label_batch[np.newaxis, np.newaxis, lbl_idx_i, ...]).cuda().float()
        for idx in range(lbl_idx_i + 1, end_i):
            input_moving = torch.from_numpy(slice_volume_batch[np.newaxis, np.newaxis, idx - 1, ...]).cuda().float()
            input_fixed = torch.from_numpy(slice_volume_batch[np.newaxis, np.newaxis, idx, ...]).cuda().float()
            pred_flow = reg_unet(input_moving, input_fixed)
            pred_label = reg_stn_label(input_label, pred_flow)
            # post process
            pred_lbl_slc = reg_postprocess(pred_label[0, 0, ...].cpu().detach().numpy(), idx - lbl_idx_i, post_morph=True)
            reg_pred[i, ..., idx - start_i] = pred_lbl_slc
            # next moving label
            input_label = torch.from_numpy(pred_lbl_slc[np.newaxis, np.newaxis, ...]).cuda().float()
    return reg_pred


def vxm_data_generator(num_slice, x_data, x_label=None, batch_size=32):
    if x_label is None:
        while True:
            idx1 = np.random.randint(0, x_data.shape[0] - 1, size=batch_size)
            moving_images = x_data[idx1, np.newaxis, ...]
            idx2 = []
            for i in idx1:
                if (i + 1) % num_slice == 0:
                    idx2.append(i - 1)
                elif i % num_slice == 0:
                    idx2.append(i + 1)
                else:
                    idx2.append(i + np.random.choice((-1, 1)))
            fixed_images = x_data[idx2, np.newaxis, ...]
            inputs = [moving_images, fixed_images]
            yield inputs
    else:
        while True:
            idx1 = np.random.randint(0, x_data.shape[0] - 1, size=batch_size)
            moving_images = x_data[idx1, np.newaxis, ...]
            moving_labels = x_label[idx1, np.newaxis, ...]
            idx2 = []
            for i in idx1:
                if (i + 1) % num_slice == 0:
                    idx2.append(i - 1)
                elif i % num_slice == 0:
                    idx2.append(i + 1)
                else:
                    idx2.append(i + np.random.choice((-1, 1)))
            fixed_images = x_data[idx2, np.newaxis, ...]
            fixed_labels = x_label[idx2, np.newaxis, ...]
            inputs_images = [moving_images, fixed_images]
            inputs_labels = [moving_labels, fixed_labels]
            yield inputs_images, inputs_labels


# find labeled slices
def label_index(slice_label_batch, labeled_bs, num_slice):
    lbl_idx = np.zeros(labeled_bs).astype(int)
    for i in range(labeled_bs):
        for j in range(num_slice):
            if sum(sum(slice_label_batch[i, ..., j])) != 0:
                lbl_idx[i] = j
    return lbl_idx


def get_ce_weight(label_batch, reg_pred, seg_pred, labeled_bs, iter_num, lbl_idx, num_slice, alpha_d):
    weight = torch.zeros_like(label_batch[:labeled_bs])
    if iter_num < 1000:
        w_consist = np.ones(label_batch[:labeled_bs].shape)
    else:
        consist = (reg_pred != seg_pred).astype(int)
        w = 2 * pow((iter_num - 1000) / 1e5, 0.5)
        w_consist = consist * w + 1

    for i in range(labeled_bs):
        curr_lbl_idx = lbl_idx[i]
        slice_weight = [pow(alpha_d, abs(j - curr_lbl_idx)) for j in range(num_slice)]
        for j in range(num_slice):
            weight[i, ..., j] = torch.from_numpy(w_consist[i, ..., j] * slice_weight[j]).float().cuda()
    return weight


def get_prediction(reg_pred, w_p, ema_outputs_soft, lbl_idx, labeled_bs):
    y = ema_outputs_soft[:labeled_bs, 1, :, :, :].cpu().data.numpy()
    prediction_prob = reg_pred * (1 - w_p) + y * w_p
    prediction = np.int64(prediction_prob > 0.5)

    seg_pred = np.zeros(reg_pred.shape)
    for i in range(labeled_bs):
        curr_lbl_idx = lbl_idx[i]
        y_i = ema_outputs_soft[i].cpu().data.numpy()
        label_map = np.argmax(y_i, axis=0)
        label_map[..., curr_lbl_idx] = reg_pred[i, ..., curr_lbl_idx]
        seg_pred[i] = ndimage.binary_fill_holes(label_map)
        prediction[i] = ndimage.binary_fill_holes(prediction[i])
        prediction[i, ..., curr_lbl_idx] = reg_pred[i, ..., curr_lbl_idx]
    return seg_pred, prediction


def save_images(volume_batch, label_batch, label_full_batch, reg_pred, seg_pred, prediction, snapshot_path, iter_num):
    nib.save(nib.Nifti1Image(volume_batch[0, 0, ...].cpu().data.numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/img_' + str(iter_num) + '.nii.gz')
    nib.save(nib.Nifti1Image(label_full_batch[0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/gt_full_' + str(iter_num) + '.nii.gz')
    nib.save(nib.Nifti1Image(label_batch[0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/gt_' + str(iter_num) + '.nii.gz')
    nib.save(nib.Nifti1Image(reg_pred[0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/reg_pred_' + str(iter_num) + '.nii.gz')
    nib.save(nib.Nifti1Image(seg_pred[0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/seg_pred_' + str(iter_num) + '.nii.gz')
    nib.save(nib.Nifti1Image(prediction[0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/pred_' + str(iter_num) + '.nii.gz')

