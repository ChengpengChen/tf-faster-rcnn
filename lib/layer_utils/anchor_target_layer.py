# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  if cfg.TRAIN.RPN_FL_CE_FUSION:
    assert cfg.TRAIN.RPN_FL_SOFTMAX, 'the fusion mode only work with softmax model'

    rpn_labels_fl, rpn_bbox_targets_fl, rpn_bbox_inside_weights_fl, rpn_bbox_outside_weights_fl = \
      _anchor_target_layer_fl(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors)
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      _anchor_target_layer_ori(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors)

    if cfg.TRAIN.RPN_FUSION_CE_OR_FL:
      return rpn_labels_fl, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    else:
      return rpn_labels_fl, rpn_labels, rpn_bbox_targets_fl, rpn_bbox_inside_weights_fl, rpn_bbox_outside_weights_fl
  else:
    # not fusion, select fl/ce first
    anchor_tartget_layer_select = _anchor_target_layer_ori if not cfg.TRAIN.RPN_FL_ENABLE \
                                else _anchor_target_layer_fl

    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_tartget_layer_select(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors)
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _anchor_target_layer_fl(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """
  sample anchor for RPN in the mode of focal loss, in which much more sample should be selected
  refer to _anchor_target_layer_ori()
  by ccp, on 01/16/2018
  """
  # fl_fg_num = cfg.TRAIN.RPN_FL_FG_NUM
  fl_ratio = cfg.TRAIN.RPN_FL_RATIO

  A = num_anchors
  total_anchors = all_anchors.shape[0]
  K = total_anchors / num_anchors

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  height, width = rpn_cls_score.shape[1:3]

  rpn_labels = []
  rpn_bbox_targets = []
  rpn_bbox_inside_weights = []
  rpn_bbox_outside_weights = []
  for im_i in np.arange(im_info.shape[0]):
    # only keep anchors inside the image
    inds_inside = np.where(
      (all_anchors[:, 0] >= -_allowed_border) &
      (all_anchors[:, 1] >= -_allowed_border) &
      (all_anchors[:, 2] < im_info[im_i, 1] + _allowed_border) &  # width
      (all_anchors[:, 3] < im_info[im_i, 0] + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    gt_boxes_inds = np.where(gt_boxes[:, 0] == im_i)[0]
    gt_boxes_im_i = gt_boxes[gt_boxes_inds, 1:]

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
      np.ascontiguousarray(anchors, dtype=np.float),
      np.ascontiguousarray(gt_boxes_im_i, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
      # assign bg labels first so that positive labels can clobber them
      # first set the negatives
      labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
      # assign bg labels last so that negative labels can clobber positives
      labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # rois_per_image = cfg.TRAIN.RPN_BATCHSIZE / cfg.TRAIN.IMS_PER_BATCH
    # subsample positive labels if we have too many
    # num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * rois_per_image)
    fl_fg_num = cfg.TRAIN.RPN_FL_FG_NUM
    fg_inds = np.where(labels == 1)[0]
    if fl_fg_num > 0:
      if len(fg_inds) > fl_fg_num:
        disable_inds = npr.choice(
          fg_inds, size=(len(fg_inds) - fl_fg_num), replace=False)
        labels[disable_inds] = -1
    else:
      fl_fg_num = len(fg_inds)

    # subsample negative labels if we have too many
    # num_bg = int(rois_per_image - np.sum(labels == 1))
    fl_bg_num = fl_ratio * fl_fg_num
    bg_inds = np.where(labels == 0)[0]
    if fl_bg_num > 0:
      if len(bg_inds) > fl_bg_num:
        disable_inds = npr.choice(
          bg_inds, size=(len(bg_inds) - fl_bg_num), replace=False)
        labels[disable_inds] = -1
    else:
      fl_bg_num = len(bg_inds)
    # print(fl_fg_num)
    # print(fl_bg_num)

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes_im_i[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
      # uniform weighting of examples (given non-uniform sampling)
      # num_examples = np.sum(labels >= 0)
      positive_weights = np.ones((1, 4)) * 1.0 / 2.0/np.sum(labels == 1)
      negative_weights = np.ones((1, 4)) * 1.0 / 2.0/np.sum(labels == 1)
      # positive_weights = np.ones((1, 4)) * 1.0 / 2.0/np.maximum(np.sum(labels == 1), 64)
      # negative_weights = np.ones((1, 4)) * 1.0 / 2.0/np.maximum(np.sum(labels == 1), 64)
      positive_weights = np.ones((1, 4)) * 1.0 / 128
      negative_weights = np.ones((1, 4)) * 1.0 / 128
      # print(np.sum(labels == 1))

    else:
      assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
              (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
      positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                          np.sum(labels == 1))
      negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                          np.sum(labels == 0))

    # sample fixed number(128) rois for regression, 05/21/2018
    labels_tmp = labels.copy()
    # fg_inds = np.where(labels_tmp == 1)[0]
    # if len(fg_inds) > 128:
    #   disable_inds = npr.choice(
    #     fg_inds, size=(len(fg_inds) - 128), replace=False)
    #   labels_tmp[disable_inds] = -1
    bbox_outside_weights[labels_tmp == 1, :] = positive_weights
    bbox_outside_weights[labels_tmp == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels.append(labels)

    # bbox_targets
    bbox_targets = bbox_targets \
      .reshape((1, height, width, A * 4))

    rpn_bbox_targets.append(bbox_targets)
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
      .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights.append(bbox_inside_weights)

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
      .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights.append(bbox_outside_weights)

  rpn_labels = np.concatenate(rpn_labels)
  rpn_bbox_targets = np.concatenate(rpn_bbox_targets)
  rpn_bbox_inside_weights = np.concatenate(rpn_bbox_inside_weights)
  rpn_bbox_outside_weights = np.concatenate(rpn_bbox_outside_weights)

  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _anchor_target_layer_ori(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """Same as the anchor target layer in original Fast/er RCNN """
  A = num_anchors
  total_anchors = all_anchors.shape[0]
  K = total_anchors / num_anchors

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  height, width = rpn_cls_score.shape[1:3]

  rpn_labels = []
  rpn_bbox_targets = []
  rpn_bbox_inside_weights = []
  rpn_bbox_outside_weights = []
  for im_i in np.arange(im_info.shape[0]):
    # only keep anchors inside the image
    inds_inside = np.where(
      (all_anchors[:, 0] >= -_allowed_border) &
      (all_anchors[:, 1] >= -_allowed_border) &
      (all_anchors[:, 2] < im_info[im_i, 1] + _allowed_border) &  # width
      (all_anchors[:, 3] < im_info[im_i, 0] + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    gt_boxes_inds = np.where(gt_boxes[:, 0] == im_i)[0]
    gt_boxes_im_i = gt_boxes[gt_boxes_inds, 1:]

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
      np.ascontiguousarray(anchors, dtype=np.float),
      np.ascontiguousarray(gt_boxes_im_i, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
      # assign bg labels first so that positive labels can clobber them
      # first set the negatives
      labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
      # assign bg labels last so that negative labels can clobber positives
      labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    rois_per_image = cfg.TRAIN.RPN_BATCHSIZE / cfg.TRAIN.IMS_PER_BATCH
    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * rois_per_image)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
      disable_inds = npr.choice(
        fg_inds, size=(len(fg_inds) - num_fg), replace=False)
      labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = int(rois_per_image - np.sum(labels == 1))
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
      disable_inds = npr.choice(
        bg_inds, size=(len(bg_inds) - num_bg), replace=False)
      labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes_im_i[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
      # uniform weighting of examples (given non-uniform sampling)
      num_examples = np.sum(labels >= 0)
      positive_weights = np.ones((1, 4)) * 1.0 / num_examples
      negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
      assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
              (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
      positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                          np.sum(labels == 1))
      negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                          np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels.append(labels)

    # bbox_targets
    bbox_targets = bbox_targets \
      .reshape((1, height, width, A * 4))

    rpn_bbox_targets.append(bbox_targets)
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
      .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights.append(bbox_inside_weights)

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
      .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights.append(bbox_outside_weights)

  rpn_labels = np.concatenate(rpn_labels)
  rpn_bbox_targets = np.concatenate(rpn_bbox_targets)
  rpn_bbox_inside_weights = np.concatenate(rpn_bbox_inside_weights)
  rpn_bbox_outside_weights = np.concatenate(rpn_bbox_outside_weights)

  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
