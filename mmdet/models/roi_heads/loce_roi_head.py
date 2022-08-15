import torch
import numpy as np

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

import torch.distributed as dist
from collections import defaultdict

from mmdet.models.utils import collect_tensor_from_dist, MFS

@HEADS.register_module()
class LoceRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)
        if self.train_cfg:
            # mean score for EBL and MFS
            self.alpha=self.train_cfg.alpha
            self.bg_score=self.train_cfg.bg_score
            self.mean_score = torch.ones(self.bbox_head.num_classes + 1).cuda() * 0.01

            # for MFS
            self.mfs = MFS(num_classes=self.bbox_head.num_classes,
                           queue_size=self.train_cfg.mfs.queue_size,
                           sampled_num_classes=self.train_cfg.mfs.sampled_num_classes,
                           sampled_num_features=self.train_cfg.mfs.sampled_num_features,
                           gpu_statictics=self.train_cfg.mfs.gpu_statictics)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_from_feat(self, bbox_feats):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _get_feat_for_memory(self, x, gt_bboxes, gt_labels, img_metas):
        # get feat e.t. for saving to memoy
        num_imgs = len(gt_labels)

        proposal_list = []
        gt_bbox_list = []
        gt_label_list = []
        neg_proposal_list = []
        for i in range(num_imgs):
            if len(gt_labels[i]) == 0:
                proposal_list.append(gt_bboxes[i].new_zeros(0, 4))
                gt_bbox_list.append(gt_bboxes[i].new_zeros(0, 4))
                gt_label_list.append(gt_labels[i].new_zeros(0))
                neg_proposal_list.append(gt_bboxes[i].new_zeros(0, 4))
                continue

            img_proposal, img_gt_bbox, img_gt_label = self.mfs.bbox_generator(gt_labels[i], gt_bboxes[i], img_metas[i])

            proposal_list.append(img_proposal)
            gt_bbox_list.append(img_gt_bbox)
            gt_label_list.append(img_gt_label)
            neg_proposal_list.append(img_proposal.new_zeros(0, 4))

        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = bbox_feats.detach()  # 0

        bbox_labels, _, bbox_targets, _ = self.bbox_head.get_targets_for_memory(proposal_list, neg_proposal_list,
                                                             gt_bbox_list, gt_label_list,
                                                             self.train_cfg)

        return bbox_feats, bbox_labels, bbox_targets

    def _compute_batch_mean_score(self, cls_score, sampling_results, gt_labels, selected_labels):
        # batch mean score for current sample (non queue sampling samples)
        # here have not to compute mean score which is computed after dist collection
        scores = cls_score.detach().softmax(1)
        batch_gt_labels = []
        batch_mean_scores = []
        for img_ind, sampling_results_img in enumerate(sampling_results):
            for gt_ind, gt_label in enumerate(gt_labels[img_ind]):
                if (sampling_results_img.pos_assigned_gt_inds == gt_ind).sum() > 0:
                    score = scores[self.bbox_sampler.num * img_ind:self.bbox_sampler.num * img_ind + len(sampling_results_img.pos_assigned_gt_inds),
                        gt_label][sampling_results_img.pos_assigned_gt_inds == gt_ind]
                    batch_gt_labels.append(gt_label.unsqueeze(0))
                    batch_mean_scores.append(score.mean().unsqueeze(0))

        # batch mean score for selected queue samples
        selected_length = len(selected_labels)
        for gt_label in set(list(selected_labels.cpu().numpy())):
            score = scores[-selected_length:, gt_label][selected_labels == gt_label]
            batch_gt_labels.append(gt_labels[0].new([gt_label]))
            batch_mean_scores.append(score.mean().unsqueeze(0))

        batch_gt_labels = torch.cat(batch_gt_labels)
        batch_mean_scores = torch.cat(batch_mean_scores)

        return batch_gt_labels, batch_mean_scores

    def _update_mean_score(self, cls_score, sampling_results, gt_labels, selected_labels, rank, alpha=0.9, bg_score=0.01):
        batch_gt_labels, batch_mean_scores = \
                self._compute_batch_mean_score(cls_score, sampling_results, gt_labels, selected_labels)

        if rank != -1:
            batch_gt_labels, batch_mean_scores = \
                collect_tensor_from_dist([batch_gt_labels, batch_mean_scores], [-1, 0])

        # compute the mean score for all collected scores
        # including both current batch samples and sampling samples together
        for gt_label in set(list(batch_gt_labels.cpu().numpy())):
            if gt_label == -1:
                continue
            batch_mean_score = batch_mean_scores[batch_gt_labels == gt_label]
            number_gt = len(batch_mean_score)
            number_alpha = alpha ** number_gt
            self.mean_score[gt_label] = number_alpha * self.mean_score[gt_label] + (
                    1 - number_alpha) * batch_mean_score.mean()

        self.mean_score[-1] = bg_score

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        try:
            rank = dist.get_rank()
        except:
            rank = -1

        # for memory-augmented feature sampling
        bbox_feats, bbox_labels, bbox_targets = self._get_feat_for_memory(x, gt_bboxes, gt_labels, img_metas)
        self.mfs.enqueue_dequeue(bbox_feats, bbox_labels, bbox_targets)
        weight_norm = self.bbox_head.fc_cls.weight.detach().cpu().clone()
        weight_norm = torch.norm(weight_norm, p=2, dim=-1)
        # print(weight_norm)
        selectd_bbox_feat, selectd_labels, selectd_reg_targets, selectd_cls_weight, selectd_reg_weight = \
                        self.mfs.probabilistic_sampler(weight_norm)


        """Run forward function and calculate loss for box head in training."""
        # prediction
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_feats = torch.cat([bbox_feats, selectd_bbox_feat])
        bbox_results = self._bbox_forward_from_feat(bbox_feats)

        # target
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        bbox_targets_all = []
        bbox_targets_all.append(torch.cat([bbox_targets[0], selectd_labels]))
        bbox_targets_all.append(torch.cat([bbox_targets[1], selectd_cls_weight]))
        bbox_targets_all.append(torch.cat([bbox_targets[2], selectd_reg_targets]))
        bbox_targets_all.append(torch.cat([bbox_targets[3], selectd_reg_weight]))

        # update mean score
        self._update_mean_score(bbox_results['cls_score'], sampling_results, gt_labels, selectd_labels, rank, alpha=self.alpha, bg_score=self.bg_score)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets_all, mean_score=self.mean_score)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]