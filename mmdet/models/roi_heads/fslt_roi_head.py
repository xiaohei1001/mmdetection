import torch
import numpy as np
import torch.nn.functional as F
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from torch import nn
import torch.distributed as dist
from collections import defaultdict

from mmdet.models.utils import  FsltMFS, CrossAttentionGenerator


@HEADS.register_module()
class FsltRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
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
            self.mfs = FsltMFS(num_classes=self.bbox_head.num_classes,
                               queue_size=self.train_cfg.mfs.queue_size,
                               sampled_num_classes=self.train_cfg.mfs.sampled_num_classes,
                               sampled_num_features=self.train_cfg.mfs.sampled_num_features,
                               gpu_statictics=self.train_cfg.mfs.gpu_statictics,
                               da_num=3)
            # 加载prototypes
            # self.class_prototypes = np.zeros((self.bbox_head.num_classes, 256, 7, 7), dtype=np.float32)
            self.class_prototypes = np.load(
                '/remote-home/share/ylzhang/fslt_complete/LOCE-master/mean_prototypes/loce_mask_r_50_train_gt_features_random_mean.npy').astype(
                np.float32)
            # 计算相似度矩阵
            self.similarity_matrix = self.compute_similarity_matrix()
            self.feature_generator = CrossAttentionGenerator(map_width=7,
                                                             embed_dim=256,
                                                             proj_dim=256,
                                                             num_heads=4)
            self.available_fushion=self.train_cfg.feature_fushion_available
            # self.feature_generator = CrossAttentionGenerator(map_width=self.train_cfg.feature_generator.map_width,
            #                                                   embed_dim=self.train_cfg.feature_generator.embed_dim,
            #                                                   proj_dim=self.train_cfg.feature_generator.proj_dim,
            #                                                   num_heads=self.train_cfg.feature_generator.num_heads)

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
            if self.train_cfg:
                self.feature_generator.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    # 计算相似度矩阵
    def compute_similarity_matrix(self):
        class_prototypes = torch.from_numpy(self.class_prototypes).cuda()
        # num_classes * 256 * 1 * 1
        class_prototypes = F.adaptive_avg_pool2d(class_prototypes, (1, 1))
        # num_classes * 256
        class_prototypes = class_prototypes.reshape(self.bbox_head.num_classes, -1)
        class_prototypes_norm = torch.norm(class_prototypes, p=2, dim=1).unsqueeze(1).expand_as(
            class_prototypes)
        class_prototypes_normalized = class_prototypes.div(class_prototypes_norm + 0.00001)
        similarity_matrix = torch.matmul(class_prototypes_normalized,
                                         class_prototypes_normalized.transpose(1, 0))
        return similarity_matrix

    def feature_fusion_V1(self,box_features, sampling_results):
        """
        feature_fushion_V1 是使用256*1*1原型对RoI Feature进行卷积得到1*7*7的相应图（对所有fg都处理 不单针对rare类）。
        使用相应图与原feature相乘然后进行全剧平均池化得到RoI Feature中的256*1*1代表性向量。（这里全局平均池化相当于是对相应图求平均，这样是否合理？）
        然后与原来feature相加。

        V2的话 是否使用256*1*1原型与RoI Feature进行深度互相关而非简单卷积，直接得到256*7*7的处理后feature。
        """
        fushion_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        fg_idx = torch.zeros((0))
        fg_label = torch.zeros((0))
        for i, sampling_result in enumerate(sampling_results):
            fg_idx = torch.cat([fg_idx, torch.tensor(range(i * 512, (i * 512) + len(sampling_result.pos_inds)))])
            fg_label = torch.cat([fg_label, sampling_result.pos_gt_labels.cpu()])
        prototype_fushion = torch.from_numpy(self.class_prototypes).cuda()
        prototype_fushion = F.adaptive_avg_pool2d(prototype_fushion, (1, 1))

        for i, fg_index_i in enumerate(fg_idx):
            fg_proposal_feature = box_features[int(fg_index_i)].detach()  # 这个proposal的原始feature   这里的feature要256*7*7的
            fg_proposal_label = fg_label[i]
            # TODO 原型产生的相关步骤
            fushion_conv.weight.data = prototype_fushion[int(fg_proposal_label)].unsqueeze(0)  # 以256*1*1的原型作为卷积核
            fg_response_map = fushion_conv(fg_proposal_feature.unsqueeze(0)).squeeze(0)
            fg_response_map_4=(fg_response_map**4)/((fg_response_map**4).sum())
            fg_response_map_4*=4  # 对相应图系数进行放大，扩大前景的影响。
            fg_proposal_feature_fliter = torch.mul(fg_proposal_feature,fg_response_map_4)
            #fg_proposal_feature_pooling = torch.nn.functional.adaptive_avg_pool2d(fg_proposal_feature, (1, 1))
            fg_proposal_feature += fg_proposal_feature_fliter
            box_features[int(fg_index_i)] = fg_proposal_feature

        return  box_features

    def feature_fusion_V2(self,box_features, sampling_results):
        """
        feature_fushion_V1 是使用256*1*1原型对RoI Feature进行卷积得到1*7*7的相应图（对所有fg都处理 不单针对rare类）。
        使用相应图与原feature相乘然后进行全剧平均池化得到RoI Feature中的256*1*1代表性向量。（这里全局平均池化相当于是对相应图求平均，这样是否合理？）
        然后与原来feature相加。

        V2的话 是否使用256*1*1原型与RoI Feature进行深度互相关而非简单卷积，直接得到256*7*7的处理后feature。
        """
        #fushion_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        fg_idx = torch.zeros((0))
        fg_label = torch.zeros((0))
        for i, sampling_result in enumerate(sampling_results):
            fg_idx = torch.cat([fg_idx, torch.tensor(range(i * 512, (i * 512) + len(sampling_result.pos_inds)))])
            fg_label = torch.cat([fg_label, sampling_result.pos_gt_labels.cpu()])
        prototype_fushion = torch.from_numpy(self.class_prototypes).cuda()
        prototype_fushion = F.adaptive_avg_pool2d(prototype_fushion, (1, 1))

        for i, fg_index_i in enumerate(fg_idx):
            fg_proposal_feature = box_features[int(fg_index_i)].detach()  # 这个proposal的原始feature   这里的feature要256*7*7的
            fg_proposal_label = fg_label[i]
            cor_feature=F.conv2d(fg_proposal_feature.unsqueeze(0), prototype_fushion[int(fg_proposal_label)].unsqueeze(0).permute(1,0,2,3), groups=256)
            box_features[int(fg_index_i)] = cor_feature.squeeze(0)

        return  box_features

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
            outs = outs + (mask_results['mask_pred'],)
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
        rank = dist.get_rank()
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
        bbox_feats = bbox_feats.detach()  # 0 TODO 这里？梯度

        bbox_labels, _, bbox_targets, _ = self.bbox_head.get_targets_for_memory(proposal_list, neg_proposal_list,
                                                                                gt_bbox_list, gt_label_list,
                                                                                self.train_cfg)
        # bbox_targets是映射到了偏移么？
        return bbox_feats, bbox_labels, bbox_targets  # 这个函数内哪里用到memory了？

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):

        try:
            rank = dist.get_rank()
        except:
            rank = -1

        # for memory-augmented feature sampling 更新memory bank
        bbox_feats, bbox_labels, bbox_targets = self._get_feat_for_memory(x, gt_bboxes, gt_labels, img_metas)
        self.mfs.enqueue_dequeue(bbox_feats, bbox_labels, bbox_targets)
        weight_norm = self.bbox_head.fc_cls.weight.detach().cpu().clone()  # TODO 这里的梯度？
        weight_norm = torch.norm(weight_norm, p=2, dim=-1)  # 这里为什么用L2范数，有什么依据或者更好的选择么？
        # 筛选出来的原始类别
        selected_classes = self.mfs.sampling_classes(self.mfs.sampled_num_classes, weight_norm)
        # 筛选出来样本
        selectd_bbox_feat, selectd_labels, selectd_reg_targets, selectd_cls_weight, selectd_reg_weight = \
            self.mfs.mfs_sampler(selected_classes, self.mfs.sampled_num_features)

        # 不需要增强的样本的编号
        nake_idxs = []
        # 需要增强的样本的编号
        da_idxs = []
        # 选的什么类别进行的增强
        selected_da_classes = []
        # 有被筛选出来的样本
        if len(selectd_labels) != 0:
            # 所有的selected_labels都要参加筛选，确定是否进行da
            da_classes = self.mfs.sampling_da_classes(selectd_labels, self.similarity_matrix)
            for i in range(len(da_classes)):
                if da_classes[i] == -1:
                    nake_idxs.append(i)
                else:
                    da_idxs.append(i)
                    selected_da_classes.append(da_classes[i])
        nake_idxs = torch.LongTensor(nake_idxs).cuda()
        da_idxs = torch.LongTensor(da_idxs).cuda()
        # 需要筛选增强样本的类别
        selected_da_classes = torch.LongTensor(selected_da_classes).cuda()

        nake_selected_bbox_feat, da_selected_bbox_feat = selectd_bbox_feat[nake_idxs], selectd_bbox_feat[da_idxs]
        nake_selected_labels, da_selected_labels = selectd_labels[nake_idxs], selectd_labels[da_idxs]
        nake_selected_reg_targets, da_selected_reg_targets = selectd_reg_targets[nake_idxs], selectd_reg_targets[
            da_idxs]
        nake_selected_cls_weight, da_selected_cls_weight = selectd_cls_weight[nake_idxs], selectd_cls_weight[da_idxs]
        nake_selected_reg_weight, da_selected_reg_weight = selectd_reg_weight[nake_idxs], selectd_reg_weight[da_idxs]

        # 按照增强类别筛选出来用来da的特征
        selectd_bbox_feat_da, _, _, _, _ = self.mfs.mfs_sampler(selected_da_classes, 1)
        if rank == 0:
            count = 0
            for c in selected_da_classes:
                if c not in selected_classes:
                    count += 1
            print(count == len(selected_da_classes))
        # 筛选出来的prototypes，这个prototypes是要增强的类别对应的prototypes
        selected_prototypes = torch.randn((0, 256, 7, 7)).cuda()
        # 要增强的类别的prototypes
        for c in da_selected_labels:
            if c not in selected_classes:
                print('no')
            prototype_c = torch.from_numpy(self.class_prototypes[c]).clone().unsqueeze(0).cuda()
            selected_prototypes = torch.cat([selected_prototypes, prototype_c], dim=0)
        # 增强
        new_da_selected_bbox_feat = self.feature_generator(da_selected_bbox_feat, selectd_bbox_feat_da,
                                                           selected_prototypes)
        """Run forward function and calculate loss for box head in training."""
        # prediction
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        # feats合体， 原有的加不增强的加增强的
        if self.available_fushion==1:
            bbox_feats = self.feature_fusion_V1(bbox_feats, sampling_results)
        elif self.available_fushion==2:
            bbox_feats = self.feature_fusion_V2(bbox_feats, sampling_results)
        bbox_feats = torch.cat([bbox_feats, nake_selected_bbox_feat, new_da_selected_bbox_feat])
        bbox_results = self._bbox_forward_from_feat(bbox_feats)

        # target
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        bbox_targets_all = []
        # targets合体， 原有的加不增强的加增强的
        bbox_targets_all.append(torch.cat([bbox_targets[0], nake_selected_labels, da_selected_labels]))
        bbox_targets_all.append(torch.cat([bbox_targets[1], nake_selected_cls_weight, da_selected_cls_weight]))
        bbox_targets_all.append(torch.cat([bbox_targets[2], nake_selected_reg_targets, da_selected_reg_targets]))
        bbox_targets_all.append(torch.cat([bbox_targets[3], nake_selected_reg_weight, da_selected_reg_weight]))

        # anchors = torch.flatten(F.adaptive_avg_pool2d(da_selected_bbox_feat, (1, 1)), start_dim=1)
        # positives = torch.flatten(F.adaptive_avg_pool2d(new_da_selected_bbox_feat, (1, 1)), start_dim=1)
        # negatives = torch.flatten(F.adaptive_avg_pool2d(selectd_bbox_feat_da, (1, 1)), start_dim=1)
        # if rank == 0:
        #     print(anchors.size())
        # # 新的loss搞上
        # loss_bbox = self.bbox_head.loss(anchors, positives, negatives,
        #                                 da_selected_labels, selected_da_classes, bbox_results['cls_score'],
        #                                 bbox_results['bbox_pred'], rois,
        #                                 *bbox_targets_all)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets_all)

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


@HEADS.register_module()
class FsltRoIHeadV2(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
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
            self.mfs = FsltMFS(num_classes=self.bbox_head.num_classes,
                               queue_size=self.train_cfg.mfs.queue_size,
                               sampled_num_classes=self.train_cfg.mfs.sampled_num_classes,
                               sampled_num_features=self.train_cfg.mfs.sampled_num_features,
                               gpu_statictics=self.train_cfg.mfs.gpu_statictics,
                               da_num=3)
            # 加载prototypes
            # self.class_prototypes = np.zeros((self.bbox_head.num_classes, 256, 7, 7), dtype=np.float32)
            self.class_prototypes = np.load(
                '/remote-home/share/ylzhang/fslt_complete/LOCE-master/mean_prototypes/loce_mask_r_50_train_gt_features_random_mean.npy').astype(
                np.float32)
            # 计算相似度矩阵
            self.similarity_matrix = self.compute_similarity_matrix()
            self.feature_generator = CrossAttentionGenerator(map_width=7,
                                                             embed_dim=256,
                                                             proj_dim=256,
                                                             num_heads=4)
            # self.feature_generator = CrossAttentionGenerator(map_width=self.train_cfg.feature_generator.map_width,
            #                                                   embed_dim=self.train_cfg.feature_generator.embed_dim,
            #                                                   proj_dim=self.train_cfg.feature_generator.proj_dim,
            #                                                   num_heads=self.train_cfg.feature_generator.num_heads)

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
            if self.train_cfg:
                self.feature_generator.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    # 计算相似度矩阵，自己跟自己是0的哦
    def compute_similarity_matrix(self):
        class_prototypes = torch.from_numpy(self.class_prototypes).cuda()
        # num_classes * 256 * 1 * 1
        class_prototypes = F.adaptive_avg_pool2d(class_prototypes, (1, 1))
        # num_classes * 256
        class_prototypes = class_prototypes.reshape(self.bbox_head.num_classes, -1)
        class_prototypes_norm = torch.norm(class_prototypes, p=2, dim=1).unsqueeze(1).expand_as(
            class_prototypes)
        class_prototypes_normalized = class_prototypes.div(class_prototypes_norm + 0.00001)
        similarity_matrix = torch.matmul(class_prototypes_normalized,
                                         class_prototypes_normalized.transpose(1, 0))
        similarity_matrix_diag = torch.diag(similarity_matrix)
        similarity_matrix_diag = torch.diag_embed(similarity_matrix_diag)
        similarity_matrix = similarity_matrix - similarity_matrix_diag
        return similarity_matrix

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
            outs = outs + (mask_results['mask_pred'],)
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

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        try:
            rank = dist.get_rank()
        except:
            rank = -1

        # for memory-augmented feature sampling 更新memory bank
        bbox_feats, bbox_labels, bbox_targets = self._get_feat_for_memory(x, gt_bboxes, gt_labels, img_metas)
        self.mfs.enqueue_dequeue(bbox_feats, bbox_labels, bbox_targets)
        weight_norm = self.bbox_head.fc_cls.weight.detach().cpu().clone()
        weight_norm = torch.norm(weight_norm, p=2, dim=-1)
        # 筛选出来的原始类别
        selected_classes = self.mfs.sampling_classes(self.mfs.sampled_num_classes, weight_norm)
        # 筛选出来样本,目前原始的想设置为6, 原始的为4, 增强的为2
        selectd_bbox_feat, selectd_labels, selectd_reg_targets, selectd_cls_weight, selectd_reg_weight, da_idxs, nake_idxs = \
            self.mfs.mfs_sampler_with_da_frac(selected_classes, self.mfs.sampled_num_features, 0.4)
        if rank == 0:
            print("da_idxs:")
            print(da_idxs)
            print(len(da_idxs))
            print("nake_idxs:")
            print(nake_idxs)
        selected_da_classes = []
        # 有被筛选出来的样本
        if len(selectd_labels) != 0:
            # 依据筛选出来的样本筛选增强样本类编号
            da_classes = self.mfs.sampling_da_classes_with_da_frac(selectd_labels[torch.LongTensor(da_idxs)], self.similarity_matrix)
            # if rank == 0:
            #     print("da_classes:")
            #     print(da_classes)
            #     print(len(da_classes))
            selected_da_classes.extend(da_classes)

        nake_idxs = torch.LongTensor(nake_idxs).cuda()
        da_idxs = torch.LongTensor(da_idxs).cuda()
        selected_da_classes = torch.LongTensor(selected_da_classes).cuda()

        nake_selected_bbox_feat, da_selected_bbox_feat = selectd_bbox_feat[nake_idxs], selectd_bbox_feat[da_idxs]
        nake_selected_labels, da_selected_labels = selectd_labels[nake_idxs], selectd_labels[da_idxs]
        nake_selected_reg_targets, da_selected_reg_targets = selectd_reg_targets[nake_idxs], selectd_reg_targets[
            da_idxs]
        nake_selected_cls_weight, da_selected_cls_weight = selectd_cls_weight[nake_idxs], selectd_cls_weight[da_idxs]
        nake_selected_reg_weight, da_selected_reg_weight = selectd_reg_weight[nake_idxs], selectd_reg_weight[da_idxs]

        selected_bbox_feat_da = self.mfs.mfs_sampler_da_feats(selected_da_classes, self.class_prototypes)
        # if rank == 0:
        #     print("selected_bbox_feat_da:")
        #     print(selected_bbox_feat_da.size())
        #
        # selected_da_prototypes = torch.randn((0, 256, 7, 7)).cuda()
        # for c in selected_da_classes:
        #     prototype_c = torch.from_numpy(self.class_prototypes[c]).clone().unsqueeze(0).cuda()
        #     selected_da_prototypes = torch.cat([selected_da_prototypes, prototype_c], dim=0)

        selected_prototypes = torch.randn((0, 256, 7, 7)).cuda()
        for c in da_selected_labels:
            prototype_c = torch.from_numpy(self.class_prototypes[c]).clone().unsqueeze(0).cuda()
            selected_prototypes = torch.cat([selected_prototypes, prototype_c], dim=0)
        # 增强
        new_da_selected_bbox_feat = self.feature_generator(da_selected_bbox_feat, selected_bbox_feat_da,
                                                           selected_prototypes)
        """Run forward function and calculate loss for box head in training."""
        # prediction
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        # feats合体， 原有的加不增强的加增强的
        bbox_feats = torch.cat([bbox_feats, nake_selected_bbox_feat, new_da_selected_bbox_feat])
        bbox_results = self._bbox_forward_from_feat(bbox_feats)

        # target
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        bbox_targets_all = []
        # targets合体， 原有的加不增强的加增强的
        bbox_targets_all.append(torch.cat([bbox_targets[0], nake_selected_labels, da_selected_labels]))
        bbox_targets_all.append(torch.cat([bbox_targets[1], nake_selected_cls_weight, da_selected_cls_weight]))
        bbox_targets_all.append(torch.cat([bbox_targets[2], nake_selected_reg_targets, da_selected_reg_targets]))
        bbox_targets_all.append(torch.cat([bbox_targets[3], nake_selected_reg_weight, da_selected_reg_weight]))

        # anchors = torch.flatten(F.adaptive_avg_pool2d(da_selected_bbox_feat, (1, 1)), start_dim=1)
        # positives = torch.flatten(F.adaptive_avg_pool2d(new_da_selected_bbox_feat, (1, 1)), start_dim=1)
        # negatives = torch.flatten(F.adaptive_avg_pool2d(selectd_bbox_feat_da, (1, 1)), start_dim=1)
        # if rank == 0:
        #     print(anchors.size())
        # # 新的loss搞上
        # loss_bbox = self.bbox_head.loss(anchors, positives, negatives,
        #                                 da_selected_labels, selected_da_classes, bbox_results['cls_score'],
        #                                 bbox_results['bbox_pred'], rois,
        #                                 *bbox_targets_all)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets_all)

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
