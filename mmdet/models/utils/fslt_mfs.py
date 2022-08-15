import math

import torch
import numpy as np
import torch.distributed as dist


class FsltMFS:
    def __init__(self, num_classes, queue_size=80, sampled_num_classes=8, sampled_num_features=4, gpu_statictics=False,
                 da_num=3):
        # for bbox generator
        self.bbox_offset_matrix = self._get_bbox_offset_matrix().cuda()

        # for feature memory module
        self.gpu_statictics = gpu_statictics
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.queue_bbox_feats = torch.zeros(self.num_classes, self.queue_size, 256, 7, 7)
        self.queue_gt_labels = - torch.ones(self.num_classes, self.queue_size).long().cuda()
        self.queue_reg_targets = torch.zeros(self.num_classes, self.queue_size, 4).cuda()
        self.queue_ptrs = [0 for _ in range(self.num_classes)]
        if self.gpu_statictics:
            self.queue_bbox_feats = self.queue_bbox_feats.cuda()

        # for probabilistic sampler
        self.sampled_num_classes = sampled_num_classes
        self.sampled_num_features = sampled_num_features
        self.da_num = da_num

    def _get_bbox_offset_matrix(self):
        bbox_offset_matrix = []
        for k in [8, 4, 2, 1]:
            bbox_offset_matrix.append(np.concatenate(
                [[-1.0 for j in range(k)] + [1.0 for j in range(k)] for i in range(16 // (2 * k))]))
        return torch.Tensor(bbox_offset_matrix)

    def bbox_generator(self, img_labels, img_bboxes, img_metas):
        w = img_bboxes[:, 2] - img_bboxes[:, 0]
        h = img_bboxes[:, 3] - img_bboxes[:, 1]

        random_offset_matrix = torch.rand(self.bbox_offset_matrix.shape).type_as(img_bboxes)
        random_bbox_offset = 1. / 6 * self.bbox_offset_matrix * random_offset_matrix * torch.stack([w, h, w, h],
                                                                                                   -1).unsqueeze(-1)

        proposals = img_bboxes.unsqueeze(-1) + random_bbox_offset
        proposals = proposals.permute([0, 2, 1])

        img_shape = img_metas['img_shape']
        x1, y1, x2, y2 = proposals.split([1, 1, 1, 1], dim=-1)
        x1 = x1.clamp(min=0, max=img_shape[1] - 1)
        y1 = y1.clamp(min=0, max=img_shape[0] - 1)
        x2 = x2.clamp(min=0, max=img_shape[1] - 1)
        y2 = y2.clamp(min=0, max=img_shape[0] - 1)
        proposals = torch.cat([x1, y1, x2, y2], dim=-1)

        proposal_list = []
        gt_bbox_list = []
        gt_label_list = []
        for label_ind, (label, bbox) in enumerate(zip(img_labels, img_bboxes)):
            # select top sample_number
            proposal = proposals[label_ind]
            gt_bbox = bbox.unsqueeze(0).repeat([len(proposal), 1])
            gt_label = label.new_ones(len(proposal)) * label

            proposal_list.append(proposal)
            gt_bbox_list.append(gt_bbox)
            gt_label_list.append(gt_label)

        img_proposal = torch.cat(proposal_list)
        img_gt_bbox = torch.cat(gt_bbox_list)
        img_gt_label = torch.cat(gt_label_list)

        return img_proposal, img_gt_bbox, img_gt_label

    def enqueue_dequeue(self, bbox_feats, gt_labels, reg_targets):
        # save feat, label and target to queue
        for label in set(list(gt_labels.cpu().numpy())):
            if label == -1:
                continue
            bbox_feat = bbox_feats[gt_labels == label][:self.queue_size]
            gt_label = gt_labels[gt_labels == label][:self.queue_size]
            reg_target = reg_targets[gt_labels == label][:self.queue_size]

            if not self.gpu_statictics:
                bbox_feat = bbox_feat.cpu()

            queue_ptr = self.queue_ptrs[label]
            feat_size = len(bbox_feat)
            feat_ptr = 0
            if queue_ptr + feat_size > self.queue_size:
                feat_ptr = self.queue_size - queue_ptr
                self.queue_bbox_feats[label, queue_ptr:] = bbox_feat[:feat_ptr]
                self.queue_gt_labels[label, queue_ptr:] = gt_label[:feat_ptr]
                self.queue_reg_targets[label, queue_ptr:] = reg_target[:feat_ptr]
                feat_size = feat_size - feat_ptr
                queue_ptr = 0
            self.queue_bbox_feats[label, queue_ptr:queue_ptr + feat_size] = bbox_feat[feat_ptr:]
            self.queue_gt_labels[label, queue_ptr:queue_ptr + feat_size] = gt_label[feat_ptr:]
            self.queue_reg_targets[label, queue_ptr:queue_ptr + feat_size] = reg_target[feat_ptr:]
            self.queue_ptrs[label] = queue_ptr + feat_size

    # select according to prob
    def sampling_classes(self, sampled_num_classes, weight_norm):
        weight_norm_fg = weight_norm[:self.num_classes]
        weight_norm_revert = 1. / (weight_norm_fg ** 4 + 10e-8)     # 这里怎么改进，组会上说这里可能对尾部过拟合，导致尾部类得分反而偏高。
        sample_probability = weight_norm_revert / weight_norm_revert.sum()
        randnum = np.sort(np.random.rand(sampled_num_classes))
        prob_point = 0
        randnum_index = 0
        select_classes = []
        for index, prob in enumerate(sample_probability):   # 这种概率随机抽样效果好么，要不要指定一下
            prob_point += prob
            if randnum[randnum_index] <= prob_point:
                while randnum[randnum_index] <= prob_point:
                    select_classes.append(index)
                    randnum_index += 1
                    if randnum_index >= len(randnum):
                        return torch.LongTensor(select_classes).cuda()

        # for sum(sample_probability) != 1 in case
        return select_classes

    def sampling_da_classes(self, selected_classes, similarity_matrix):
        # 选择与当前类别最相似的da_num个类别（包含自己）
        _, class_indices = similarity_matrix[selected_classes].topk(self.da_num, dim=1)
        da_classes = []
        class_indices = class_indices.cpu().numpy()
        N, K = class_indices.shape
        for n in range(N):
            class_candidates = class_indices[n]
            # 从da_num中随机筛选出来一个
            da_class = np.random.choice(class_candidates, 1, replace=False)[0]
            # 筛选到的样本在selected_classes中,或者是当前要da的类别里没有样本
            if (da_class in selected_classes) or (((self.queue_gt_labels[da_class] == da_class).sum()) == 0):
                da_classes.append(-1)
            else:
                da_classes.append(da_class)
        return da_classes

    # 依据筛选的class筛选样本
    def mfs_sampler(self, selected_classes, sampled_num_features):
        selected_labels_list = []
        selected_reg_targets_list = []
        selected_bbox_feat_list = []
        for label in selected_classes:
            # if queue for current class is empty
            length = (self.queue_gt_labels[label] == label).sum()
            if length == 0:
                continue

            random_inds = torch.randint(0, length, (sampled_num_features, 1))[:, 0]

            sampled_feats = self.queue_bbox_feats[label][random_inds]
            if not self.gpu_statictics:
                sampled_feats = sampled_feats.cuda()
            selected_bbox_feat_list.append(sampled_feats)   # 挑4个该类别的缓存feature出来
            selected_labels_list.append(self.queue_gt_labels[label][random_inds]) # 取对应的4个label，都是一样的
            selected_reg_targets_list.append(self.queue_reg_targets[label][random_inds]) # 取出对应的box回归系数

        # cat all selected feat e.t. from all gt classes
        if len(selected_labels_list) > 0:
            selected_labels = torch.cat(selected_labels_list)
            selected_reg_targets = torch.cat(selected_reg_targets_list)
            selected_bbox_feat = torch.cat(selected_bbox_feat_list).cuda()
        else:
            selected_labels = self.queue_gt_labels.new([])
            selected_reg_targets = self.queue_reg_targets.new([]).reshape(0, 4)
            selected_bbox_feat = self.queue_bbox_feats.new([]).cuda().reshape(0, 256, 7, 7)
            # # selected_labels = torch.randn(0).cuda()
            # # selected_reg_targets = torch.randn((0, 4)).cuda()
            # selected_labels = self.queue_gt_labels.new([])
            # selected_reg_targets = self.queue_reg_targets.new([])
            # selected_bbox_feat = torch.randn((0, 256, 7, 7)).cuda()
        selected_cls_weight = selected_reg_targets.new_ones(selected_labels.shape)
        selected_reg_weight = selected_reg_targets.new_ones(selected_reg_targets.shape)

        return selected_bbox_feat, selected_labels, selected_reg_targets, selected_cls_weight, selected_reg_weight

    # 依据筛选的class筛选样本
    def mfs_sampler_with_da_frac(self, selected_classes, sampled_num_features, da_frac):
        selected_labels_list = []
        selected_reg_targets_list = []
        selected_bbox_feat_list = []
        nake_indices = []
        da_indices = []
        count = 0
        for label in selected_classes:
            # if queue for current class is empty
            length = (self.queue_gt_labels[label] == label).sum()
            if length == 0:
                continue

            random_inds = torch.randint(0, length, (sampled_num_features, 1))[:, 0]

            sampled_feats = self.queue_bbox_feats[label][random_inds]
            origin_idxs = set([i for i in range(sampled_num_features)])
            da_idxs = set([i for i in range(math.floor(sampled_num_features * da_frac))])
            for id in origin_idxs:
                if id in da_idxs:
                    da_indices.append(id + count)
                else:
                    nake_indices.append(id + count)
            count += sampled_num_features
            if not self.gpu_statictics:
                sampled_feats = sampled_feats.cuda()
            selected_bbox_feat_list.append(sampled_feats)
            selected_labels_list.append(self.queue_gt_labels[label][random_inds])
            selected_reg_targets_list.append(self.queue_reg_targets[label][random_inds])

        # cat all selected feat e.t. from all gt classes
        if len(selected_labels_list) > 0:
            selected_labels = torch.cat(selected_labels_list)
            selected_reg_targets = torch.cat(selected_reg_targets_list)
            selected_bbox_feat = torch.cat(selected_bbox_feat_list).cuda()
        else:
            selected_labels = self.queue_gt_labels.new([])
            selected_reg_targets = self.queue_reg_targets.new([]).reshape(0, 4)
            selected_bbox_feat = self.queue_bbox_feats.new([]).cuda().reshape(0, 256, 7, 7)
            # # selected_labels = torch.randn(0).cuda()
            # # selected_reg_targets = torch.randn((0, 4)).cuda()
            # selected_labels = self.queue_gt_labels.new([])
            # selected_reg_targets = self.queue_reg_targets.new([])
            # selected_bbox_feat = torch.randn((0, 256, 7, 7)).cuda()
        selected_cls_weight = selected_reg_targets.new_ones(selected_labels.shape)
        selected_reg_weight = selected_reg_targets.new_ones(selected_reg_targets.shape)
        # da_indices = torch.LongTensor(da_indices).cuda()
        # nake_indices = torch.LongTensor(nake_indices).cuda()
        # rank = dist.get_rank()
        # if rank == 0:
        #     print('rrrrr')
        #     print(count)
        #     print(len(selected_labels))
        return selected_bbox_feat, selected_labels, selected_reg_targets, selected_cls_weight, selected_reg_weight, da_indices, nake_indices

    def sampling_da_classes_with_da_frac(self, selected_classes, similarity_matrix):
        _, class_indices = similarity_matrix[selected_classes].topk(self.da_num, dim=1)
        da_classes = []
        class_indices = class_indices.cpu().numpy()
        N, K = class_indices.shape
        for n in range(N):
            class_candidates = class_indices[n]
            da_class = np.random.choice(class_candidates, 1, replace=False)[0]
            # 当前的筛选样本中没有样本
            da_classes.append(da_class)
        return da_classes

    def mfs_sampler_da_feats(self, selected_classes, class_prototypes):
        # 类原型
        class_prototypes = torch.from_numpy(class_prototypes)
        selected_bbox_feat_list = []
        for label in selected_classes:
            # if queue for current class is empty
            length = (self.queue_gt_labels[label] == label).sum()
            if length == 0:
                sampled_feats = class_prototypes[label].unsqueeze(0)
            else:
                random_inds = torch.randint(0, length, (1, 1))[:, 0]
                sampled_feats = self.queue_bbox_feats[label][random_inds]
            if not self.gpu_statictics:
                sampled_feats = sampled_feats.cuda()
            selected_bbox_feat_list.append(sampled_feats)
        if len(selected_bbox_feat_list) > 0:
            selected_bbox_feat = torch.cat(selected_bbox_feat_list).cuda()
        else:
            selected_bbox_feat = self.queue_bbox_feats.new([]).cuda().reshape(0, 256, 7, 7)
        return selected_bbox_feat
