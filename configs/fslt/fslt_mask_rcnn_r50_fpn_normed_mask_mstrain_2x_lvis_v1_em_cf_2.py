_base_ = './mask_rcnn_r50_fpn_normed_mask_mstrain_2x_lvis_v1.py'

model = dict(
    roi_head=dict(
        type='FsltRoIHead',
        bbox_head=dict(
            type='FsltShared2FCBBoxHead',
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            )
    ),
    train_cfg=dict(
        rcnn=dict(
            alpha=0.9,
            bg_score=0.01,
            mfs=dict(
                queue_size=80,
                gpu_statictics=False,
                sampled_num_classes=8,
                sampled_num_features=4
            )
        )
    )
)

evaluation = dict(interval=3, metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=0.001,
    step=[3, 3]
)
total_epochs = 3

# custon hooks: InitializerHook is defined in mmdet/core/utils/initializer_hook.py
custom_hooks = [
    dict(type="InitializerHook")
]

load_from = './work_dirs/fslt_mask_rcnn_r50_fpn_normed_mask_mstrain_2x_lvis_v1_em_da_2/epoch_2.pth'

# Train which part, 0 for all, 1 for fc_cls, fc_reg, rpn and mask_head
# selectp = 1
selectp = 1
