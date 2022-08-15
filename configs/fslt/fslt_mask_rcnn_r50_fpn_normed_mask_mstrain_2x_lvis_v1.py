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
            feature_fushion_available=1,   # 0 不适用特征修正 1使用响应图特征修正 2使用深度互卷积修正
            mfs=dict(
                queue_size=80,
                gpu_statictics=False,
                sampled_num_classes=8,
                sampled_num_features=4
            )
        )
    )
)

evaluation = dict(interval=6, metric=['bbox', 'segm'])

# learning policy
lr_config = dict(
    policy='step',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=0.001,
    step=[3, 5])
total_epochs = 6

# custon hooks: InitializerHook is defined in mmdet/core/utils/initializer_hook.py
custom_hooks = [
    dict(type="InitializerHook")
]

#load_from = './model_checkpoint/epoch_24.pth'
work_dir = 'debug/time_1'
# Train which part, 0 for all, 1 for fc_cls, fc_reg, rpn and mask_head
selectp = 1