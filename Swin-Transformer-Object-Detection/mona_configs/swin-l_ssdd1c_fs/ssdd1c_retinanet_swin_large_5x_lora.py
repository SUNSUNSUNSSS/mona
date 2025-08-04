_base_ = ["../_base_/datasets/ssdd1c_fs.py", "../_base_/default_runtime.py"]

# 预训练模型路径
load_from = "/data1/sunzr/project/mona_objdet/work_dirs/hrrsd1c_retinanet_swin_large_3x_mona/epoch_10.pth"

# 模型配置
model = dict(
    type="RetinaNet",
    pretrained=None,
    backbone=dict(
        type="SwinTransformer_lora",
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        # out_indices=(0, 1, 2, 3),
        out_indices=(1, 2, 3),
        use_checkpoint=False,
        frozen_stages=1,
    ),
    neck=dict(
        type="FPN",
        in_channels=[384, 768, 1536],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_input",
        num_outs=5,
    ),
    bbox_head=dict(
        type="RetinaHead",
        num_classes=1,  # 修改为1个类别
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    # 训练和测试设置
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=100,
    ),
)

# 优化器配置
optimizer = dict(


    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

# 学习率调度
lr_config = dict(policy="step", step=[6,11,16,18])
runner = dict(type="EpochBasedRunnerAmp", max_epochs=20)

# do not use mmdet version fp16


optimizer_config = dict(grad_clip=None)
# FP16训练配置（可选）
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )