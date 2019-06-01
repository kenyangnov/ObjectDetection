# 1.整体结构
config文件为模型设置文件，以dict形式构建，主要部分：模型结构设定、模型训练/验证/测试参数设定、数据集设定、优化器设定、学习率策略等。

two-stage和one-stage的参数构成略有不同（主要体现在train_cfg/test_cfg），下面分别列出:

# 2.two-stage
## 2.1 model主要参数
* input_size
* model
	+ type
	+ pretrained
	+ backbone
	+ neck
	+ bbox_head

## 2.2 (train/test)_cfg主要参数
* train_cfg
	+ assigner
	+ smoothl1_beta
	+ allowed_border
	+ pos_weight
	+ neg_pos_ratio
	+ debug
* test_cfg
	+ nms
	+ min_bbox_size
	+ score_thr
	+ max_per_img

## 2.3 dataset主要参数
* dataset_type
* data_root
* img_norm_sfg
* data_root
	+ imgs_per_gpu
	+ workers_per_gpu
	+ train
	+ val
	+ test
## 2.4 optimizer主要参数
* optimizer
* optimizer_config

## 2.5 learningrate 策略
* lr_config
* checkpoint_config

## 2.6 模型运行参数设置
* total_epochs
* dist_params
* log_level
* work_dir
* load_from
* resume_from
* workflow
	
# 3.one-stage
## 3.1 model主要参数
* inputsize
* pretrained
* backbone
* neck
* rpn_head
* bbox_roi_extractor
* bbox_head

## 3.2 (train/test)_cfg主要参数
* train_cfg
	+ rpn
	+ rpn_proposal
	+ rcnn
* test_cfg
	+ rpn
	+ rcnn

## 3.3 dataset主要参数
* dataset_type
* data_root
* img_norm_sfg
* data_root
	+ imgs_per_gpu
	+ workers_per_gpu
	+ train
	+ val
	+ test
## 3.4 optimizer主要参数
* optimizer
* optimizer_config

## 3.5 learningrate 策略
* lr_config
* checkpoint_config

## 3.6 模型运行参数设置
* total_epochs
* dist_params
* log_level
* work_dir
* load_from
* resume_from
* workflow