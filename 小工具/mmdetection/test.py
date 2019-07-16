from mmdet.apis import init_detector, inference_detector, show_result

config_file = '/home/wl/mmdetection/haier_faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '/home/wl/mmdetection/work_dirs/haier/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test2.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)
'''
# test a list of images and write the results to image files
imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs)):
    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
'''
