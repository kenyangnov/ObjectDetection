import numpy as np

def get_IoU(pred_bbox, gt_bbox):
    """
    return iou score between pred / gt bboxes
    :param pred_bbox: predict bbox coordinate
    :param gt_bbox: ground truth bbox coordinate
    :return: iou score
    """

    #获取坐标
    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maxinum(ixmax - ixmin + 1., 0.)	#因为是像素点的坐标，所以要加上1
    ih = np.maximum(iymax - iymin + 1., 0.)
    #求交集
    inter = iw * ih
    #求并集
    uni = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1) + \
          (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1) - inter
    #求交并比
    overlap = inter / uni
    return overlap

def get_max_IoU(pred_bbox, gt_bbox):
    """
    given 1 gt bbox, >1 pred bboxes, return max iou score for the given gt bbox and pred_bboxes
    :param pred_bbox: predict bboxes coordinates, we need to find the max iou score with gt bbox for these pred bboxes
    :param gt_bbox: ground truth bbox coordinate
    :return: max iou score
    """
    if pred_bbox.shape[0] > 0:
        ixmin = np.maximum(pred_bbox[:, 0], gt_bbox[0])
        iymin = np.maximum(pred_bbox[:, 1], gt_bbox[1])
        ixmax = np.minimum(pred_bbox[:, 2], gt_bbox[2])
        iymax = np.minimum(pred_bbox[:, 3], gt_bbox[3])
		iw = np.maxinum(ixmax - ixmin + 1., 0.)	#因为是像素点的坐标，所以要加上1
		ih = np.maximum(iymax - iymin + 1., 0.)
        #求交集
        inters = iw * ih
        #求并集
        uni = (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) +
              (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.) -
              inters
        #求交并比
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        index_max = np.argmax(overlaps)
        return overlaps, ovmax, index_max
