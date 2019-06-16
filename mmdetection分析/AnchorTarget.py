import torch

def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bbox_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bbox_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    #todo


def images_to_levels(target, num_level_anchors):
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets

def anchor_target_single(flat_anchors,
                         valid_flag,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flag, img_meta['img_shape'][:2], cfg.allowed_border)
    
    if not inside_flags.any():
        return (None, ) * 6
    
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_ind
     




    def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
        img_h, img_w = img_shape[:2]
        if allowed_border >= 0:
            inside_flags = valid_flag & \
                (flat_anchors[:, 0] >= -allowed_border) & \
                (flat_anchors[:, 1] >= -allowed_border) & \
                (flat_anchors[:, 2] < img_w + allowed_border) & \
                (flat_anchors[:, 3] < img_h + allowed_border)
        else:
            inside_flags = valid_flags
        return inside_flags

    def unmap(data, count, inds, fill=0):
        if data.dim()==1:
            ret = data.new_full((count, ), fill)
            ret[inds] = data
        else:
            new_size = (count, ) + data.size()[1:]
            ret = data.new_full(new_size, fill)
            ret[inds, :] = data
        return ret
