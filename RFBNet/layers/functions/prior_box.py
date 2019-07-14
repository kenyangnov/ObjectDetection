import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        #config.py中min_dim的值等于输入图片尺寸
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
    
    # 生成所有的PriorBox
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            # 产生特征图对应的坐标(i,j)
            for i, j in product(range(f), repeat=2):
                # 观察config.py文件，看出steps[k]*feature_maps[k]约等于image_size
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 观察config.py文件，看出min_sizes[k+1]项是max_sizes[k]项
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime] # 正方形priorbox

                # rest of aspect ratios (cfg文件中设置的aspect ratios)
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
        # out相当于含有多组不同尺寸特征图的priorbox