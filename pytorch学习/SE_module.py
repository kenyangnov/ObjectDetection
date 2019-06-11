class Conv2dSamePadding(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
		super().__init__(in_channels, out_channels, kernel_size, stride, 0 , dilation, groups, bias)
		self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2
	
	def forward(self, x):
		ih, iw = x.size()[-2:]
		kh, kw = self.weight.size()[-2:]
		sh, sw = self.stride
		oh, ow = math.ceil(ih/sh), math.ceil(iw/sw)
		pad_h = max((oh-1) * self.stride[0] + (kh-1)*self.dilation[0]+1 - ih, 0)
		pad_w = max((ow-1) * self.stride[1] + (kw-1)*self.dilation[1]+1 - iw, 0)
		if pad_h>0 or pad_w<0:
			x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
		return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

		
class MBConvBlock(nn.Module):
	def __init__(self, block_args, global_params):
		if self.has_se:
			num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
			self._se_reduce = Conv2dSamePadding(in_channels = oup, out_channels = num_squeezed_channels, kernel_size = 1)
			self._se_expand = Conv2dSamePadding(in_channels = num_squeezed_channels, out_channels = oup, kernel_size = 1)

	def forward(self, inputs, drop_connect_rate = None):
		if self.has_se:
			x_squeezed = F.adaptive_avg_pool2d(x, 1)
			x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed))) 
			x = torch.sigmoid(x_squeezed) * x
			
	