## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference:

import os

import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from UDL.Basis.module import PatchMergeModule
from UDL.Basis.criterion_metrics import SetCriterion
from torch import optim
from models.base_model import DerainModel

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x






def ka_window_partition(x, window_size):
    """
    input: (B, C, H, W)
    output: (num_windows, B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(2, 4, 0, 1, 3, 5).contiguous().view(-1, B, C, window_size, window_size)
    return windows


def ka_window_reverse(windows, window_size, H, W):
    """
    input: (num_windows, B, C, window_size, window_size)
    output: (B, C, H, W)
    """
    B = windows.shape[1]
    x = windows.contiguous().view(H // window_size, W // window_size, B, -1, window_size, window_size)
    x = x.permute(2, 3, 0, 4, 1, 5).contiguous().view(B, -1, H, W)
    return x


class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(channel, channel // 16 if channel >= 64 else channel, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(channel // 16 if channel >= 64 else channel, channel, kernel_size=1),
                                nn.Sigmoid(), )

    def forward(self, x):
        channel_weight = self.se(x)
        x = x * channel_weight
        return x
    

class ConvLayer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x, kernels=None):

        x = self.conv(x)

        return x, self.conv.weight.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

class KernelAttention(nn.Module):
    """
    第一个分组卷积产生核，然后计算核的自注意力，调整核，第二个分组卷积产生输出，skip connection
    
    Args:
        dim: 输入通道数
        window_size: 窗口大小
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        attn_drop: 注意力dropout
        proj_drop: 输出dropout
        ka_window_size: kernel attention window size
        kernel_size: 卷积核大小
        kernel_dim_scale: 卷积核通道数缩放因子, 未使用
        stride: 卷积步长
        padding: 卷积padding
    """

    def __init__(self, dim, num_heads, ka_window_size=16, kernel_size=3, kernel_dim_scale=1, stride=1, padding=1, K=2, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(KernelAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.img_size = 64
        self.window_size = ka_window_size
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.kernel_dim_scale = kernel_dim_scale

        self.scale = qk_scale or (dim//num_heads) ** (-0.5)

        self.hidden_dim = int(dim * kernel_dim_scale)
        self.win_num = 16 #(self.img_size//self.window_size)**2

        self.norm = nn.LayerNorm(dim)

        self.num_layers = self.win_num
        self.convlayers = nn.ModuleList()
        self.selayers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ConvLayer(self.dim, self.kernel_size, stride=stride, padding=padding)
            self.convlayers.append(layer)
        for j_layer in range(self.num_layers):
            layer = SELayer(self.dim)
            self.selayers.append(layer)

        self.proj_qkv = nn.Linear(self.dim, self.dim*3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_out = nn.Linear(self.dim, self.dim)

        self.layernorm = nn.LayerNorm(dim)


    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        shortcut = x

        # x_windows:  win_num, bs, c, wh, ww
        x_windows = ka_window_partition(x, self.window_size)
        x_windows_origin = x_windows


        ### 下面对每个窗口进行卷积并获取卷积核
        # TODO: 这里如何写成并行运算的方法
        i = 0
        kernels = []
        
        for convlayer in self.convlayers:
            # kernel: out_c, k_size**2, in_c
            _, kernel = convlayer(x_windows[i], kernels=None)
            kernels.append(kernel)
            i = i + 1
        # kernels:  列表中有win_num个 out_c, k_size**2, in_c 的张量



        ### 下面想要计算所有卷积核间的自注意力
        # kernels:  out_c, win_num*k_size**2, in_c
        kernels = torch.cat(kernels, 1)

        # kernels_qkv:  3, out_c, num_heads, win_num*k_size**2, in_c/num_heads
        kernels_qkv = self.proj_qkv(kernels).reshape(self.dim, self.win_num*self.kernel_size**2, 3, self.num_heads, self.dim//self.num_heads).permute(2, 0, 3, 1, 4)

        # out_c, num_heads, win_num*k_size**2, in_c/num_heads
        kernels_q, kernels_k, kernels_v = kernels_qkv[0], kernels_qkv[1], kernels_qkv[2]
        kernels_q = kernels_q * self.scale

        # attn: out_c, num_heads, win_num*k_size**2, win_num*k_size**2
        attn = (kernels_q @ kernels_k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # kernels:  out_c, win_num*k_size**2, in_c
        kernels = (attn @ kernels_v).transpose(1, 2).reshape(self.dim, self.win_num*self.kernel_size**2, self.dim)

        # TODO check: 此处kernels由win_num*k_size**2拆开，win_num的维度是在k_size前面还是后面
        # kernels:  win_num, out_c, in_c, k_size, k_size
        kernels = self.proj_out(kernels).reshape(self.dim, self.win_num, self.kernel_size, self.kernel_size, self.dim).permute(1, 0, 4, 2, 3)


        ### 下面计算SELayer输出并重新进行卷积
        # kernels:  win_num, out_c, in_c, k_size, k_size
        # x_windows_origin:  win_num, bs, c, wh, ww
        i = 0
        x_windows_out = []

        for selayer in self.selayers:
            # kernel:  out_c, in_c, k_size, k_size
            kernel = selayer(kernels[i])
            # x_window:  bs, c, wh, ww
            x_window = F.conv2d(x_windows_origin[i], weight=kernel, bias=None, stride=self.stride, padding=self.padding).unsqueeze(0)

            # TODO check: 此处由1, bs*c, h, w变为1, bs, c, h, w的操作是否正确
            # x_window:  1, bs, c, wh, ww
            # x_window = x_window.view(B, self.dim, self.window_size, self.window_size).unsqueeze(0)
            x_windows_out.append(x_window)
            i = i + 1

        # x_windows:  win_num, bs, c, wh, ww
        x_windows = torch.cat(x_windows_out, 0)

        # x:  bs, c, h, w
        x = ka_window_reverse(x_windows, self.window_size, H, W)

        x = self.layernorm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = shortcut + x

        return x





##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.__class__.__name__ = 'XCTEB'
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(PatchMergeModule):
    def __init__(self,
        args,
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        ka_window_size = 16,
        kernel_size = 3,
        padding = 1,
        stride = 1,
        K = 2
    ):

        super(Restormer, self).__init__()
        self.args = args
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        ka_window_size = ka_window_size

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.kernelattention_down1 = KernelAttention(dim=dim, num_heads=heads[0], ka_window_size=ka_window_size, kernel_size=kernel_size, kernel_dim_scale=1, stride=stride, padding=padding, K=K, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        ka_window_size = ka_window_size // 2
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.kernelattention_down2 = KernelAttention(dim=int(dim*2**1), num_heads=heads[1], ka_window_size=ka_window_size, kernel_size=kernel_size, kernel_dim_scale=1, stride=stride, padding=padding, K=K, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        ka_window_size = ka_window_size // 2
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.kernelattention_down3 = KernelAttention(dim=int(dim*2**2), num_heads=heads[2], ka_window_size=ka_window_size, kernel_size=kernel_size, kernel_dim_scale=1, stride=stride, padding=padding, K=K, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        # ka_window_size = ka_window_size // 2

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.kernelattention_up3 = KernelAttention(dim=int(dim*2**2), num_heads=heads[2], ka_window_size=ka_window_size, kernel_size=kernel_size, kernel_dim_scale=1, stride=stride, padding=padding, K=K, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        ka_window_size = ka_window_size * 2

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.kernelattention_up2 = KernelAttention(dim=int(dim*2**1), num_heads=heads[1], ka_window_size=ka_window_size, kernel_size=kernel_size, kernel_dim_scale=1, stride=stride, padding=padding, K=K, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        ka_window_size = ka_window_size * 2
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.kernelattention_up1 = KernelAttention(dim=int(dim*2**1), num_heads=heads[0], ka_window_size=ka_window_size, kernel_size=kernel_size, kernel_dim_scale=1, stride=stride, padding=padding, K=K, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        ka_window_size = ka_window_size * 2
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1 = self.kernelattention_down1(out_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2 = self.kernelattention_down2(out_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc_level3 = self.kernelattention_down3(out_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        out_dec_level3 = self.kernelattention_up3(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.kernelattention_up2(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.kernelattention_up1(out_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1


    def train_step(self, *args, **kwargs):#data, *args, **kwargs):

        return self(*args, **kwargs)

        # return outputs, loss

    def val_step(self, *args, **kwargs):

        # metrics = {}
        #
        # O, B = batch['O'].cuda(), batch['B'].cuda()
        # samples = sub_mean(O)
        # derain = self.forward_chop(samples)
        # pred = quantize(add_mean(derain), 255)
        # normalized = pred[0]
        # tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
        #
        # # imageio.imwrite(os.path.join(saved_path, ''.join([batch['filename'][0], '.png'])),
        # #                 tensor_cpu.numpy())
        #
        # with torch.no_grad():
        #     metrics.update(ssim=reduce_mean(torch.mean(self.ssim(pred / 255.0, B))))
        #     metrics.update(psnr=reduce_mean(self.psnr(pred, B * 255.0, 4, 255.0)))

        return self.forward_chop(*args, **kwargs)

    # def set_metrics(self, criterion, rgb_range=255.):
    #
    #     self.criterion = criterion
    #     self.psnr = PSNR_ycbcr()
    #     self.ssim = SSIM(size_average=False, data_range=rgb_range)





class build_Restormer(DerainModel, name='Restormer_KAv2'):

    def __call__(self, cfg):
        scheduler = None
        loss = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = Restormer(cfg).cuda()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)  ## optimizer 1: Adam
        # model.set_metrics(criterion)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=20)
        return model, criterion, optimizer, scheduler

