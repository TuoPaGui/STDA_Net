import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from copy import deepcopy
import math
import flash_attention
import torch.distributions as dist
from torch.nn.utils import spectral_norm
from kymatio.torch import Scattering1D
import random

from torchaudio.compliance.kaldi import spectrogram
from torchdiffeq import odeint
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch, Data
import torch.nn.init as init
import librosa.display
import librosa
import itertools
from collections import OrderedDict

# 可视化
import matplotlib.pyplot as plt
from tensorflow.python.tools.optimize_for_inference_lib import INPUT_ORDER
import seaborn as sns

############################################ 频谱转换 ############################################
class SpectrogramTransform(nn.Module):
    def __init__(self, Fs=100,n_fft=256, hop_length=100, window_fn=torch.hamming_window,max_freq=30,):
        super(SpectrogramTransform, self).__init__()
        self.Fs = Fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window_fn(n_fft)

        self.max_freq = max_freq

    def forward(self, data):
        # 输入形状: (batch_size, num_channels, time_steps)
        batch_size, num_channels, _ = data.shape
        # 调整形状为 (batch_size*num_channels, time_steps)
        input_reshaped = data.view(batch_size * num_channels, -1)
        # 将窗函数移动到设备上，如果输入数据在GPU上
        window = self.window.to(data.device)

        # 计算STFT
        spec = torch.stft(input_reshaped, n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          window=window, center=True, normalized=True,
                          onesided=True, return_complex=True)

        # 新增频率截断逻辑
        freq_resolution = self.Fs / self.n_fft  # 频率分辨率
        max_freq_bin = int(self.max_freq // freq_resolution)  # 最大频率对应的bin索引
        max_freq_bin = min(max_freq_bin, spec.size(1)-1)  # 防止超出范围
        # 截断频率范围 (0-max_freq Hz)
        spec = spec[:, :max_freq_bin+1, :]  # 保留到max_freq_bin的频率点

        spec_power = torch.abs(spec) ** 2  # 计算功率谱
        spec_abs = spec_power

        # 对数尺度变换
        spec_abs = torch.log10(spec_abs + 1e-9)  # 加一个小量防止 log(0)

        # 逐样本逐通道标准化
        mean = spec_abs.mean(dim=(1, 2), keepdim=True)  # (batch*channels, 1, 1)
        std = spec_abs.std(dim=(1, 2), keepdim=True) + 1e-9
        spec_abs = (spec_abs - mean) / std

        # 调整形状并转置维度
        spec_abs = spec_abs.view(batch_size, num_channels, spec_abs.size(1), spec_abs.size(2))
        spec_abs = spec_abs.transpose(2, 3)  # (batch, channels, time, freq)

        return spec_abs

class PhysioGuidedEnhancer(nn.Module):
    def __init__(self, num_channels, num_filters, band_range, freq_res, num_bins):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.freq_res = freq_res
        self.num_bins = num_bins
        f_start, f_end = band_range

        center_pos = torch.linspace(f_start, f_end, num_filters) / freq_res
        center_pos = center_pos.view(1, num_filters, 1)
        self.raw_centers = nn.Parameter(center_pos.repeat(num_channels, 1, 1))

        std_val = ((f_end - f_start) / num_filters) / freq_res
        self.widths = nn.Parameter(torch.full((num_channels, num_filters, 1), std_val))

        # 可学习频带增强系数（每个滤波器）
        self.gains = nn.Parameter(torch.ones(num_channels, num_filters, 1))

        self.register_buffer('freqs', torch.arange(num_bins).float().view(1, 1, num_bins))

    def forward(self, spec_slice, freq_offset):
        # 计算每个滤波器的中心位置 mu（softplus 保证正值，再加上当前频段的起始 bin 偏移）
        mu = F.softplus(self.raw_centers) + freq_offset

        # 计算每个滤波器的宽度 std（softplus 保证正，+1e-3 防止为 0）
        std = F.softplus(self.widths) + 1e-3

        # 限制中心 mu 在当前频段范围内
        mu = mu.clamp(freq_offset, freq_offset + self.freqs.shape[-1] - 1)
        # 限制宽度 std，避免过窄或过宽（下限 0.5，上限约为该频段长度/滤波器数的两倍）
        std = std.clamp(0.5, 2 * self.freqs.shape[-1] / self.num_filters)
        # 生成高斯权重曲线，每个 (通道, 滤波器) 对应一条长度 F 的曲线
        gauss = torch.exp(-0.5 * ((self.freqs - mu) / std) ** 2)  # [C, N, F]
        # 对每条曲线在频率维度归一化，使权重和为 1
        # 在频率维度
        # F 上归一化之后，每条滤波器曲线的 所有点加起来 = 1。
        # 这时，曲线的峰值高度不再固定为 1，而是取决于 std：
        # std 大（曲线宽） → 峰值会很低（因为要分布到很多点上）。
        weights = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-6)
        # 乘以可学习的增强系数 gains（控制每个滤波器整体强度） 形状: [C, N, F]
        weights = self.gains * weights  # 加权增强
        # 按照权重在频率维度加权求和，把 [B, C, T, F] 压缩到 [B, C, T, N]
        return torch.einsum('bctf,cnf->bctn', spec_slice, weights)

class CrossBandSpatialAttention(nn.Module):
    def __init__(self, num_bands=6, num_channels=3, reduction=4):
        super().__init__()
        inter_channels = num_channels * 2
        self.fc1 = nn.Conv2d(num_bands * num_channels, inter_channels, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(inter_channels, num_bands * num_channels, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats):  # feats: [B, band, C, T, F]
        B, band, C, T, Freq = feats.shape
        x = feats.view(B, band * C, T, Freq)
        attn = self.fc2(self.act(self.fc1(x)))
        attn = self.softmax(attn.view(B, band, C, T, Freq))
        return attn

class SleepBandEnhancerPlusPlus(nn.Module):
    """
    关键改动：
      1) 去除“先插值到各自 F_band 再对齐到 max_f_band”的流程；
         现在：各频段先压到固定 N=num_filters（本就固定），直接做跨频带注意力。
      2) 重建阶段按每段原始 f_len 还原：默认 'linear'（逐位置线性层，等价频域 1x1 卷积）；
         可选 'interp'（双线性插值）以便快速对比。
      3) 修复了高斯中心坐标体系，避免绝对/相对坐标混用。
    """
    def __init__(self, num_channels=3, freq_bins=103, Fs=100, n_fft=256,
                 num_filters=6, reconstruct_mode: str = 'interp'):
        """
        reconstruct_mode: 'linear' | 'interp'
           - 'linear': 每频段使用 nn.Linear(num_filters -> f_len) 学习式还原（默认）
           - 'interp'：使用插值还原（便于做 ablation）
        """
        super().__init__()
        self.freq_res = Fs / n_fft
        self.num_channels = num_channels
        self.freq_bins = freq_bins
        self.num_filters = num_filters
        assert reconstruct_mode in ('linear', 'interp')
        self.reconstruct_mode = reconstruct_mode

        self.band_cfg = OrderedDict([
            ('delta', (0.5, 4)),
            ('theta', (4, 8)),
            ('alpha', (8, 12)),
            ('sigma', (12, 16)),
            ('beta1', (16, 24)),
            ('beta2', (24, 30))
        ])

        self.branches = nn.ModuleDict()
        self.align_convs = nn.ModuleDict()
        self.reconstructors = nn.ModuleDict()  # 仅在 linear 模式下用
        self.band_order = []

        for name, (f_start, f_end) in self.band_cfg.items():
            start_bin = int(f_start / self.freq_res)
            end_bin = int(f_end / self.freq_res)
            num_bins = end_bin - start_bin + 1

            # 频段内的生理引导增强（输出最后一维为 N=num_filters）
            self.branches[name] = PhysioGuidedEnhancer(
                num_channels, num_filters, (f_start, f_end), self.freq_res, num_bins
            )
            # 通道对齐（不改 H=W 形状）
            self.align_convs[name] = nn.Conv2d(num_channels, num_channels, kernel_size=1)

            # 线性重建：N -> f_len（每段各自的 f_len）
            if self.reconstruct_mode == 'linear':
                self.reconstructors[name] = nn.Linear(num_filters, num_bins, bias=True)

            self.band_order.append((name, start_bin, end_bin))

        self.cross_attn = CrossBandSpatialAttention(
            num_bands=len(self.band_cfg), num_channels=num_channels
        )

        # 每个频段整体权重（band-level gain）
        self.band_gain = nn.Parameter(torch.ones(len(self.band_order)))

        self.gate = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 1),
            nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 1),
            nn.BatchNorm2d(num_channels),
            nn.GELU()
        )

    def forward(self, spec):  # spec: [B, C, T, F]   e.g., [256, 4, 31, 77]
        B, C, T, Freq = spec.shape
        band_feats = []  # 每段形状：[B, C, T, N=num_filters]

        # 1) 逐频段：裁剪 -> 增强 -> 通道对齐；不再插值到 f_len
        for name, start_bin, end_bin in self.band_order:
            sliced = spec[:, :, :, start_bin:end_bin + 1]                  # [B, C, T, f_len]
            filtered = self.branches[name](sliced, start_bin)              # [B, C, T, N]
            aligned = self.align_convs[name](filtered)                     # [B, C, T, N]
            band_feats.append(aligned)

        # 2) 跨频带注意力（在 band 维 softmax），最后一维为固定 N=num_filters
        #    stack 后形状：[B, num_bands, C, T, N]
        band_stack = torch.stack(band_feats, dim=1)
        attn_weights = self.cross_attn(band_stack)
        band_stack = band_stack * attn_weights * self.band_gain.view(1, -1, 1, 1, 1)

        # 3) 重建到原始频轴：对每段从 N -> f_len，再写回各自 [start:end]
        enhanced = torch.zeros(B, C, T, Freq, device=spec.device, dtype=spec.dtype)

        for i, (name, start_bin, end_bin) in enumerate(self.band_order):
            f_len = end_bin - start_bin + 1
            narrow = band_stack[:, i]  # [B, C, T, N]

            if self.reconstruct_mode == 'linear':
                # 逐位置线性映射（等价于频域 1×1 卷积），学习式还原到 f_len
                y = narrow.reshape(B * C * T, self.num_filters)                  # [BCT, N]
                y = self.reconstructors[name](y)                                 # [BCT, f_len]
                band_feat = y.view(B, C, T, f_len)                               # [B, C, T, f_len]
            else:
                # 插值还原：N -> f_len
                band_feat = F.interpolate(narrow, size=(T, f_len),
                                          mode='bilinear', align_corners=False)  # [B, C, T, f_len]

            enhanced[:, :, :, start_bin:end_bin + 1] += band_feat

        # 4) 残差门控融合 + 投影
        residual = enhanced - spec
        enhanced = spec + self.gate(residual)*residual
        return self.proj(enhanced)


class Spectral_Enhancement(nn.Module):
    def __init__(self, num_channels=4, pool_size=(64, 64)):
        super(Spectral_Enhancement, self).__init__()

        # STFT 变换
        self.spectrogram = SpectrogramTransform()

        self.filterbank = SleepBandEnhancerPlusPlus(num_channels=num_channels)

        self.norm = nn.BatchNorm2d(num_channels)

    def forward(self, x):

        # STFT变换
        spec = self.spectrogram(x)
        # 频带自适应增强
        spec = self.filterbank(spec)

        return spec

# 动态卷积
class DynamicKernelConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, max_kernel_size=(7, 7), stride=1,
                 use_bn=True, use_act=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_kernel = max_kernel_size
        self.stride = stride
        self.use_bn = use_bn
        self.use_act = use_act

        # 静态权重：所有样本共享
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *max_kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # 动态参数预测网络：根据输入特征生成掩模参数
        self.param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 从整个特征图提取全局信息
            nn.Flatten(),
            nn.Linear(in_channels, 3),  # 输出 alpha_h, alpha_w, sigma
        )

        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            self.act = nn.GELU()

    def generate_gaussian_mask(self, kernel_size, center_h, center_w, sigma, device):
        H, W = kernel_size
        y = torch.linspace(0, 1, steps=H, device=device).unsqueeze(1).repeat(1, W)
        x = torch.linspace(0, 1, steps=W, device=device).unsqueeze(0).repeat(H, 1)
        # 每个批次样本的参数都不同
        gauss = torch.exp(-((x - center_w) ** 2 + (y - center_h) ** 2) / (2 * sigma ** 2))
        return gauss  # 形状: [B, H, W]

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # 1. 动态预测掩模参数
        params = self.param_predictor(x)
        alpha_h = F.sigmoid(params[:, 0]).view(B, 1, 1)  # 归一化到 (0,1)
        alpha_w = F.sigmoid(params[:, 1]).view(B, 1, 1)
        sigma = F.softplus(params[:, 2]).view(B, 1, 1) + 1e-3  # 保证正数

        # 2. 根据动态参数生成掩模
        # 每个批次样本都有一个独立的掩模
        mask2d = self.generate_gaussian_mask(
            self.max_kernel, alpha_h, alpha_w, sigma, device=device
        ).unsqueeze(1).unsqueeze(1)  # 形状: [B, 1, 1, H, W]

        # 3. 将静态权重与动态掩模融合
        # 这一步是关键，将批次维度的掩模应用到共享的权重上
        # 💡 使用 unsqueeze(0) 和广播来生成批次维度的动态权重
        weight_dynamic = self.weight.unsqueeze(0) * mask2d  # 形状: [B, O, I, H, W]

        pad_h = self.max_kernel[0] // 2
        pad_w = self.max_kernel[1] // 2

        # 4. 执行卷积操作
        # 由于 F.conv2d 不支持批次维度的权重，我们使用 `groups` 参数来模拟
        # 关键: 将输入 x 的批次维度和通道维度合并
        conv_out = F.conv2d(
            x.view(1, B * C, H, W),
            weight_dynamic.view(B * self.out_channels, C, *self.max_kernel),
            bias=self.bias.repeat(B),
            stride=self.stride,
            padding=(self.max_kernel[0] // 2, self.max_kernel[1] // 2),
            groups=B
        )
        # 现在 conv_out 已经被正确赋值了
        out = conv_out.view(B, self.out_channels, conv_out.shape[2], conv_out.shape[3])

        # 5. 应用 BN 和激活函数
        if self.use_bn:
            out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out

class TransformerFusion(nn.Module):
    def __init__(self, in_channels=256, embed_dim=100, num_heads=5, num_layers=2, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.project = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.project(x)                     # [B, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)        # [B, H*W, embed_dim]
        x = self.encoder(x)                     # [B, H*W, embed_dim]
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, -1, H, W) # [B, embed_dim, H, W]
        return self.dropout(x)

class DAMS_CNN(nn.Module):
    def __init__(self, in_channels=4, drate=0.5):
        super(DAMS_CNN, self).__init__()

        # 小感受野分支
        self.features1 = nn.Sequential(
            DynamicKernelConv2D(in_channels, 64, max_kernel_size=(4, 7), stride=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (31,77) -> (15,38)
            nn.Dropout2d(drate),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))  # 稍微压缩频域
        )

        # 大感受野分支
        self.features2 = nn.Sequential(
            DynamicKernelConv2D(in_channels, 64, max_kernel_size=(8, 13), stride=2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (15,38) -> (7,19)
            nn.Dropout2d(drate),

            nn.Conv2d(64, 128, kernel_size=(3, 9), stride=1, padding=(1, 4), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))  # 控制下采样速率
        )

        # 融合：拼接 + 通道注意力
        self.fusion = TransformerFusion(in_channels=256, embed_dim=100, num_heads=5)

    def forward(self, x):  # x: (B, 4, 31, 103)

        x1 = self.features1(x)  # (B, 128, H1, W1)
        x2 = self.features2(x)  # (B, 128, H2, W2)

        # 对齐空间大小
        if x1.shape[2:] != x2.shape[2:]:
            target_size = (min(x1.shape[2], x2.shape[2]), min(x1.shape[3], x2.shape[3]))
            x1 = F.adaptive_avg_pool2d(x1, target_size)
            x2 = F.adaptive_avg_pool2d(x2, target_size)

        x_fused = torch.cat([x1, x2], dim=1)  # [B, 256, H, W]
        x_fused = self.fusion(x_fused)        # [B, 128, H, W]

        return x_fused

################################################ 残差因果TCN ###################################################
class ResidualTCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size, dilation, dropout=0.3, expansion=2, causal=False):
        super().__init__()
        hidden_dim = d_model * expansion
        self.causal = causal

        if causal:
            self.padding = (kernel_size - 1) * dilation
            self.pad = nn.ConstantPad1d((self.padding, 0), 0)  # 只在左侧 pad
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2
            self.pad = nn.ConstantPad1d((self.padding, self.padding), 0)  # 双向 pad（same padding）

        self.block = nn.Sequential(
            self.pad,
            nn.Conv1d(d_model, hidden_dim, kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(hidden_dim, d_model, kernel_size=1),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out

class TemporalTCNBlock(nn.Module):
    def __init__(self, d_model=100, levels=4, kernel_size=3, dropout=0.3, causal=False):
        super().__init__()
        self.network = nn.Sequential(*[
            ResidualTCNBlock(d_model, kernel_size, dilation=2 ** i, dropout=dropout, causal=causal)
            for i in range(levels)
        ])

    def forward(self, x):  # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.network(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return x

# Attention-Guided global-local encoder
class AGLE(nn.Module):
    def __init__(self, d_model=128, dropout=0.3, n_heads=4, use_layerscale=False, causal=False):
        super().__init__()
        self.d_model = d_model
        self.use_layerscale = use_layerscale
        self.causal = causal

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.tcn = TemporalTCNBlock(d_model=d_model, dropout=dropout, causal=causal)

        # 增强 gate 的表达能力、保证训练稳定性，最终生成一个更有效、更可控的动态门控信号
        self.guide_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        self.gamma = nn.Parameter(torch.ones(1, 1, d_model) * 1e-3) if use_layerscale else None

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        self.debug_counter = 0
        self.debug_every = 30  # 每 N 次打印一次
        self.debug_flag = False
    def forward(self, x):  # x: (B, T, C)
        self.debug_counter += 1
        B, T, _ = x.size()

        # Pre-LN 保证 Attention 输入特征平稳，提升训练稳定性
        x_norm1 = self.norm1(x)

        if self.causal:
            attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        else:
            attn_mask = None  # 看全局
        # attn_out(1,256,128)  attn_weights:(1,256,256)
        attn_out, attn_weights = self.attn(x_norm1, x_norm1, x_norm1, attn_mask=attn_mask)

        # attn_weights:(1,256,256) x_norm1(1,256,128) batch matrix multiply 批量矩阵乘法
        query_summary = torch.bmm(attn_weights, x_norm1)  # (1,256,128)
        gate = torch.sigmoid(self.guide_proj(query_summary))  # 控制调制强度

        x_tcn = self.tcn(x)
        x_tcn = x_tcn * (1 + gate)

        fusion = torch.cat([x_tcn, attn_out], dim=-1)  # (B, T, 2C)
        fusion = self.fusion_proj(fusion)  # → (B, T, C)

        if self.gamma is not None:
            fusion = self.gamma * fusion  # 控制融合后的表达强度
        fusion_norm = self.norm2(fusion)
        # 前馈神经网络子层 提升特征表达能力，增加特征交互，加深网络的非线性建模能力
        # FFN 是浅层增强，不负责重建复杂全局依赖，残差帮助融合上下文和局部增强信息。
        out = self.feedforward(fusion_norm) + fusion
        return out

############################################# 主干网络 ##############################################
class STDA_Net(nn.Module):
    def __init__(self, in_channels=4, num_classes=5, base_filters=64, dropout=0.3):
        super().__init__()

        self.spectral_enhancement = Spectral_Enhancement(num_channels = in_channels)

        self.dams_cnn = DAMS_CNN(in_channels = in_channels)

        # Use Hybrid TCN+Transformer
        self.agle = AGLE(
            d_model=100,
            dropout=0.2,
            n_heads=5,
            causal=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(100, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.feature_norm = nn.LayerNorm(128)

    def forward(self, data, labels, criterion=None):

        spec_data = self.spectral_enhancement(data)

        features = self.dams_cnn(spec_data)  # [B, 128, 8, 8]

        pooled = self.global_pool(features)  # [B, 128]

        feat_encoded = self.agle(pooled.unsqueeze(0)).squeeze(0)

        logits = self.classifier(feat_encoded)

        if criterion is not None and labels is not None:
            ce_loss = criterion(logits, labels)
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)
        return {'logits': logits, 'total_loss': ce_loss, 'features': feat_encoded}
