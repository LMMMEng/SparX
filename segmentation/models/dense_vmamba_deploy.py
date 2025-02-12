import os
import copy
import time
import math
import torch
import itertools
import numpy as np
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from torch.utils import checkpoint
from collections import OrderedDict
from einops import rearrange, repeat
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log
from timm.models.layers import DropPath, trunc_normal_
from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN



import sys
sys.path.append("models/mamba/")
from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
from reparam_block import DilatedReparamBlock


# import selective scan ==============================
import selective_scan_cuda
import selective_scan_cuda_core
import selective_scan_cuda_oflex



# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, 
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()
        

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_groups=1, num_channels=num_channels, **kwargs)



class Mlp(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, mlp_kernel=3, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        if mlp_kernel is not None:
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=mlp_kernel, padding=mlp_kernel//2, groups=hidden_features)     
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        
        x = self.fc1(x)
        if hasattr(self, 'dwconv'):
            x = x + self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


# =====================================================
class SS2D(nn.Module):

    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        expansion_ratio=2.0,
        ssm_ratio=1,
        ssm_stride=1,
        dt_rank="auto",
        norm_layer=LayerNorm2d,
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv
        # ======================
        dropout=0.0,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        stage_idx=None,
        stage_dense_idx=None,
        layer_idx=None,
        max_dense_depth=None,
        dense_step=None,
        is_cross_layer=None,
        dense_layer_idx=None,
        dense_expansion=None,
        base_dim=None,
        # ======================
        group_dim=64,
        sr_ratio=1,
        use_triton=True,
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        ssm_dim = int(d_inner * ssm_ratio)
        identity_dim = d_inner - ssm_dim
        self.use_triton = use_triton
        
        self.layer_idx = layer_idx
        self.ssm_dim = ssm_dim
        self.identity_dim = identity_dim
        
        attn_dim = int(d_model * 0.75)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        k_group = 4

        # in proj =======================================
        # d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = nn.Conv2d(d_model, d_inner, kernel_size=1)
        # self.local_conv = nn.Conv2d(d_inner, d_inner, kernel_size=7, padding=3, groups=d_inner)
        self.act = act_layer()

        
        if ssm_stride > 1:
            self.ssm_downsample = nn.Sequential(
                nn.Conv2d(d_inner, d_inner, kernel_size=ssm_stride+1, padding=(ssm_stride+1)//2, stride=ssm_stride, groups=d_inner, bias=False),
                nn.BatchNorm2d(d_inner),
            )
        
        self.is_cross_layer = is_cross_layer
        
        if is_cross_layer:

            if dense_layer_idx == stage_dense_idx[0]:
                intra_dim = d_model
                inner_dim = d_model * dense_layer_idx
            else:
                count = 0
                for item in stage_dense_idx:
                    if item == dense_layer_idx:
                        break
                    count += 1
                count = min(max_dense_depth, count)
                intra_dim = d_model * count
                inner_dim = d_model * (dense_step - 1)
                
            current_d_dim = int(intra_dim + inner_dim)

            self.d_norm = norm_layer(current_d_dim)
            del self.in_proj
            self.d_proj = nn.Sequential(
                nn.Conv2d(current_d_dim, current_d_dim, kernel_size=3, padding=1, groups=current_d_dim, bias=False),
                nn.BatchNorm2d(current_d_dim),
                nn.GELU(),
                nn.Conv2d(current_d_dim, d_inner, kernel_size=1),
                nn.GELU(),
            )
            
            self.channel_padding = d_inner - d_inner//3*3

            self.q = nn.Sequential(
                self.get_sr(d_model, sr_ratio),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
            )
            
            self.k = nn.Sequential(
                self.get_sr(d_model, sr_ratio),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
            )
            
            self.v = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
            )
            
            self.in_proj = nn.Sequential(
                nn.Conv2d(d_model*3, d_model*3, kernel_size=3, padding=1, groups=d_model*3),
                nn.GELU(),
                nn.Conv2d(d_model*3, int(d_inner//3*3), kernel_size=1, groups=3),     
            )

        # conv =======================================
        if d_conv > 1:
            
            # self.conv2d = nn.Conv2d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=(d_conv-1)//2)
            self.conv2d = DilatedReparamBlock(d_inner, kernel_size=5, deploy=True)
            
            
        # out proj =======================================
        self.out_norm = norm_layer(ssm_dim)
        self.out_proj = nn.Conv2d(d_inner, d_model, kernel_size=1)
        
        # self.x_proj = nn.Conv1d(ssm_dim*k_group, (dt_rank+d_state*2)*k_group, kernel_size=1, groups=k_group)
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj
        
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, ssm_dim, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, ssm_dim, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(ssm_dim, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * ssm_dim)))
            self.A_logs = nn.Parameter(torch.randn((k_group * ssm_dim, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, ssm_dim, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, ssm_dim))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * ssm_dim)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * ssm_dim, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, ssm_dim, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, ssm_dim)))    
            
    @staticmethod
    def get_sr(dim, sr_ratio):
        
        if sr_ratio > 1:
            sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=(sr_ratio+1)//2, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
        else:
            sr = nn.Identity()
            
        return sr
      
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None, backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    def _cross_scan(self, x):
        if self.use_triton:
            return CrossScanTriton.apply(x)
        else:
            return CrossScan.apply(x)
    
    def _cross_merge(self, x):
        if self.use_triton:
            return CrossMergeTriton.apply(x)
        else:
            return CrossMerge.apply(x)
    
    def reparameterize_layer(self):
        if hasattr(self.conv2d, 'merge_dilated_branches'):
            self.conv2d.merge_dilated_branches()
    
    
    def forward_ssm(
        self,
        x,
        # ==============================
        to_dtype=False, # True: final out to dtype
        force_fp32=False, # True: input fp32
    ):

        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        
        
        if hasattr(self, 'ssm_downsample'):
            _H, _W = x.shape[2:]
            x = self.ssm_downsample(x)
  
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        
        xs = self._cross_scan(x)
        
        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight, bias=None, groups=K)
        # x_dbl = self.x_proj(xs.reshape(B, -1, L))
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        xs = xs.reshape(B, -1, L)
        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)
              
        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)
                  
        ys = self._selective_scan(xs, 
                                  dts, 
                                  As, 
                                  Bs, 
                                  Cs, 
                                  Ds, 
                                  delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)


        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, H, W)
        
        if hasattr(self, 'ssm_downsample'):
            # import pdb; pdb.set_trace()
            y = F.interpolate(y, size=(_H, _W), mode='bilinear', align_corners=False)
        
        # y = self._selective_merge(ys.reshape(B, K, -1, H, W), x)
        y = self.out_norm(y)
        
        return (y.to(x.dtype) if to_dtype else y)

    def forward(self, x, shortcut):
        
        if self.is_cross_layer:
            
            s = self.d_proj(self.d_norm(shortcut))
            k, v = torch.chunk(s, 2, dim=1)
            
            q = self.q(x)
            k = self.k(k)
            v = self.v(v)
            
            B, C, H, W = x.shape
            sr_H, sr_W = q.shape[2:]
            g_dim = q.shape[1] // 4
                        
            q = q.reshape(-1, g_dim, sr_H*sr_W)
            k = k.reshape(-1, g_dim, sr_H*sr_W)
            v = v.reshape(-1, g_dim, H*W)
            
            scale = q.shape[-1] ** -0.5
            attn = q @ k.transpose(-1, -2) * scale
            attn = F.softmax(attn, dim=-1)
            attn = (attn @ v).reshape(B, -1, H, W)
            # attn = F.scaled_dot_product_attention(q, k, v).reshape(B, -1, H, W)
            
            if (H, W) == (sr_H, sr_W):
                x = q.reshape(B, -1, H, W)
                
            x = torch.cat([x, attn, v.reshape(B, -1, H, W)], dim=1)
            x = rearrange(x, 'b (c g) h w -> b (g c) h w', g=3).contiguous()
            x = self.in_proj(x)
            
            if self.channel_padding > 0:
                if self.channel_padding == 1:
                    pad = torch.mean(x, dim=1, keepdim=True)
                    x = torch.cat([x, pad], dim=1)
                else:
                    pad = rearrange(x, 'b c h w -> b (h w) c')
                    pad = F.adaptive_avg_pool1d(pad, self.channel_padding)
                    pad = rearrange(pad, 'b (h w) c -> b c h w', h=H, w=W)
                    x = torch.cat([x, pad], dim=1)
        else:
            x = self.in_proj(x)
            # x = F.conv2d(x, weight=self.in_proj.weight[:, :, None, None], bias=self.in_proj.bias)
            
        if self.d_conv > 1:
            x = self.conv2d(x) # (b, d, h, w)
            # x = F.gelu(x)
            # if hasattr(self, 'conv_proj'):
            #     x = rearrange(x, 'b (g c) h w -> b (c g) h w', g=4)
            #     x = self.conv_proj(x)
            
        x = self.act(x)
        
        # if self.ssm_dim == x.shape[1]:
        #     y = self.forward_ssm(x, shortcut=None)
        # else:
        #     x, identity = torch.split(x, dim=1, split_size_or_sections=[self.ssm_dim, self.identity_dim])
        #     y = self.forward_ssm(x, shortcut=None)
        #     y = torch.cat([y, identity], dim=1)
        #     y = rearrange(y, 'b (g c) h w -> b (c g) h w', g=2)
        
        y = self.forward_ssm(x)   

        out = self.out_proj(y)
        out = self.dropout(out)
        
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = LayerNorm2d,
        # =============================
        ssm_d_state: int=16,
        expansion_ratio=2.0,
        ssm_ratio=1,
        ssm_stride=1,
        ssm_dt_rank = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        layer_idx=None,
        stage_idx=None,
        stage_dense_idx=None,
        max_dense_depth=None,
        dense_step=None,
        is_cross_layer=None,
        dense_layer_idx=None,
        dense_expansion=None,
        base_dim=None,
        group_dim=64,
        sr_ratio=1,
        ls_init_value=1e-5,
        use_triton=True,
        **kwargs,
    ):
        super().__init__()
        
        self.ssm_branch = expansion_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.pos_embed = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                expansion_ratio=expansion_ratio,
                ssm_ratio=ssm_ratio,
                ssm_stride=ssm_stride,
                dt_rank=ssm_dt_rank,
                norm_layer=norm_layer,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                # ==========================
                layer_idx=layer_idx,
                stage_idx=stage_idx,
                stage_dense_idx=stage_dense_idx,
                max_dense_depth=max_dense_depth,
                dense_step=dense_step,
                is_cross_layer=is_cross_layer,
                dense_layer_idx=dense_layer_idx,
                dense_expansion=dense_expansion,
                base_dim=base_dim,
                group_dim=group_dim,
                sr_ratio=sr_ratio,
                use_triton=use_triton,
                **kwargs,
            )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if self.mlp_branch:
           
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate)

        
        if ls_init_value is not None:
            self.layerscale_1 = LayerScale(hidden_dim, init_value=ls_init_value)
            self.layerscale_2 = LayerScale(hidden_dim, init_value=ls_init_value)
        else:
            self.layerscale_1 = nn.Identity()
            self.layerscale_2 = nn.Identity()
    
    def _forward(self, x, shortcut):
        
        x = x + self.pos_embed(x)
        
        if self.ssm_branch:
            x = self.layerscale_1(x) + self.drop_path(self.op(self.norm(x), shortcut)) # Token Mixer
            
        if self.mlp_branch:
            x = self.layerscale_2(x) + self.drop_path(self.mlp(self.norm2(x))) # FFN
            
        shortcut = x

        return (x, shortcut)

    def forward(self, input):

        input, shortcut = input
        
        if shortcut is not None:
            shortcut = torch.cat(shortcut, dim=1)
        
        if self.use_checkpoint and input.requires_grad:
            return checkpoint.checkpoint(self._forward, input, shortcut, use_reentrant=True)
        else:
            return self._forward(input, shortcut)


class VSSM(nn.Module):
    def __init__(
        self,
        pretrained=None,
        deploy=False,
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 5, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=1,
        expansion_ratio=[2, 2, 2, 2],
        ssm_ratio=[1, 1, 1, 1],
        ssm_stride=[1, 1, 1, 1],
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,        
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v3noz",
        # =========================
        mlp_ratio=[4, 4, 4, 4],
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        ls_init_value=[None, None, None, None],
        # =========================
        drop_path_rate=0,
        norm_layer=LayerNorm2d,
        use_checkpoint=[0, 0, 0, 0],
        dense_config=None,
        group_dim=[64, 64, 128, 256],
        sr_ratio=[8, 4, 2, 1],
        stem_type='v1',
        use_triton=True,
        **kwargs,
    ):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        
        dense_idx = []
        for i in range(4):
            dense_step = dense_config['dense_step'][i]
            dense_start = dense_config['dense_start'][i]
            d_idx = [i for i in range(depths[i] - 1, dense_start - 1, -dense_step)][::-1]
            dense_idx.append(d_idx)

        dense_config.update(dense_idx=dense_idx)
        self.dense_config = dense_config
        
        print('dense_config = ', dense_config)
        
        if stem_type == 'v2':
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_chans, dims[0]//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]//2),
                nn.GELU(),        
                nn.Conv2d(dims[0]//2, dims[0]//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]//2),
                nn.GELU(),
                nn.Conv2d(dims[0]//2, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
            )
            
        else:
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_chans, dims[0]//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]//2),
                nn.GELU(),        
                nn.Conv2d(dims[0]//2, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
            )
        
        self.layers = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
       
            downsample = nn.Sequential(
                nn.Conv2d(self.dims[i_layer], self.dims[i_layer+1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.dims[i_layer+1]),
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint[i_layer],
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                expansion_ratio=expansion_ratio[i_layer],
                ssm_ratio=ssm_ratio[i_layer],
                ssm_stride=ssm_stride[i_layer],
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio[i_layer],
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                ls_init_value=ls_init_value[i_layer],
                # =================
                stage_idx=i_layer,
                max_dense_depth=dense_config["max_dense_depth"][i_layer],
                dense_step=dense_config["dense_step"][i_layer],
                dense_idx=dense_config["dense_idx"][i_layer],
                dense_expansion=dense_config["dense_expansion"][i_layer],
                base_dim=dense_config["base_dim"][i_layer],
                group_dim=group_dim[i_layer],
                sr_ratio=sr_ratio[i_layer],
                use_triton=use_triton,
                **kwargs,
            ))
            
            self.norm.append(LayerNorm2d(self.dims[i_layer], eps=1e-6))

        self.apply(self._init_weights)

        if pretrained is not None:
            load_checkpoint(self, pretrained, logger='current')

        if deploy:
            print_log('reparameterizing model', logger='current')
            self.reparameterize_model()
        
    def _init_weights(self, m: nn.Module):
        
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def reparameterize_model(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize_layer'):
                m.reparameterize_layer()
 
    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        ssm_d_state=16,
        expansion_ratio=2.0,
        ssm_ratio=1,
        ssm_stride=1,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        ls_init_value=None,
        # ===========================
        stage_idx=None,
        max_dense_depth=None,
        dense_step=None,
        dense_idx=None,
        dense_expansion=None,
        base_dim=None,
        group_dim=64,
        sr_ratio=1,
        use_triton=True,
        **kwargs,
    ):

        depth = len(drop_path)
        dense_layer_count = -1

        blocks = []
        for d in range(depth):

            if d in dense_idx:
                is_cross_layer = True
                dense_layer_idx = d
            else:
                is_cross_layer = False
                dense_layer_idx = None
            
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                expansion_ratio=expansion_ratio,
                ssm_ratio=ssm_ratio,
                ssm_stride=ssm_stride,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                ls_init_value=ls_init_value,
                use_checkpoint=(d<use_checkpoint),
                # ============================
                layer_idx=d,
                stage_idx=stage_idx,
                max_dense_depth=max_dense_depth,
                dense_step=dense_step,
                is_cross_layer=is_cross_layer,
                stage_dense_idx=dense_idx,
                dense_layer_idx=dense_layer_idx,
                dense_expansion=dense_expansion,
                base_dim=base_dim,
                group_dim=group_dim,
                sr_ratio=sr_ratio,
                use_triton=use_triton,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    
    def layer_forward(self, layers, x, s, dense_cfg, stage_idx):

        max_dense_depth, dense_idx = dense_cfg
        
        inner_list = []
        cross_list = []
        
        if s is not None:
            inner_list.append(s)
        
        for idx, layer in enumerate(layers.blocks):
            
            if idx in dense_idx:
                    
                input = (x, inner_list)
                
                if len(cross_list) > 0:
                    inner_list.extend(cross_list)
                                 
                x, s = layer(input)
                
                cross_list.append(s)
                
                inner_list = []
            else:
                input = (x, None)
                x, s = layer(input)
                inner_list.append(s)
                
            if (max_dense_depth is not None) and len(cross_list) > max_dense_depth:
                cross_list = cross_list[-max_dense_depth:]
              
        # x = layers.downsample(x)
        
        return x
    
    def forward(self, x):
        
        outs = []
        
        d_cfg = self.dense_config
        s = None
        x = self.patch_embed(x)
        
        for idx, layer in enumerate(self.layers):
            
            max_dense_depth = d_cfg['max_dense_depth'][idx]
            dense_idx = d_cfg['dense_idx'][idx]
            cfg = (max_dense_depth, dense_idx)
            
            x = self.layer_forward(layer, x, s, cfg, idx)
            outs.append(self.norm[idx](x))
            x = layer.downsample(x)
            s = x
        
        return outs


def _cfg(url=None, **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.875,
        'interpolation': 'bicubic',  # 'bilinear' or 'bicubic'
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }   


@MODELS.register_module()
def densevmamba_t_deploy(**kwargs):
    
    '''
     
    Mask R-CNN
    GFLOPs = 
    Params(M) = 
    
    '''
    
    dense_config = {
        'dense_step': [1, 1, 2, 1], ## highly impact params and speed
        'dense_start': [100, 1, 0, 0],
        'max_dense_depth': [100, 100, 4, 100],
        'dense_expansion': [1, 1, 1, 1],
        'base_dim': [None, None, None, None]}
    

    model =  VSSM(depths=[2, 2, 7, 2],
                  dims=[96, 192, 320, 512],
                  expansion_ratio=[2, 2, 2, 2],
                  group_dim=[24, 48, 80, 128],
                  sr_ratio=[8, 4, 2, 1],
                  # =========================
                  ssm_d_state=1,
                  ssm_ratio=[1, 1, 1, 1],
                  ssm_stride=[1, 1, 1, 1],
                  ssm_dt_rank="auto",
                  ssm_act_layer=nn.SiLU,
                  ssm_conv=5,
                  ssm_conv_bias=False,
                  ssm_drop_rate=0.0,
                  ssm_init="v0",
                  # =========================
                  mlp_ratio=[4, 4, 4, 4],
                  mlp_act_layer=nn.GELU,
                  mlp_drop_rate=0.0,
                  ls_init_value=[None, None, 1, 1],
                  # =========================
                  # drop_path_rate=0.1,
                  # norm_layer=partial(LayerNorm2d, eps=1e-6),
                  norm_layer=GroupNorm,
                  # use_checkpoint=[0, 0, 0, 0],
                  dense_config=dense_config,
                  **kwargs)
        
    return model



# @MODELS.register_module()
# def densevmamba_s(pretrained=None, deploy=False, **kwargs):
    
#     '''
    
#     Mask R-CNN
#     GFLOPs = 
#     Params(M) = 
    
#     '''
    
#     dense_config = {
#         'dense_step': [1, 1, 3, 1], ## highly impact params and speed
#         'dense_start': [100, 1, 0, 0],
#         'max_dense_depth': [100, 100, 3, 100],
#         'dense_expansion': [1, 1, 1, 1],
#         'base_dim': [None, None, None, None],
#         'cross_stage': [False, False, False, False],
#     }
    
#     model =  VSSM(depths=[2, 2, 18, 2],
#                   dims=[96, 192, 320, 544],
#                   expansion_ratio=[2, 2, 2, 2],
#                   group_dim=[24, 48, 80, 128],
#                   sr_ratio=[8, 4, 2, 1],
#                   # =========================
#                   ssm_d_state=1,
#                   ssm_ratio=[1, 1, 1, 1],
#                   ssm_stride=[1, 1, 1, 1],
#                   ssm_dt_rank="auto",
#                   ssm_act_layer=nn.SiLU,
#                   ssm_conv=5,
#                   ssm_conv_bias=False,
#                   ssm_drop_rate=0.0,
#                   ssm_init="v0",
#                   # =========================
#                   mlp_ratio=[4, 4, 4, 4],
#                   mlp_act_layer=nn.GELU,
#                   mlp_drop_rate=0.0,
#                   ls_init_value=[None, None, 1, 1],
#                   # =========================
#                   # drop_path_rate=0.1,
#                   # norm_layer=partial(LayerNorm2d, eps=1e-6),
#                   norm_layer=GroupNorm,
#                   # use_checkpoint=[0, 0, 0, 0],
#                   stem_type='v2',
#                   dense_config=dense_config,
#                   **kwargs)
    
#     if pretrained is not None:
#         load_checkpoint(model, pretrained, logger='current')
    
#     if deploy:
#         print_log('reparameterizing model', logger='current')
#         model.reparameterize_model()
    
#     return model



@MODELS.register_module()
def densevmamba_s3_deploy(**kwargs):
    
    '''
    
    Mask R-CNN
    GFLOPs = 
    Params(M) = 
    
    '''
    
    dense_config = {
        'dense_step': [1, 1, 3, 1], ## highly impact params and speed
        'dense_start': [100, 1, 0, 0],
        'max_dense_depth': [100, 100, 3, 100],
        'dense_expansion': [1, 1, 1, 1],
        'base_dim': [None, None, None, None],
        'cross_stage': [False, False, False, False],
    }
    
    model =  VSSM(depths=[2, 2, 17, 2],
                  dims=[96, 192, 328, 544],
                  expansion_ratio=[2, 2, 2, 2],
                  group_dim=[24, 48, 80, 128],
                  sr_ratio=[8, 4, 2, 1],
                  # =========================
                  ssm_d_state=1,
                  ssm_ratio=[1, 1, 1, 1],
                  ssm_stride=[1, 1, 1, 1],
                  ssm_dt_rank="auto",
                  ssm_act_layer=nn.SiLU,
                  ssm_conv=5,
                  ssm_conv_bias=False,
                  ssm_drop_rate=0.0,
                  ssm_init="v0",
                  # =========================
                  mlp_ratio=[4, 4, 4, 4],
                  mlp_act_layer=nn.GELU,
                  mlp_drop_rate=0.0,
                  ls_init_value=[None, None, 1, 1],
                  # =========================
                  # drop_path_rate=0.1,
                  # norm_layer=partial(LayerNorm2d, eps=1e-6),
                  norm_layer=GroupNorm,
                  # use_checkpoint=[0, 0, 0, 0],
                  stem_type='v2',
                  dense_config=dense_config,
                  **kwargs)
    
    # if pretrained is not None:
    #     load_checkpoint(model, pretrained, logger='current')
    
    # if deploy:
    #     print_log('reparameterizing model', logger='current')
    #     model.reparameterize_model()
    
    return model


# @MODELS.register_module()
# def densevmamba_b(pretrained=None, deploy=False, **kwargs):
    
#     '''
#     GFLOPs = 15.6
#     Params(M) = 82.8
#     FPS = 652
    
#     Baseline:
#     89M 
#     15.4G
#     754 FPS
#     '''
    
#     dense_config = {
#         'dense_step': [1, 1, 3, 1], ## highly impact params and speed
#         'dense_start': [100, 1, 0, 0],
#         'max_dense_depth': [100, 100, 3, 100],
#         'dense_expansion': [1, 1, 1, 1],
#         'base_dim': [None, None, None, None],
#         'cross_stage': [False, False, False, False],
#     }
    
    
#     model =  VSSM(depths=[2, 2, 20, 2],
#                   dims=[120, 240, 408, 696],
#                   expansion_ratio=[2, 2, 2, 2],
#                   group_dim=[24, 48, 80, 128],
#                   sr_ratio=[8, 4, 2, 1],
#                   # =========================
#                   ssm_d_state=1,
#                   ssm_ratio=[1, 1, 1, 1],
#                   ssm_stride=[1, 1, 1, 1],
#                   ssm_dt_rank="auto",
#                   ssm_act_layer=nn.SiLU,
#                   ssm_conv=5,
#                   ssm_conv_bias=False,
#                   ssm_drop_rate=0.0,
#                   ssm_init="v0",
#                   # =========================
#                   mlp_ratio=[4, 4, 4, 4],
#                   mlp_act_layer=nn.GELU,
#                   mlp_drop_rate=0.0,
#                   ls_init_value=[None, None, 1, 1],
#                   # =========================
#                   # drop_path_rate=0.1,
#                   # norm_layer=partial(LayerNorm2d, eps=1e-6),
#                   norm_layer=GroupNorm,
#                   stem_type='v2',
#                   # use_checkpoint=[0, 0, 0, 0],
#                   dense_config=dense_config,
#                   **kwargs
#             )
    
#     if pretrained is not None:
#         load_checkpoint(model, pretrained, logger='current')
    
#     if deploy:
#         print_log('reparameterizing model', logger='current')
#         model.reparameterize_model()
    
#     return model



@MODELS.register_module()
def densevmamba_bv1_deploy(**kwargs):
    
    '''
    GFLOPs = 
    Params(M) = 
    FPS = 
    
    Baseline:
    89M 
    15.4G
    754 FPS
    '''
    
    dense_config = {
        'dense_step': [1, 1, 3, 1], ## highly impact params and speed
        'dense_start': [100, 1, 0, 2],
        'max_dense_depth': [100, 100, 3, 100],
        'dense_expansion': [1, 1, 1, 1],
        'base_dim': [None, None, None, None],
        'cross_stage': [False, False, False, False],
    }
    
    model =  VSSM(depths=[2, 2, 21, 3],
                  dims=[120, 240, 396, 636],
                  expansion_ratio=[2, 2, 2, 2],
                  ssm_conv=5,
                  ls_init_value=[None, None, 1, 1],
                  norm_layer=GroupNorm,
                  stem_type='v2',
                  dense_config=dense_config,
                  **kwargs)
    
    # if pretrained is not None:
    #     load_checkpoint(model, pretrained, logger='current')
    
    # if deploy:
    #     print_log('reparameterizing model', logger='current')
    #     model.reparameterize_model()
    
    return model



@MODELS.register_module()
def densevmamba_bv3_deploy(**kwargs):
    
    '''
    GFLOPs = 15.9
    Params(M) =  83.5
    FPS = 630
    
    Baseline:
    89M 
    15.4G
    754 FPS
    
    Params =  105.913571
    FLOPs =  500.17293664
    
    '''
    
    dense_config = {
        'dense_step': [1, 1, 3, 1], ## highly impact params and speed
        'dense_start': [100, 1, 0, 1],
        'max_dense_depth': [100, 100, 3, 100],
        'dense_expansion': [1, 1, 1, 1],
        'base_dim': [None, None, None, None],
        'cross_stage': [False, False, False, False],
    }
    
    
    model =  VSSM(depths=[2, 2, 22, 3],
                  dims=[120, 240, 384, 660],
                  expansion_ratio=[2, 2, 2, 2],
                  ssm_conv=5,
                  ls_init_value=[None, None, None, None],
                  norm_layer=GroupNorm,
                  stem_type='v2',
                  dense_config=dense_config,
                  **kwargs)
    
    # if pretrained is not None:
    #     load_checkpoint(model, pretrained, logger='current')
    
    # if deploy:
    #     print_log('reparameterizing model', logger='current')
    #     model.reparameterize_model()
    
    return model


# @MODELS.register_module()
# def densevmamba_bv3(pretrained=None, deploy=False, **kwargs):
    
#     '''
#     GFLOPs = 15.9
#     Params(M) =  83.5
#     FPS = 630
    
#     Baseline:
#     89M 
#     15.4G
#     754 FPS
    
#     Params =  105.913571
#     FLOPs =  500.17293664
    
#     '''
    
#     dense_config = {
#         'dense_step': [1, 1, 3, 1], ## highly impact params and speed
#         'dense_start': [100, 1, 0, 1],
#         'max_dense_depth': [100, 100, 3, 100],
#         'dense_expansion': [1, 1, 1, 1],
#         'base_dim': [None, None, None, None],
#         'cross_stage': [False, False, False, False],
#     }
    
    
#     model =  VSSM(depths=[2, 2, 22, 3],
#                   dims=[120, 240, 384, 660],
#                   expansion_ratio=[2, 2, 2, 2],
#                   group_dim=[24, 48, 80, 128],
#                   sr_ratio=[8, 4, 2, 1],
#                   # =========================
#                   ssm_d_state=1,
#                   ssm_ratio=[1, 1, 1, 1],
#                   ssm_stride=[1, 1, 1, 1],
#                   ssm_dt_rank="auto",
#                   ssm_act_layer=nn.SiLU,
#                   ssm_conv=5,
#                   ssm_conv_bias=False,
#                   ssm_drop_rate=0.0,
#                   ssm_init="v0",
#                   # =========================
#                   mlp_ratio=[4, 4, 4, 4],
#                   mlp_act_layer=nn.GELU,
#                   mlp_drop_rate=0.0,
#                   ls_init_value=[None, None, None, None],
#                   # =========================
#                   # drop_path_rate=0.1,
#                   # norm_layer=partial(LayerNorm2d, eps=1e-6),
#                   norm_layer=GroupNorm,
#                   # use_checkpoint=[0, 0, 0, 0],
#                   stem_type='v2',
#                   dense_config=dense_config,
#                   **kwargs)
    
#     if pretrained is not None:
#         load_checkpoint(model, pretrained, logger='current')
    
#     return model


if __name__ == "__main__":
    import sys
    from time import sleep
    sys.path.append(".")
    from tools.analysis_tools.ssm_flops import mmdet_flops, mmseg_flops

    # model = densevmamba_t(pretrained='/mnt/users/Practice/Mamba/VMamba/pretrained_weights/small_83.5.pth')
    
    # cfg = 'configs/ade20k/sfpn.d_vm_tiny_4xb8.py'
    # mmseg_flops(cfg)
    
    # cfg = 'configs/ade20k/upernet.d_vm_basev3_4xb4.py'
    # mmseg_flops(cfg)
    # sleep(5)
    cfg = 'configs/ade20k/mamba/sfpn.d_vm_basev1.py'
    mmseg_flops(cfg)
    