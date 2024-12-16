import math
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from collections import OrderedDict
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mmdet.registry import MODELS

import selective_scan_cuda_oflex

try:
    from csm_triton import CrossScanTriton, CrossMergeTriton
except:
    from .csm_triton import CrossScanTriton, CrossMergeTriton



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
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()
        

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_groups=1, num_channels=num_channels, **kwargs)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)     
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


# =====================================================
class SparXSS2D(nn.Module):

    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=1,
        dt_rank="auto",
        norm_layer=LayerNorm2d,
        act_layer=nn.SiLU,
        d_conv=3,
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        stage_dense_idx=None,
        layer_idx=None,
        max_dense_depth=None,
        dense_step=None,
        is_cross_layer=None,
        dense_layer_idx=None,
        sr_ratio=1,
        cross_stage=False,   
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        d_inner = int(d_model * ssm_ratio)
        
        self.layer_idx = layer_idx
        self.d_inner = d_inner
        
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        k_group = 4

        # in proj =======================================
        self.in_proj = nn.Conv2d(d_model, d_inner, kernel_size=1)
        self.act = act_layer()
        
        self.is_cross_layer = is_cross_layer
        
        if is_cross_layer:
            if dense_layer_idx == stage_dense_idx[0]:
                intra_dim = d_model
                inner_dim = d_model * dense_layer_idx
            else:
                count = 1 if cross_stage else 0
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
                nn.Conv2d(current_d_dim, d_model*2, kernel_size=1),
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
            
            
            # Learnable scale for more stable training in large-resolution tasks
            init_scale = (((800 * 1333) / 16) / 64) ** -0.5
            self.scale = nn.Parameter(torch.ones(1) * init_scale, requires_grad=True)
            
            self.in_proj = nn.Sequential(
                nn.Conv2d(d_model*3, d_model*3, kernel_size=3, padding=1, groups=d_model*3),
                nn.GELU(),
                nn.Conv2d(d_model*3, int(d_inner//3*3), kernel_size=1, groups=3),     
            )

        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=(d_conv-1)//2)
            
        # out proj =======================================
        self.out_norm = norm_layer(d_inner)
        self.out_proj = nn.Conv2d(d_inner, d_model, kernel_size=1)

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
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))    
            
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
        return CrossScanTriton.apply(x)
    
    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)

    def forward_ssm(self, x, to_dtype=False, force_fp32=False):

        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
  
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        
        xs = self._cross_scan(x)
        
        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight, bias=None, groups=K)
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
            xs = xs.to(torch.float32)
            dts = dts.to(torch.float32)
            Bs = Bs.to(torch.float32)
            Cs = Cs.to(torch.float32)
                  
        ys = self._selective_scan(xs, dts, As, Bs, 
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)

        y = self._cross_merge(ys.reshape(B, K, -1, H, W))
        y = self.out_norm(y.reshape(B, -1, H, W))
        
        if to_dtype:
            y = y.to(x.dtype)

        return y
    

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
            
            if self.training:
                attn = q @ k.transpose(-1, -2) * self.scale
                attn = F.softmax(attn, dim=-1)
                attn = (attn @ v).reshape(B, -1, H, W)
            else:
                attn = F.scaled_dot_product_attention(q, k, v, scale=float(self.scale)).reshape(B, -1, H, W)
            
            if (H, W) == (sr_H, sr_W):
                x = q.reshape(B, -1, H, W)
                
            x = torch.cat([x, attn, v.reshape(B, -1, H, W)], dim=1)
            x = rearrange(x, 'b (c g) h w -> b (g c) h w', g=3).contiguous() ### channel shuffle for efficiency
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
            
        if self.d_conv > 1:
            x = self.conv2d(x)
            
        x = self.act(x)
        x = self.forward_ssm(x)   
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x


class SparXVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=0,
        drop_path=0,
        norm_layer=LayerNorm2d,
        ssm_d_state=16,
        ssm_ratio=1,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_drop_rate=0,
        ssm_init="v0",
        mlp_ratio=4,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0,
        use_checkpoint=False,
        post_norm=False,
        layer_idx=None,
        stage_dense_idx=None,
        max_dense_depth=None,
        dense_step=None,
        is_cross_layer=None,
        dense_layer_idx=None,
        sr_ratio=1,
        cross_stage=False,
        ls_init_value=1e-5,
        **kwargs,
    ):

        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.pos_embed = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)

        self.norm = norm_layer(hidden_dim)
        self.op = SparXSS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            norm_layer=norm_layer,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            layer_idx=layer_idx,
            stage_dense_idx=stage_dense_idx,
            max_dense_depth=max_dense_depth,
            dense_step=dense_step,
            is_cross_layer=is_cross_layer,
            dense_layer_idx=dense_layer_idx,
            sr_ratio=sr_ratio,
            cross_stage=cross_stage,
            **kwargs,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
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
        x = self.layerscale_1(x) + self.drop_path(self.op(self.norm(x), shortcut)) # Token Mixer
        x = self.layerscale_2(x) + self.drop_path(self.mlp(self.norm2(x))) # FFN
        shortcut = x

        return (x, shortcut)

    def forward(self, x):

        input, shortcut = x
        
        if isinstance(shortcut, (list, tuple)):
            shortcut = torch.cat(shortcut, dim=1)
        
        if self.use_checkpoint and input.requires_grad:
            return checkpoint.checkpoint(self._forward, input, shortcut, use_reentrant=True)
        else:
            return self._forward(input, shortcut)


class SparXMamba(nn.Module):
    def __init__(
        self,
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 5, 2], 
        dims=[96, 192, 384, 768],
        ssm_d_state=1,
        ssm_ratio=[2, 2, 2, 2],
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,        
        ssm_conv=5,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        mlp_ratio=[4, 4, 4, 4],
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        ls_init_value=[None, None, 1, 1],
        stem_type='v1',
        drop_path_rate=0,
        norm_layer=GroupNorm,
        use_checkpoint=[0, 0, 0, 0],
        dense_config=None,
        sr_ratio=[8, 4, 2, 1],
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
        self.cross_stage = dense_config['cross_stage']
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0]//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]//2),
            nn.GELU(),
            nn.Conv2d(dims[0]//2, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0])
        )
        
        if stem_type == 'v2':
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_chans, dims[0]//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]//2),
                nn.GELU(),        
                nn.Conv2d(dims[0]//2, dims[0]//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]//2),
                nn.GELU(),
                nn.Conv2d(dims[0]//2, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0])
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
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio[i_layer],
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                mlp_ratio=mlp_ratio[i_layer],
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                ls_init_value=ls_init_value[i_layer],
                max_dense_depth=dense_config["max_dense_depth"][i_layer],
                dense_step=dense_config["dense_step"][i_layer],
                dense_idx=dense_config["dense_idx"][i_layer],
                sr_ratio=sr_ratio[i_layer],
                cross_stage=self.cross_stage[i_layer],
                **kwargs,
            ))
            
            self.norm.append(LayerNorm2d(self.dims[i_layer], eps=1e-6))
        
        self.apply(self._init_weights)
        
        if torch.distributed.is_initialized():
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0, 0], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        ssm_d_state=16,
        ssm_ratio=1,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        ls_init_value=None,
        max_dense_depth=None,
        dense_step=None,
        dense_idx=None,
        sr_ratio=1,
        cross_stage=False,
        **kwargs,
    ):

        depth = len(drop_path)

        blocks = []
        for d in range(depth):

            if d in dense_idx:
                is_cross_layer = True
                dense_layer_idx = d
            else:
                is_cross_layer = False
                dense_layer_idx = None
            
            blocks.append(SparXVSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                ls_init_value=ls_init_value,
                use_checkpoint=(d<use_checkpoint),
                layer_idx=d,
                max_dense_depth=max_dense_depth,
                dense_step=dense_step,
                is_cross_layer=is_cross_layer,
                stage_dense_idx=dense_idx,
                dense_layer_idx=dense_layer_idx,
                sr_ratio=sr_ratio,
                cross_stage=cross_stage,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    
    def layer_forward(self, layers, x, s, d_cfg):
        
        dense_step, dense_start, max_dense_depth, dense_idx, cross_stage = d_cfg

        inner_list = []
        cross_list = []
        
        if s is not None:
            if cross_stage: 
                cross_list.append(s)
            else: 
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

        return x
    
    
    def forward(self, x):
        
        outs = []
        d_cfg = self.dense_config
        x = self.patch_embed(x)
        s = None
        
        for idx, layer in enumerate(self.layers):
            
            dense_step = d_cfg['dense_step'][idx]
            dense_start = d_cfg['dense_start'][idx]
            max_dense_depth = d_cfg['max_dense_depth'][idx]
            dense_idx = d_cfg['dense_idx'][idx]
            cross_stage = d_cfg['cross_stage'][idx]
            _d_cfg = (dense_step, dense_start, max_dense_depth, dense_idx, cross_stage)
            x = self.layer_forward(layer, x, s, _d_cfg)
            outs.append(self.norm[idx](x))
            x = layer.downsample(x)
            s = x
    
        return tuple(outs)


@MODELS.register_module()
def sparx_mamba_t(pretrained=False, **kwargs):
    
    dense_config = {
        'dense_step': [1, 1, 2, 1],
        'dense_start': [100, 1, 0, 0],
        'max_dense_depth': [100, 100, 3, 100],
        'cross_stage': [False, False, False, False],
    }
    
    model =  SparXMamba(depths=[2, 2, 7, 2],
                        dims=[96, 192, 320, 512],
                        sr_ratio=[8, 4, 2, 1],
                        dense_config=dense_config,
                        **kwargs)
    if pretrained:
        pretrained = 'https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_tiny_in1k.pth'
        state_dict = torch.hub.load_state_dict_from_url(url=pretrained, map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict, strict=False)
    
    return model


@MODELS.register_module()
def sparx_mamba_s(pretrained=False, **kwargs):
    
    dense_config = {
        'dense_step': [1, 1, 3, 1],
        'dense_start': [100, 1, 0, 0],
        'max_dense_depth': [100, 100, 3, 100],
        'cross_stage': [False, False, False, False],
    }
    
    model =  SparXMamba(depths=[2, 2, 17, 2],
                        dims=[96, 192, 328, 544],
                        sr_ratio=[8, 4, 2, 1],
                        stem_type='v2',
                        dense_config=dense_config,
                        **kwargs)
    
    if pretrained:
        pretrained = 'https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_small_in1k.pth'
        state_dict = torch.hub.load_state_dict_from_url(url=pretrained, map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict, strict=False)
    
    return model


@MODELS.register_module()
def sparx_mamba_b(pretrained=False, **kwargs):
    
    dense_config = {
        'dense_step': [1, 1, 3, 2],
        'dense_start': [100, 1, 0, 2],
        'max_dense_depth': [100, 100, 3, 100],
        'cross_stage': [False, False, False, False],
    }
    
    model =  SparXMamba(depths=[2, 2, 21, 3],
                        dims=[120, 240, 396, 636],
                        sr_ratio=[8, 4, 2, 1],
                        stem_type='v2',
                        dense_config=dense_config,
                        **kwargs)
    
    if pretrained:
        pretrained = 'https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_base_in1k.pth'
        state_dict = torch.hub.load_state_dict_from_url(url=pretrained, map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict, strict=False)
    
    return model