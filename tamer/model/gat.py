import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridGATLayer(nn.Module):
    """Graph Attention layer specialised for grid-structured feature maps.

    The layer first projects the features into multi-head representations,
    performs 8-neighbour message passing (with self-loops) via attention, and
    then applies a position-wise feed-forward network similar to a transformer
    block. Masked (padding) positions are kept at zero.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        ff_multiplier: float = 2.0,
        attn_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )
        if attn_kernel_size % 2 == 0:
            raise ValueError("attn_kernel_size must be an odd number.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.kernel_size = attn_kernel_size
        self.kernel_elems = attn_kernel_size * attn_kernel_size

        self.qkv_proj = nn.Conv2d(d_model, d_model * 3, kernel_size=1, bias=False)
        self.attn_src = nn.Parameter(
            torch.empty(1, num_heads, self.head_dim, 1, 1))
        self.attn_dst = nn.Parameter(
            torch.empty(1, num_heads, self.head_dim, 1, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.out_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        ff_dim = int(math.ceil(d_model * ff_multiplier))
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        for module in self.ff:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply the graph attention layer.

        Parameters
        ----------
        x : torch.Tensor
            [b, h, w, d_model]
        mask : torch.Tensor
            [b, h, w], where True indicates padding that should be ignored.
        """
        residual = x
        attn_out = self._graph_attention(x, mask)
        attn_out = self.out_dropout(attn_out)
        x = residual + attn_out
        x = self._apply_mask(x, mask)
        x = self._apply_norm(self.norm1, x)

        residual = x
        ff_out = self._apply_ff(x)
        x = residual + ff_out
        x = self._apply_mask(x, mask)
        x = self._apply_norm(self.norm2, x)
        return x

    def _graph_attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        # project to q, k, v
        x_ = x.permute(0, 3, 1, 2)  # [b, d, h, w]
        qkv = self.qkv_proj(x_)  # [b, 3d, h, w]
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, h, w)
        k = k.view(b, self.num_heads, self.head_dim, h, w)
        v = v.view(b, self.num_heads, self.head_dim, h, w)

        alpha_src = (q * self.attn_src).sum(dim=2)  # [b, heads, h, w]
        alpha_dst = (k * self.attn_dst).sum(dim=2)  # [b, heads, h, w]

        unfold_kwargs = dict(kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        k_unfold = F.unfold(
            k.view(b, self.num_heads * self.head_dim, h, w), **unfold_kwargs
        ).view(b, self.num_heads, self.head_dim, self.kernel_elems, h * w)
        v_unfold = F.unfold(
            v.view(b, self.num_heads * self.head_dim, h, w), **unfold_kwargs
        ).view(b, self.num_heads, self.head_dim, self.kernel_elems, h * w)
        alpha_dst_unfold = F.unfold(
            alpha_dst.view(b, self.num_heads, h, w), **unfold_kwargs
        ).view(b, self.num_heads, self.kernel_elems, h * w)

        alpha_src_flat = alpha_src.view(b, self.num_heads, 1, h * w)
        scores = self.leaky_relu(alpha_src_flat + alpha_dst_unfold)

        # neighbour mask (True = invalid edge)
        mask_float = mask.float()
        mask_padded = F.pad(
            mask_float.unsqueeze(1), (self.kernel_size // 2,) * 4, value=1.0
        )
        neighbor_mask = F.unfold(
            mask_padded, kernel_size=self.kernel_size
        ).view(b, 1, self.kernel_elems, h * w)
        neighbor_mask = neighbor_mask.bool().expand(-1, self.num_heads, -1, -1)

        scores = scores.masked_fill(neighbor_mask, float("-inf"))
        center_mask = mask.view(b, 1, 1, h * w).expand(-1, self.num_heads, self.kernel_elems, -1)
        scores = torch.where(center_mask, torch.zeros_like(scores), scores)

        attn = torch.softmax(scores, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_dropout(attn)

        out = (attn.unsqueeze(2) * v_unfold).sum(dim=3)
        out = out.view(b, self.num_heads * self.head_dim, h, w)
        out = self.out_proj(out)
        out = out.permute(0, 2, 3, 1)
        return self._apply_mask(out, mask)

    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(mask.unsqueeze(-1), 0.0)

    def _apply_norm(self, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        x = x.view(b * h * w, d)
        x = norm(x)
        return x.view(b, h, w, d)

    def _apply_ff(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        out = self.ff(x.view(b * h * w, d))
        return out.view(b, h, w, d)


class GridGAT(nn.Module):
    """Stacked Grid-GAT layers with shared masking."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        ff_multiplier: float = 2.0,
        attn_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GridGATLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_multiplier=ff_multiplier,
                    attn_kernel_size=attn_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x

