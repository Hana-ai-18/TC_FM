# # """TCNM/utils/masking.py - Attention masks"""
# # import torch

# # class TriangularCausalMask():
# #     def __init__(self, B, L, device="cpu"):
# #         mask_shape = [B, 1, L, L]
# #         with torch.no_grad():
# #             self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)
# #     @property
# #     def mask(self):
# #         return self._mask

# # class ProbMask():
# #     def __init__(self, B, H, L, index, scores, device="cpu"):
# #         _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
# #         _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
# #         batch_idx = torch.arange(B, device=device)[:, None, None]
# #         head_idx = torch.arange(H, device=device)[None, :, None]
# #         indicator = _mask_ex[batch_idx, head_idx, index, :]
# #         self._mask = indicator.view(scores.shape)
# #     @property
# #     def mask(self):
# #         return self._mask

# """TCNM/utils/masking.py - Attention masks for Transformer components"""
# import torch


# class TriangularCausalMask:
#     """
#     Upper-triangular causal mask — prevents attending to future positions.
#     Shape: [B, 1, L, L]  (True = masked / ignored)
#     """
#     def __init__(self, B: int, L: int, device: str = "cpu"):
#         with torch.no_grad():
#             self._mask = torch.triu(
#                 torch.ones([B, 1, L, L], dtype=torch.bool, device=device),
#                 diagonal=1,
#             )

#     @property
#     def mask(self) -> torch.Tensor:
#         return self._mask


# class ProbMask:
#     """
#     ProbSparse attention mask used in Informer-style Transformers.
#     Selects top-k queries and masks the rest.

#     Args:
#         B      : batch size
#         H      : number of heads
#         L      : query length
#         index  : selected query indices  [B, H, top_k]
#         scores : attention score tensor  [B, H, top_k, L_kv]
#         device : target device
#     """
#     def __init__(
#         self,
#         B:      int,
#         H:      int,
#         L:      int,
#         index:  torch.Tensor,
#         scores: torch.Tensor,
#         device: str = "cpu",
#     ):
#         L_kv = scores.shape[-1]

#         _mask    = torch.ones(L, L_kv, dtype=torch.bool, device=device).triu(1)
#         _mask_ex = _mask[None, None, :].expand(B, H, L, L_kv)   # [B, H, L, L_kv]

#         batch_idx = torch.arange(B, device=device)[:, None, None]  # [B, 1, 1]
#         head_idx  = torch.arange(H, device=device)[None, :, None]  # [1, H, 1]
#         indicator = _mask_ex[batch_idx, head_idx, index, :]         # [B, H, top_k, L_kv]

#         self._mask = indicator.view(scores.shape)

#     @property
#     def mask(self) -> torch.Tensor:
#         return self._mask

"""
TCNM/utils/masking.py
=====================
Attention masks for Transformer components.

Classes
-------
TriangularCausalMask  — standard upper-triangular causal mask
ProbMask              — ProbSparse mask for Informer-style attention
"""

from __future__ import annotations
import torch


class TriangularCausalMask:
    """
    Upper-triangular causal mask — prevents attending to future positions.

    Shape : [B, 1, L, L]   (True = masked / ignored by nn.MultiheadAttention)

    Usage
    -----
    mask = TriangularCausalMask(B, L, device=x.device)
    attn_out = attn(q, k, v, attn_mask=mask.mask)
    """

    def __init__(self, B: int, L: int, device: str | torch.device = "cpu"):
        with torch.no_grad():
            self._mask: torch.Tensor = torch.triu(
                torch.ones([B, 1, L, L], dtype=torch.bool, device=device),
                diagonal=1,
            )

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class ProbMask:
    """
    ProbSparse attention mask used in Informer-style Transformers.

    Only the top-k selected queries need masks; the rest are discarded.
    The result matches the shape of the sparse score tensor so it can be
    applied directly as an additive mask.

    Args
    ----
    B      : batch size
    H      : number of heads
    L      : full query length  (before top-k selection)
    index  : selected query indices   shape [B, H, top_k]
    scores : sparse score tensor      shape [B, H, top_k, L_kv]
    device : target device string or torch.device
    """

    def __init__(
        self,
        B:      int,
        H:      int,
        L:      int,
        index:  torch.Tensor,
        scores: torch.Tensor,
        device: str | torch.device = "cpu",
    ):
        L_kv = scores.shape[-1]

        # Full causal mask [L, L_kv]
        _mask    = torch.ones(L, L_kv, dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, L_kv)   # [B, H, L, L_kv]

        # Select only the rows corresponding to the chosen top-k queries
        b_idx = torch.arange(B, device=device)[:, None, None]   # [B, 1, 1]
        h_idx = torch.arange(H, device=device)[None, :, None]   # [1, H, 1]
        indicator = _mask_ex[b_idx, h_idx, index, :]             # [B, H, top_k, L_kv]

        self._mask: torch.Tensor = indicator.view(scores.shape)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask