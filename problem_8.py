# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional


@triton.jit
def _flash_attn2_gqa_forward_kernel(
    # --- Pointers ---
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # --- Strides for [B, H, S, D] / [B, H, S] layouts (row-major over D) ---
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h,
    # --- Params ---
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # --- Constexpr tile sizes ---
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Single-kernel causal FlashAttention-2 forward with GQA.
    Stores:
      O: [B, Hq, S, D]  (output)
      M: [B, Hq, S]     (row-wise log-sum-exp in natural log)
    """
    # 1) Block / head indexing
    q_block_idx    = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx      = batch_head_idx // N_Q_HEADS
    q_head_idx     = batch_head_idx %  N_Q_HEADS

    # 2) GQA mapping
    group_size  = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # 3) Online-softmax accumulators (fp32)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 4) Load Q block
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, HEAD_DIM)
    q_ptrs = (
        Q_ptr
        + batch_idx * q_stride_b
        + q_head_idx * q_stride_h
        + (q_offsets[:, None] * q_stride_s + d_offsets[None, :])
    )
    q_mask  = q_offsets[:, None] < SEQ_LEN
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    LOG2E      = 1.44269504
    qk_scale_2 = softmax_scale * LOG2E
    NEG_INF    = -1e9

    # 5) Phase 1: off-diagonal tiles (past K/V)
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (
            K_ptr
            + batch_idx * k_stride_b
            + kv_head_idx * k_stride_h
            + (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        )
        v_ptrs = (
            V_ptr
            + batch_idx * v_stride_b
            + kv_head_idx * v_stride_h
            + (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        )
        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0)
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale_2

        dk         = q_offsets[:, None] - k_offsets[None, :]
        causal_ok  = dk >= 0
        in_bounds  = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)
        valid_mask = in_bounds & causal_ok
        s_ij = tl.where(valid_mask, s_ij, NEG_INF)

        m_ij  = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        acc   = acc * alpha[:, None]
        l_i   = l_i * alpha
        p_ij  = tl.exp2(s_ij - m_new[:, None])
        acc  += tl.dot(p_ij.to(tl.float32), v_block.to(tl.float32))
        l_i  += tl.sum(p_ij, axis=1)
        m_i   = m_new

    # 6) Phase 2: diagonal tiles (triangular mask)
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (
            K_ptr
            + batch_idx * k_stride_b
            + kv_head_idx * k_stride_h
            + (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        )
        v_ptrs = (
            V_ptr
            + batch_idx * v_stride_b
            + kv_head_idx * v_stride_h
            + (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        )
        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0)
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale_2

        dk         = q_offsets[:, None] - k_offsets[None, :]
        causal_ok  = dk >= 0
        in_bounds  = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)
        valid_mask = in_bounds & causal_ok
        s_ij = tl.where(valid_mask, s_ij, NEG_INF)

        m_ij  = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)
        acc   = acc * alpha[:, None]
        l_i   = l_i * alpha
        p_ij  = tl.exp2(s_ij - m_new[:, None])
        acc  += tl.dot(p_ij.to(tl.float32), v_block.to(tl.float32))
        l_i  += tl.sum(p_ij, axis=1)
        m_i   = m_new

    # 7) Normalize and store
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    o_block  = acc / l_i_safe[:, None]

    log2_l = tl.log2(l_i_safe)
    LN2    = 0.6931471805599453
    L_e    = (m_i + log2_l) * LN2  # natural log

    o_ptrs = (
        O_ptr
        + batch_idx * o_stride_b
        + q_head_idx * o_stride_h
        + (q_offsets[:, None] * o_stride_s + d_offsets[None, :])
    )
    tl.store(o_ptrs, o_block.to(O_ptr.dtype.element_ty), mask=q_mask)

    m_ptrs = (
        M_ptr
        + batch_idx * m_stride_b
        + q_head_idx * m_stride_h
        + q_offsets
    )
    tl.store(m_ptrs, L_e.to(tl.float32), mask=(q_offsets < SEQ_LEN))


class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    Forward: Triton kernel.
    Backward: PyTorch math (head-wise) for correctness & clarity.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal: bool = True, softmax_scale: Optional[float] = None):
        assert is_causal, "This kernel only supports causal attention"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must be CUDA"
        assert q.dim() == k.dim() == v.dim() == 4, "Expected [B, H, S, D]"
        B, Hq, S, D = q.shape
        _, Hkv, S_k, D_k = k.shape
        assert (S, D) == (S_k, D_k) and k.shape == v.shape, "K and V must match Q on (S, D) and each other"
        assert Hq % Hkv == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D)

        o = torch.empty_like(q)
        M = torch.empty((B, Hq, S), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(S, BLOCK_M), B * Hq)

        _flash_attn2_gqa_forward_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1),
            softmax_scale,
            S,
            Hq,
            Hkv,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, o)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = Hq
        ctx.num_kv_heads = Hkv
        return o

    @staticmethod
    def backward(ctx, do):
        """
        Backward (PyTorch math), head-by-head to keep memory reasonable:
          dV = P^T @ dO
          dP = dO @ V^T
          dS = (dP - sum(dP*P, -1, keepdim=True)) * P
          dQ = (dS @ K) * scale
          dK = (dS^T @ Q) * scale
        with causal mask and GQA accumulation (sum over Q-heads sharing a KV-head).
        """
        q, k, v, o = ctx.saved_tensors
        B, Hq, S, D = q.shape
        Hkv = ctx.num_kv_heads
        scale = ctx.softmax_scale
        groups = Hq // Hkv
        device = q.device

        # Allocate grads
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)  # accumulate in fp32
        dv = torch.zeros_like(v, dtype=torch.float32)

        # Upper-triangular mask (k > q) for causal attention
        # We'll apply it by setting logits to -inf before softmax.
        # Build once per sequence length on device to avoid per-head allocs.
        # mask_invalid: True where k > q
        tri_mask = torch.triu(torch.ones((S, S), device=device, dtype=torch.bool), diagonal=1)

        # Loop over batch and Q-heads to keep peak memory bounded
        for b in range(B):
            for h in range(Hq):
                kv = h // groups  # mapped KV head

                # Slices
                Qh = q[b, h]            # [S, D]
                Kh = k[b, kv]           # [S, D]
                Vh = v[b, kv]           # [S, D]
                dOh = do[b, h]          # [S, D]

                # Compute logits S = (QK^T) * scale  (fp32 for stability)
                S_logits = (Qh.to(torch.float32) @ Kh.to(torch.float32).T) * scale  # [S, S]

                # Apply causal mask: set strictly-upper-triangular to -inf
                S_logits = S_logits.masked_fill(tri_mask, float("-inf"))

                # Softmax to get probabilities P (fp32)
                P = torch.softmax(S_logits, dim=-1, dtype=torch.float32)  # [S, S]

                # dV = P^T @ dO
                dVh = P.T @ dOh.to(torch.float32)  # [S, D]
                dv[b, kv] += dVh

                # dP = dO @ V^T
                dP = dOh.to(torch.float32) @ Vh.to(torch.float32).T  # [S, S]

                # dS = (dP - <dP,P>_cols) * P
                # where <dP,P>_cols = sum_j dP_ij * P_ij, i.e. rowwise dot
                row_dot = (dP * P).sum(dim=-1, keepdim=True)  # [S, 1]
                dS = (dP - row_dot) * P  # [S, S]

                # Mask dS (not strictly necessary since P=0 on masked positions,
                # but keeps numerics consistent)
                dS = dS.masked_fill(tri_mask, 0.0)

                # dQ = dS @ K * scale
                dQh = (dS @ Kh.to(torch.float32)) * scale  # [S, D]
                dq[b, h] = dQh.to(dq.dtype)

                # dK = dS^T @ Q * scale
                dKh = (dS.T @ Qh.to(torch.float32)) * scale  # [S, D]
                dk[b, kv] += dKh

        # Cast dk/dv back to original dtype
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

        # No gradients for is_causal / softmax_scale (both non-tensor)
        return dq, dk, dv, None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)
