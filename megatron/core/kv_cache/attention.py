from .cache_utils import ChunkedTensor
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from megatron.core import parallel_state
from megatron.core.context_parallel.dattention import flip_cp, flip_cp_, slice_cp
from megatron.core.cudnn_attn import cudnn_attn_check_capability, CudnnAttnFunc
from megatron.core.tensor_parallel.random import get_during_recomputing
from typing import Tuple

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist


# q, k, v, out: [s, b, a, d]
# softmax_lse:  [s, b, a, 1]
def attn_forward(q, k, v, causal):
    q, k, v = [x.transpose(0, 1) for x in [q, k, v]]
    if cudnn_attn_check_capability(use_causal_mask_bottom_right=True):
        out = torch.empty_like(q)
        softmax_lse = CudnnAttnFunc.forward_no_ctx(q, k, v, causal, False, False, None, None, None, None, out)
    else:
        softmax_scale = q.shape[-1] ** (-0.5)
        out, q_padded, k_padded, v_padded, out_padded, softmax_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            0,
            softmax_scale,
            causal,
            False, 0, 0, 0
        )
        softmax_lse = softmax_lse.unsqueeze(-1)
        assert q.shape == q_padded.shape and \
            k.shape == k_padded.shape and \
            v.shape == v_padded.shape and \
            out.shape == out_padded.shape, "padding is not supported."
    out = out.transpose(0, 1)
    softmax_lse = softmax_lse.permute(2, 0, 1, 3)
    return out, softmax_lse


def attn_backward(dout, q, k, v, out, softmax_lse, dk, dv, causal):
    dq = torch.empty_like(q)
    dout, q, k, v, out, dq, dk, dv = [x.transpose(0, 1) for x in [dout, q, k, v, out, dq, dk, dv]]
    softmax_lse = softmax_lse.permute(1, 2, 0, 3)
    if cudnn_attn_check_capability(use_causal_mask_bottom_right=True):
        CudnnAttnFunc.backward_no_ctx(dout, q, k, v, out, softmax_lse, causal, False, None, None, None, None, dq, dk, dv)
    else:
        softmax_lse = softmax_lse.squeeze(-1)
        softmax_scale = q.shape[-1] ** (-0.5)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            0,
            softmax_scale,
            0, 0, 0,
            causal,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
    dq = dq.transpose(0, 1)
    return dq


_CHUNK_STREAM = None


def get_chunk_stream():
    global _CHUNK_STREAM
    if _CHUNK_STREAM is None:
        _CHUNK_STREAM = torch.cuda.Stream()
    return _CHUNK_STREAM

torch._dynamo.config.cache_size_limit = 100

@torch.compile(fullgraph=True)
def update_out_lse(out: torch.Tensor, out_: torch.Tensor, lse: torch.Tensor, lse_: torch.Tensor) -> None:
    # delta_exp = (lse_ - lse).exp()
    # lse += delta_exp.log1p()
    # out += out_ * delta_exp
    # out /= (1 + delta_exp)
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out -= F.sigmoid(lse_ - lse) * (out - out_)
    lse -= F.logsigmoid(lse - lse_)


@torch.compile(fullgraph=True)
def log_scale_flip(out: torch.Tensor, lse: torch.Tensor, se_all: torch.Tensor, CP: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lse_all = se_all.log()
    out = (out * torch.exp(lse - lse_all)).to(out.dtype)
    return flip_cp(out, 0, CP), lse_all


@torch.compile(fullgraph=True)
def sum_scale_flip(out: torch.Tensor, lse: torch.Tensor, lse_all: torch.Tensor, CP: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lse_all = torch.logsumexp(lse_all, dim=0)
    out = (out * torch.exp(lse - lse_all)).to(out.dtype)
    return flip_cp(out, 0, CP), lse_all


# There are two methods to merge the `lse`:
#   1. Merge via `all_reduce`. It must scales the exponentials down to a relatively small range.
#   2. Merge via `all_gather`. It sums the gathered `lse`, which is numerically stabilized.
def merge_out_lse(out: torch.Tensor, lse: torch.Tensor, scale, cp_group):
    """
    Merge out and lse in the CP group.
    If scale is not None, it will be applied to the `lse` and `all_reduce` is used to merge `lse`.
    Otherwise, `all_gather` and local `logsumexp` are used to merge `lse`.
    """
    CP = dist.get_world_size(cp_group)
    rank = dist.get_rank(cp_group)
    if scale:   # scale down, then all_reduce.
        shift = math.log(scale)
        lse -= shift
        se = lse.exp()
        dist.all_reduce(se.as_strided([se.numel()], [1]), group=cp_group)
        out, lse = log_scale_flip(out, lse, se, CP)
    else:   # all_gather, then sum.
        lse_all = lse.new_empty((CP, *lse.shape))
        dist.all_gather_into_tensor(lse_all, lse.contiguous(), group=cp_group)
        out, lse = sum_scale_flip(out, lse, lse_all, CP)
    oi = out.new_empty(out.shape[0] // CP, *out.shape[1:])
    req = dist.reduce_scatter_tensor(oi, out, group=cp_group, async_op=True)
    lsei = slice_cp(lse, 0, CP, rank)
    if scale:
        lsei += shift
    return oi, lsei, req


def a2a_out_lse(out: torch.Tensor, lse: torch.Tensor, cp_group):
    CP = dist.get_world_size(cp_group)
    lse = lse.contiguous()
    flip_cp_(lse, 0, CP)
    lse_t = torch.empty_like(lse)
    req = dist.all_to_all_single(lse_t, lse, group=cp_group, async_op=True)
    flip_cp_(out, 0, CP)
    out_t = torch.empty_like(out)
    req = dist.all_to_all_single(out_t, out, group=cp_group, async_op=True)
    return out_t, lse_t, req


@torch.compile(fullgraph=True)
def sum_out_lse(out_t: torch.Tensor, lse_t: torch.Tensor, world_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if world_size == 1:
        return out_t, lse_t
    lse_t = lse_t.unflatten(0, (world_size, -1))
    out_t = out_t.unflatten(0, (world_size, -1))
    lsei = torch.logsumexp(lse_t, dim=0)
    oi = torch.sum(out_t * torch.exp(lse_t - lsei), dim=0).type_as(out_t)
    return oi, lsei


DEBUG_PAIR = False


def pair_print_debug(msg, *args):
    if DEBUG_PAIR:
        rank = parallel_state.get_pipeline_model_parallel_rank()
        print(f"rank{rank}: {msg}: {args}", flush=True)


def attn_forward_pair(ctx_pair, q, cp_group=None):
    """
    Offload the attention forward calculation to another pipeline rank.
    This generator runs the following steps:
        1. Send the query and portions of key-value to the peer.
        2. Calculate the attention forward on the peer.
        3. Send the output and softmax_lse back.
    """
    CP = dist.get_world_size(cp_group) if cp_group else 1

    n = (ctx_pair.nbwd if get_during_recomputing() else ctx_pair.nfwd) if ctx_pair else 0
    pair_print_debug('forward pair+', ctx_pair)
    if n:
        p = ctx_pair.peer
        group = parallel_state.get_pipeline_model_parallel_group()
        # group = torch.distributed.group.WORLD
        peer = parallel_state.get_pipeline_model_parallel_global_rank(p)
        # assert all(req.wait() for req in ctx_pair.reqs.pop())
        reqs = ctx_pair.reqs.pop()
        k_other, v_other = ctx_pair.other_kv.pop()

    # pre-attn pair
    if n < 0:
        assert k_other == n and v_other == n
        sends = [dist.P2POp(dist.isend, q, peer, group)]
        pair_print_debug('q send+', q.shape)
        reqs += dist.batch_isend_irecv(sends); del sends
    elif n > 0:
        q_other = torch.empty_like(q)
        recvs = [dist.P2POp(dist.irecv, q_other, peer, group)]
        pair_print_debug('q recv+', q.shape)
        reqs += dist.batch_isend_irecv(recvs); del recvs

    # yield to local attn
    pair_print_debug('attn+')
    # need shapes of out and lse, because they may be incontiguous.
    out, lse = yield n
    pair_print_debug('attn-')

    # post-attn pair
    if n > 0:
        if cp_group:
            assert all(req.wait() for req in reqs)
            pair_print_debug('q recv-')
            qi = q_other
            q_other = qi.new_empty(CP * qi.shape[0], *qi.shape[1:])
            pair_print_debug('q ag+', q_other.shape)
            reqs = [dist.all_gather_into_tensor(q_other, qi, group=cp_group, async_op=True)]
            del qi

    yield

    if n < 0:
        assert all(req.wait() for req in reqs)
        pair_print_debug('q send-')
        if cp_group:
            out_recv = out.new_empty(out.shape[0] // CP, *out.shape[1:])
            lse_recv = lse.new_empty(lse.shape[0] // CP, *lse.shape[1:])
        else:
            # out and lse may be incontiguous, and their strides are kept as is.
            out_recv = torch.empty_like(out)
            lse_recv = torch.empty_like(lse)
        recv_out = dist.P2POp(dist.irecv, out_recv, peer, group)
        recv_lse = dist.P2POp(dist.irecv, lse_recv, peer, group)
        pair_print_debug('out recv+', out_recv.shape)
        reqs = dist.batch_isend_irecv([recv_out, recv_lse])

    yield

    # merge local out in CP
    if cp_group:
        # out, lse, cp_req = merge_out_lse(out, lse, None, cp_group)
        out, lse, cp_req = a2a_out_lse(out, lse, cp_group)

    # post-attn calc
    if n < 0:
        assert all(req.wait() for req in reqs)
        pair_print_debug('out recv-')
        if cp_group:
            cp_req.wait()
            out, lse = sum_out_lse(out, lse, CP)
        update_out_lse(out, out_recv, lse, lse_recv)
    elif n > 0:
        assert all(req.wait() for req in reqs)
        pair_print_debug('q ag-' if cp_group else 'q recv-')
        if cp_group:
            flip_cp_(q_other, 0, CP)
        out_other, lse_other = attn_forward(q_other, k_other, v_other, False)
        del q_other, k_other, v_other
        if cp_group:
            cp_req.wait()
            # merge remote out in CP
            # out_other, lse_other, cp_req = merge_out_lse(out_other, lse_other, None, cp_group)
            out_other, lse_other, cp_req = a2a_out_lse(out_other, lse_other, cp_group)
            out, lse = sum_out_lse(out, lse, CP)
            cp_req.wait()
            out_other, lse_other = sum_out_lse(out_other, lse_other, CP)
        send_out = dist.P2POp(dist.isend, out_other, peer, group)
        send_lse = dist.P2POp(dist.isend, lse_other, peer, group)
        pair_print_debug('out send+', out_other.shape)
        reqs = dist.batch_isend_irecv([send_out, send_lse])
        assert all(req.wait() for req in reqs)
        pair_print_debug('out send-')
    else:
        if cp_group:
            cp_req.wait()
            out, lse = sum_out_lse(out, lse, CP)
    pair_print_debug('forward pair-', ctx_pair)
    yield out, lse


def attn_backward_pair(ctx_pair, dout, cp_group=None):
    """
    Offload the attention backward calculation to another pipeline rank.
    This generator runs the following steps:
        1. Send the dout, query, out, lse and portions of key-value to the peer.
        2. Calculate the attention backward on the peer.
        3. Send the gradients of query and key-value back.
    """
    CP = dist.get_world_size(cp_group) if cp_group else 1

    p, n = (ctx_pair.peer, ctx_pair.nbwd) if ctx_pair else (None, 0)
    pair_print_debug('backward pair+', ctx_pair)
    if n:
        group = parallel_state.get_pipeline_model_parallel_group()
        # group = torch.distributed.group.WORLD
        peer = parallel_state.get_pipeline_model_parallel_global_rank(p)
        # assert all(req.wait() for req in ctx_pair.reqs.pop())
        reqs = ctx_pair.reqs.pop()
        k_other, v_other = ctx_pair.other_kv.pop()
        q_other, out_other, lse_other = ctx_pair.other_qo.pop()

    # pre-attn pair
    if n < 0:
        assert k_other == n and v_other == n
        sends = [dist.P2POp(dist.isend, dout, peer, group)]
        pair_print_debug('dout send+', dout.shape)
        reqs += dist.batch_isend_irecv(sends); del sends
    elif n > 0:
        dout_other = torch.empty_like(dout)
        recvs = [dist.P2POp(dist.irecv, dout_other, peer, group)]
        pair_print_debug('dout recv+', dout.shape)
        reqs += dist.batch_isend_irecv(recvs); del recvs

    # yield to local attn
    pair_print_debug('attn+')
    dq, dk, dv = yield n
    pair_print_debug('attn-')

    # post-attn pair
    if n > 0:
        if cp_group:
            assert all(req.wait() for req in reqs)
            pair_print_debug('dout recv-')
            def get_other(t):
                t_other = t.new_empty(CP * t.shape[0], *t.shape[1:])
                return t, t_other
            qi, q_other = get_other(q_other)
            oi, out_other = get_other(out_other)
            lsei, lse_other = get_other(lse_other)
            doi, dout_other = get_other(dout_other)
            pair_print_debug('dout ag+', dout_other.shape)
            with dist._coalescing_manager(cp_group, async_ops=True) as cm:
                dist.all_gather_into_tensor(q_other, qi, group=cp_group, async_op=True)
                dist.all_gather_into_tensor(out_other, oi, group=cp_group, async_op=True)
                dist.all_gather_into_tensor(lse_other, lsei, group=cp_group, async_op=True)
                dist.all_gather_into_tensor(dout_other, doi, group=cp_group, async_op=True)
            reqs = cm.works
            del qi, oi, lsei, doi

    yield

    if n < 0:
        assert all(req.wait() for req in reqs)
        pair_print_debug('dout send-')
        dq_recv = torch.empty_like(dout)
        recv_dq = dist.P2POp(dist.irecv, dq_recv, peer, group)
        recv_dk = dist.P2POp(dist.irecv, dk, peer, group)
        recv_dv = dist.P2POp(dist.irecv, dv, peer, group)
        pair_print_debug('dqkv recv+', dq_recv.shape, dk.shape, dv.shape)
        reqs = dist.batch_isend_irecv([recv_dq, recv_dk, recv_dv])

    yield

    # merge local dq in CP
    if cp_group:
        flip_cp_(dq, 0, CP)
        dqi = dout   # reuse dout memory
        cp_req = dist.reduce_scatter_tensor(dqi, dq, group=cp_group, async_op=True)
        dq = dqi

    # post-attn pair
    if n < 0:
        assert all(req.wait() for req in reqs)
        pair_print_debug('dqkv recv-')
        if cp_group:
            cp_req.wait()
        dq += dq_recv
    elif n > 0:
        assert all(req.wait() for req in reqs)
        pair_print_debug('dout ag-' if cp_group else 'dout recv-')
        if cp_group:
            flip_cp_(q_other, 0, CP)
            flip_cp_(out_other, 0, CP)
            flip_cp_(lse_other, 0, CP)
            flip_cp_(dout_other, 0, CP)
        dk_other = torch.empty_like(k_other)
        dv_other = torch.empty_like(v_other)
        dq_other = attn_backward(dout_other, q_other, k_other, v_other, out_other, lse_other, dk_other, dv_other, False)
        del dout_other, q_other, k_other, v_other, out_other, lse_other
        if cp_group:
            flip_cp_(dq_other, 0, CP)
            cp_req.wait()
            dqi = dq_other.new_empty(dq_other.shape[0] // CP, *dq_other.shape[1:])
            dist.reduce_scatter_tensor(dqi, dq_other, group=cp_group)
            dq_other = dqi
        send_dq = dist.P2POp(dist.isend, dq_other, peer, group)
        send_dk = dist.P2POp(dist.isend, dk_other, peer, group)
        send_dv = dist.P2POp(dist.isend, dv_other, peer, group)
        pair_print_debug('dqkv send+', dq_other.shape, dk_other.shape, dv_other.shape)
        reqs = dist.batch_isend_irecv([send_dq, send_dk, send_dv])
        assert all(req.wait() for req in reqs)
        pair_print_debug('dqkv send-')

    if cp_group:
        cp_req.wait()
    pair_print_debug('backward pair-', ctx_pair)
    yield dq


class ChunkedAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, ctx_pair):
        assert isinstance(k, ChunkedTensor) and isinstance(v, ChunkedTensor)
        ctx.kv = k, v
        ctx.causal = causal
        ctx.ctx_pair = ctx_pair
        event = None
        chunk_stream = get_chunk_stream()
        chunk_stream.wait_stream(torch.cuda.current_stream())
        pair_gen = attn_forward_pair(ctx_pair, q)
        m = max(0, -next(pair_gen))  # the number of sent kv
        k_tensors = k._tensors[m:][::-1]
        v_tensors = v._tensors[m:][::-1]
        n = len(k_tensors)
        # calculate the local attn.
        for i in range(n):
            last_kv = i == 0
            stream = (torch.cuda.current_stream(), chunk_stream)[i % 2]
            with torch.cuda.stream(stream):
                out_, lse_ = attn_forward(q, k_tensors[i], v_tensors[i], causal and last_kv)
                if i == 0:
                    out, lse = out_, lse_
                else:
                    event.wait()
                    update_out_lse(out, out_, lse, lse_)
                event = stream.record_event(event)
                out_, lse_ = None, None
                if i == 0:
                    pair_gen.send((out, lse))
                if n == 1 or i + 2 == n:
                    next(pair_gen)
        event.wait()
        # calculate or receive the remote attn.
        out, lse = next(pair_gen)

        ctx.save_for_backward(q, out, lse)
        # if gradient is not required, qo will not be exchanged for backward.
        if ctx_pair and ctx_pair.nbwd and q.requires_grad:
            ctx_pair.local_qo.append([q, out, lse])
        return out

    @staticmethod
    def backward(ctx, dout):
        q, out, lse = ctx.saved_tensors
        k, v = ctx.kv; del ctx.kv
        causal = ctx.causal
        ctx_pair = ctx.ctx_pair
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        event = None
        seqlen = q.shape[0]
        offset = k.shape[0]
        pair_gen = attn_backward_pair(ctx_pair, dout)
        chunk_stream = get_chunk_stream()
        chunk_stream.wait_stream(torch.cuda.current_stream())
        m = max(0, -next(pair_gen))  # the number of sent kv
        k_tensors = k._tensors[m:][::-1]
        v_tensors = v._tensors[m:][::-1]
        n = len(k_tensors)
        for i in range(n):
            last_kv = i == 0
            stream = (torch.cuda.current_stream(), chunk_stream)[i % 2]
            with torch.cuda.stream(stream):
                offset -= seqlen
                dk_, dv_ = dk[offset:offset + seqlen], dv[offset:offset + seqlen]
                dq_ = attn_backward(dout, q, k_tensors[i], v_tensors[i], out, lse, dk_, dv_, causal and last_kv)
                if i == 0:
                    dq = dq_
                else:
                    event.wait()
                    dq += dq_
                event = stream.record_event(event)
                dq_ = None
                if i == 0:
                    pair_gen.send((dq, dk[:m * seqlen], dv[:m * seqlen]))
                if n == 1 or i + 2 == n:
                    next(pair_gen)
        event.wait()
        # calculate or receive the remote dq, dk, dv.
        dq = next(pair_gen)
        return dq, dk, dv, None, None


class ConcatAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        ctx.causal = causal
        k_ = k.concat() if isinstance(k, ChunkedTensor) else k
        v_ = v.concat() if isinstance(v, ChunkedTensor) else v
        out, softmax_lse = attn_forward(q, k_, v_, ctx.causal)
        ctx.save_for_backward(q, out, softmax_lse)
        ctx.kv = k, v
        return out

    @staticmethod
    def backward(ctx, dout):
        q, out, softmax_lse = ctx.saved_tensors
        k, v = ctx.kv; del ctx.kv
        k_ = k.concat() if isinstance(k, ChunkedTensor) else k
        v_ = v.concat() if isinstance(v, ChunkedTensor) else v
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        dq = attn_backward(dout, q, k_, v_, out, softmax_lse, dk, dv, ctx.causal)
        return dq, dk, dv, None


def cache_aware_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, ctx_pair=None):
    assert not dropout_p and not softmax_scale
    # TODO(lizhouyang): warm up for cudnn sdpa with ctx_pair.
    CudnnAttnFunc._ALLOW_GRAPH_CREATION = True
    if isinstance(k, ChunkedTensor) and isinstance(v, ChunkedTensor):
        return ChunkedAttnFunc.apply(q, k, v, causal, ctx_pair)
    else:
        return ConcatAttnFunc.apply(q, k, v, causal)


############################################
# Context Parallelism passing Query and Out
############################################

class CPQOAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qi, k, v, cp_group, ctx_pair):
        ctx.kv = k, v
        ctx.cp_group = cp_group
        ctx.ctx_pair = ctx_pair
        CP = dist.get_world_size(ctx.cp_group)
        rank = dist.get_rank(ctx.cp_group)
        q = qi.new_empty(CP * qi.shape[0], *qi.shape[1:])
        q._handle = dist.all_gather_into_tensor(q, qi, group=ctx.cp_group, async_op=True)
        cp_stream = torch.cuda.current_stream() # get_chunk_stream()
        pair_gen = attn_forward_pair(ctx_pair, qi, cp_group)
        m = max(0, -next(pair_gen)) # the number of sent kv
        k_tensors = getattr(k, "_tensors", [k])
        v_tensors = getattr(v, "_tensors", [v])
        ctx.concat_kv = True
        if ctx.concat_kv:
            k_chunks = [k[m:-1].concat()] if len(k_tensors) > 1 else []
            v_chunks = [v[m:-1].concat()] if len(v_tensors) > 1 else []
        else:
            k_chunks = k_tensors[m:-1]
            v_chunks = v_tensors[m:-1]
        k_chunks += list(k_tensors[-1].chunk(2))
        v_chunks += list(v_tensors[-1].chunk(2))
        n = len(k_chunks)
        offset_q = [0] * (n - 2) + [q.shape[0] * rank // CP, q.shape[0] * (2 * CP - 2 * rank - 1) // (2 * CP)]
        if rank * 2 >= CP:
            k_chunks[-1], k_chunks[-2] = k_chunks[-2], k_chunks[-1]
            v_chunks[-1], v_chunks[-2] = v_chunks[-2], v_chunks[-1]
            offset_q[-1], offset_q[-2] = offset_q[-2], offset_q[-1]
        if n == 2:
            out = torch.zeros_like(q)
            lse = q.new_full(q.shape[:3] + (1,), -16, dtype=torch.float)
            event = torch.cuda.current_stream().record_event()
        else:
            out = None
            lse = None
            event = None
        q._handle.wait(); del q._handle
        flip_cp_(q, 0, CP)
        cp_stream.wait_stream(torch.cuda.current_stream())
        for i in range(n):
            causal = i + 2 >= n
            stream = (torch.cuda.current_stream(), cp_stream)[i % 2]
            with torch.cuda.stream(stream):
                out_, lse_ = attn_forward(q[offset_q[i]:], k_chunks[i], v_chunks[i], -causal)
                if out is None:
                    out = out_
                    lse = lse_
                else:
                    event.wait()
                    update_out_lse(out[offset_q[i]:], out_, lse[offset_q[i]:], lse_)
                event = stream.record_event(event)
                out_ = None
                lse_ = None
            if i == 0:
                pair_gen.send((out, lse))
            if (n == 2 and i == 0) or i + 3 == n:
                next(pair_gen)
        event.wait()
        del k_chunks, v_chunks
        # calculate or receive the remote attn.
        oi, lsei = next(pair_gen)

        # if gradient is not required, qo will not be exchanged for backward.
        if ctx_pair and ctx_pair.nbwd and qi.requires_grad:
            ctx_pair.local_qo.append([qi, oi, lsei])
        q = qi.new_empty(()).expand_as(q)
        q._shard = qi
        out = oi.new_empty(()).expand_as(out)
        out._shard = oi
        lse = lsei.new_empty(()).expand_as(lse)
        lse._shard = lsei
        return oi, q, out, lse

    @staticmethod
    def backward(ctx, doi, q, out, lse):
        k, v = ctx.kv; del ctx.kv
        ctx_pair = ctx.ctx_pair
        CP = dist.get_world_size(ctx.cp_group)
        rank = dist.get_rank(ctx.cp_group)
        dout = doi.new_empty(CP * doi.shape[0], *doi.shape[1:])
        dout._handle = dist.all_gather_into_tensor(dout, doi, group=ctx.cp_group, async_op=True)
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        cp_stream = torch.cuda.current_stream() # get_chunk_stream()
        pair_gen = attn_backward_pair(ctx_pair, doi, ctx.cp_group)
        m = max(0, -next(pair_gen)) # the number of sent kv
        mlength = m * k._tensors[0].shape[0]
        dk_ = dk[mlength:]
        dv_ = dv[mlength:]
        k_tensors = getattr(k, "_tensors", [k])
        v_tensors = getattr(v, "_tensors", [v])
        if ctx.concat_kv:
            k_chunks = [k[m:-1].concat()] if len(k_tensors) > 1 else []
            v_chunks = [v[m:-1].concat()] if len(v_tensors) > 1 else []
        else:
            k_chunks = k_tensors[m:-1]
            v_chunks = v_tensors[m:-1]
        k_chunks += list(k_tensors[-1].chunk(2))
        v_chunks += list(v_tensors[-1].chunk(2))
        del k, v, k_tensors, v_tensors
        q._handle.wait(); del q._handle
        flip_cp_(q, 0, CP)
        out._handle.wait(); del out._handle
        flip_cp_(out, 0, CP)
        lse._handle.wait(); del lse._handle
        flip_cp_(lse, 0, CP)
        n = len(k_chunks)
        offset_q = [0] * (n - 2) + [q.shape[0] * rank // CP, q.shape[0] * (2 * CP - 2 * rank - 1) // (2 * CP)]
        offset_k = [0] * n
        for i in range(1, n):
            offset_k[i] = offset_k[i - 1] + k_chunks[i - 1].shape[0]
        if rank * 2 >= CP:
            k_chunks[-1], k_chunks[-2] = k_chunks[-2], k_chunks[-1]
            v_chunks[-1], v_chunks[-2] = v_chunks[-2], v_chunks[-1]
            offset_q[-1], offset_q[-2] = offset_q[-2], offset_q[-1]
            offset_k[-1], offset_k[-2] = offset_k[-2], offset_k[-1]
        if n == 2:
            dq = torch.zeros_like(q)
            event = torch.cuda.current_stream().record_event()
        else:
            dq = None
            event = None
        dout._handle.wait(); del dout._handle
        flip_cp_(dout, 0, CP)
        cp_stream.wait_stream(torch.cuda.current_stream())
        for i in range(n):
            causal = i + 2 >= n
            stream = (torch.cuda.current_stream(), cp_stream)[i % 2]
            with torch.cuda.stream(stream):
                seqlen_k = k_chunks[i].shape[0]
                dq_ = attn_backward(dout[offset_q[i]:],
                                    q[offset_q[i]:],
                                    k_chunks[i],
                                    v_chunks[i],
                                    out[offset_q[i]:],
                                    lse[offset_q[i]:],
                                    dk_[offset_k[i]:offset_k[i] + seqlen_k],
                                    dv_[offset_k[i]:offset_k[i] + seqlen_k],
                                    -causal)
                if dq is None:
                    dq = dq_
                else:
                    event.wait()
                    dq[offset_q[i]:] += dq_
                event = stream.record_event(event)
                dq_ = None
            if i == 0:
                pair_gen.send((dq, dk[:mlength], dv[:mlength]))
            if (n == 2 and i == 0) or i + 3 == n:
                next(pair_gen)
        event.wait()
        del dout, k_chunks, v_chunks
        # calculate or receive the remote dq, dk, dv.
        dqi = next(pair_gen)
        return dqi, dk, dv, None, None


class SaveShardedForBackwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, x, *data):
        ctx.group = group
        shards = [t._shard for t in data]
        ctx.save_for_backward(*shards)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        shards = ctx.saved_tensors
        size = dist.get_world_size(ctx.group)

        data = []
        with dist._coalescing_manager(ctx.group, async_ops=True) as cm:
            for s in shards:
                t = s.new_empty(size * s.shape[0], *s.shape[1:])
                dist.all_gather_into_tensor(t, s, group=ctx.group, async_op=True)
                t._handle = cm
                data.append(t)

        return None, grad_x, *data


def cp_qo_attn_func(qi, ki, vi, cp_group, *, kv_cache, layer_idx):
    assert cudnn_attn_check_capability(use_causal_mask_bottom_right=False), \
        "cuDNN SDPA is required by CP-QO"
    # TODO(lizhouyang): warm up for cudnn sdpa with ctx_pair.
    CudnnAttnFunc._ALLOW_GRAPH_CREATION = True
    ki, vi = kv_cache.update(layer_idx, [ki, vi])
    oi, *qo = CPQOAttnFunc.apply(qi, ki, vi, cp_group, kv_cache.ctx_pair)
    def save_data(x):
        return SaveShardedForBackwardFunc.apply(parallel_state.get_context_parallel_group_slow(), x, *qo)
    return oi, save_data
