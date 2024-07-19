# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

import contextlib
import math
import torch

from collections import defaultdict


_MEMCPY_STREAM = dict()
_GPU_BUFFER_POOL = dict()
_CPU_BUFFER_POOL = list()


def set_ideal_affinity_for_current_gpu():
    import cuda.cuda
    import cuda.cudart
    import pynvml
    import uuid
    err, device_id = cuda.cudart.cudaGetDevice()
    assert err == cuda.cudart.cudaError_t.cudaSuccess
    err, device_uuid = cuda.cuda.cuDeviceGetUuid(device_id)
    assert err == cuda.cuda.CUresult.CUDA_SUCCESS
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)


def get_memcpy_stream(key):
    if key not in _MEMCPY_STREAM:
        _MEMCPY_STREAM[key] = torch.cuda.Stream()
    return _MEMCPY_STREAM[key]


def get_persistent_gpu_buffer(key, size):
    if key not in _GPU_BUFFER_POOL or _GPU_BUFFER_POOL[key].numel() < size:
        _GPU_BUFFER_POOL[key] = None  # release before allocate
        _GPU_BUFFER_POOL[key] = torch.empty(size, dtype=torch.uint8, device="cuda")
        _GPU_BUFFER_POOL[key].ref_cnt = 0
    return _GPU_BUFFER_POOL[key][:size]


def get_cpu_buffer(size):
    best_i = -1
    for i, buffer in enumerate(_CPU_BUFFER_POOL):
        if buffer.numel() >= size:
            if best_i == -1 or buffer.numel() < _CPU_BUFFER_POOL[best_i].numel():
                best_i = i
    if best_i != -1:
        return _CPU_BUFFER_POOL.pop(best_i)[:size]
    if _CPU_BUFFER_POOL:
        _CPU_BUFFER_POOL.pop()
    set_ideal_affinity_for_current_gpu()
    import offload_utils_ext
    buffer = offload_utils_ext.wrap_cuda_malloc_host(size)
    return buffer[:size]


def recycle_cpu_buffer(buffer):
    _CPU_BUFFER_POOL.append(buffer._base)


def copy2d_(dst, src):
    assert dst.dtype == src.dtype, "dtype mismatch"
    if not dst.is_contiguous():
        raise NotImplementedError(f"unsupported dst shape {dst.shape} stride {dst.stride()}")
    shape = src.shape
    stride = src.stride()
    if stride[-1] == 1 and all(stride[i] == shape[i + 1] * stride[i + 1] for i in range(0, len(shape) - 2)):
        import offload_utils_ext
        dw = src.dtype.itemsize
        cudaMemcpyDefault = 4
        offload_utils_ext.wrap_cuda_memcpy_2d_async(dst.data_ptr(), shape[-1] * dw, src.data_ptr(), stride[-2] * dw,
                                                 shape[-1] * dw, shape[:-1].numel(), cudaMemcpyDefault,
                                                 torch.cuda.current_stream().cuda_stream)
    else:
        raise NotImplementedError(f"unsupported src shape {shape} stride {stride}")


def fast_contiguous(x):
    if x.is_contiguous():
        return x
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    copy2d_(out, x)
    return out


class TensorWrap:
    def __init__(self, x):
        self.x = x
        self.shape = x.shape
        self.dtype = x.dtype
        self.device = x.device
        self.base = None


class TensorPack:
    def __init__(self, tensor_wrap):
        self.tensor_wrap = tensor_wrap

    def get(self):
        return self.tensor_wrap.x

    def __del__(self):
        self.tensor_wrap.x = None
        if self.tensor_wrap.base is not None:
            self.tensor_wrap.base.ref_cnt -= 1


class ActivationGroup:
    def __init__(self, tensors, offload_ratio):
        self.tensors = tensors
        self.offload_ratio = offload_ratio

    def offload_prologue(self, *, use_bucket):
        if not self.tensors:
            return None, None, None
        self.map = list()
        top = 0
        for i, tensor in enumerate(self.tensors):
            duplicate_flag = False
            if tensor.x.is_contiguous():
                for j, prev_tensor in enumerate(self.tensors[:i]):
                    if tensor.x.data_ptr() == prev_tensor.x.data_ptr() and prev_tensor.x.is_contiguous() and tensor.device == prev_tensor.device and tensor.shape.numel() == prev_tensor.shape.numel():
                        begin_idx, end_idx, _0 = self.map[j]
                        duplicate_flag = True
                        self.map.append((begin_idx, end_idx, duplicate_flag))
                        break
            if not duplicate_flag:
                n = tensor.shape.numel() * tensor.dtype.itemsize
                self.map.append((top, top + n, duplicate_flag))
                top += n
        MiB = 2 ** 20
        offload_size = (int(math.ceil(top * self.offload_ratio)) + MiB - 1) // MiB * MiB
        if use_bucket:
            buffer = get_persistent_gpu_buffer("offload", offload_size)
        else:
            buffer = None
        copy_tasks = []
        partially_offloaded_bases = set()
        for tensor, (begin_idx, end_idx, duplicate_flag) in zip(self.tensors, self.map):
            assert tensor.x.device.type == "cuda"
            if end_idx <= offload_size:
                if not duplicate_flag:
                    if tensor.x._base is not None:
                        partially_offloaded_bases.add(tensor.x._base)
                    if use_bucket:
                        buffer[begin_idx:end_idx].view(tensor.dtype).view(tensor.shape).copy_(tensor.x)
                    else:
                        copy_tasks.append((begin_idx, end_idx, tensor.x))
                tensor.x = None
            elif begin_idx < offload_size:
                if not duplicate_flag:
                    if tensor.x._base is not None:
                        partially_offloaded_bases.add(tensor.x._base)
                    linear_data = fast_contiguous(tensor.x).view(-1).view(torch.uint8)
                    if use_bucket:
                        buffer[begin_idx:].copy_(linear_data[:offload_size - begin_idx])
                    else:
                        copy_tasks.append((begin_idx, offload_size, linear_data[:offload_size - begin_idx]))
                    self.remained_not_offloaded = linear_data[offload_size - begin_idx:].clone()
                tensor.x = None
            elif tensor.x._base in partially_offloaded_bases:
                if duplicate_flag:
                    raise NotImplementedError("does not support partially offload duplicate tensors")
                tensor.x = tensor.x.clone()
        self.buffer_cpu = get_cpu_buffer(offload_size)
        stream = get_memcpy_stream("offload")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if use_bucket:
                self.buffer_cpu.copy_(buffer, non_blocking=True)
            else:
                for begin_idx, end_idx, x in copy_tasks:
                    if x.is_contiguous():
                        self.buffer_cpu[begin_idx:end_idx].view(x.dtype).view(x.shape).copy_(x, non_blocking=True)
                    else:
                        copy2d_(self.buffer_cpu[begin_idx:end_idx].view(x.dtype).view(x.shape), x)
        return stream, buffer, copy_tasks

    def offload_epilogue(self, stream, buffer, copy_tasks):
        if not self.tensors:
            return
        torch.cuda.current_stream().wait_stream(stream)

    def onload_prologue(self, *, overlap_d2h_h2d, ping_pong_onload):
        if not self.tensors:
            return None, None, ping_pong_onload
        stream_key = "onload" if overlap_d2h_h2d else "offload"
        if ping_pong_onload:
            buffer_key = "onload_ping"
            buffer = get_persistent_gpu_buffer(buffer_key, self.buffer_cpu.numel())
            if buffer._base.ref_cnt > 0:
                buffer_key = "onload_pong"
        else:
            buffer_key = stream_key
        stream = get_memcpy_stream(stream_key)
        buffer = get_persistent_gpu_buffer(buffer_key, self.buffer_cpu.numel())
        assert buffer._base.ref_cnt == 0, "last onload tensors are not fully deleted"
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            buffer.copy_(self.buffer_cpu, non_blocking=True)
        return stream, buffer, ping_pong_onload

    def onload_epilogue(self, stream, buffer, ping_pong_onload):
        if not self.tensors:
            return
        torch.cuda.current_stream().wait_stream(stream)
        recycle_cpu_buffer(self.buffer_cpu)
        self.buffer_cpu = None
        offload_size = buffer.numel()
        duplicate_tensors = dict()
        for tensor, (begin_idx, end_idx, duplicate_flag) in zip(self.tensors, self.map):
            if end_idx <= offload_size:
                tensor.x = buffer[begin_idx:end_idx].view(tensor.dtype).view(tensor.shape)
                if ping_pong_onload:
                    tensor.base = buffer._base
                    tensor.base.ref_cnt += 1
                else:
                    tensor.x = tensor.x.clone()
            elif begin_idx < offload_size:
                if not duplicate_flag:
                    tensor.x = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                    linear_data = tensor.x.view(-1).view(buffer.dtype)
                    linear_data[:offload_size - begin_idx].copy_(buffer[begin_idx:])
                    linear_data[offload_size - begin_idx:].copy_(self.remained_not_offloaded)
                    self.remained_not_offloaded = None
                    duplicate_tensors[begin_idx, end_idx] = linear_data
                else:
                    tensor.x = duplicate_tensors[begin_idx, end_idx].view(tensor.dtype).view(tensor.shape)
        del self.tensors
        del self.map


class ForwardEmptyBackwardIdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.empty((), dtype=x.dtype, device=x.device).expand_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class ForwardLeftBackwardRightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        return left

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


groups = dict()


@contextlib.contextmanager
def record(key, offload_ratio):
    if offload_ratio == 0.:
        yield
        groups[key] = ActivationGroup([], offload_ratio)
        return

    tensors = list()

    def pack_hook(x):
        tensor_wrap = TensorWrap(x)
        is_parameter = isinstance(x, torch.nn.Parameter)
        is_main_grad = getattr(x, "is_main_grad", False)
        is_too_small = x.numel() * x.element_size() < 1024 * 1024
        is_rope_freqs = x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 1
        if not is_parameter and not is_main_grad and not is_too_small and not is_rope_freqs:
            tensors.append(tensor_wrap)
        return TensorPack(tensor_wrap)

    def unpack_hook(tensor_pack):
        x = tensor_pack.get()
        return x

    import transformer_engine.pytorch.cpu_offload
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        CPUOffloadEnabled = transformer_engine.pytorch.cpu_offload.CPUOffloadEnabled
        transformer_engine.pytorch.cpu_offload.CPUOffloadEnabled = True
        try:
            yield
        finally:
            transformer_engine.pytorch.cpu_offload.CPUOffloadEnabled = CPUOffloadEnabled

    groups[key] = ActivationGroup(tensors, offload_ratio)


@contextlib.contextmanager
def offload_async(key):
    group = groups[key]
    args = group.offload_prologue(use_bucket=False)
    yield
    group.offload_epilogue(*args)


@contextlib.contextmanager
def onload_async(key):
    group = groups[key]
    args = group.onload_prologue(overlap_d2h_h2d=True, ping_pong_onload=True)
    yield
    group.onload_epilogue(*args)


def get_forward_tensor_and_backward_handle(x):
    backward_handle = torch.empty((), dtype=x.dtype, device=x.device).expand_as(x)
    backward_handle.requires_grad_(x.requires_grad)
    x.requires_grad_(False)
    x = ForwardLeftBackwardRightFunction.apply(x, backward_handle)
    return x, backward_handle


def forward_empty_backward_identity(x):
    return ForwardEmptyBackwardIdentityFunction.apply(x)
