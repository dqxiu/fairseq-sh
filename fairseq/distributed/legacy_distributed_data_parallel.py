# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
A modified version of the legacy DistributedDataParallel module that uses c10d
communication primitives. This version is simpler than the latest PyTorch
version and is useful for debugging. Notably it does not overlap gradient
communication with the backward pass, which makes it slower but more robust
than the PyTorch version.

This version also supports the *no_sync* context manager, which allows faster
training with `--update-freq`.
"""

from collections import OrderedDict
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

from fairseq.distributed import utils

try:
    from torchscale.component.xmoe.global_groups import get_moe_group
except ModuleNotFoundError:
    get_moe_group = None

def start_pdb_on_rank_zero():
    rank = torch.distributed.get_rank()
    if rank == 0:
        import pdb
        pdb.set_trace()
    else:
        import time
        time.sleep(1e6)



class LegacyDistributedDataParallel(nn.Module):
    """Implements distributed data parallelism at the module level.

    A simplified version of :class:`torch.nn.parallel.DistributedDataParallel`.
    This version uses a c10d process group for communication and does not
    broadcast buffers.

    Args:
        module (~torch.nn.Module): module to be parallelized
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
        buffer_size (int, optional): number of elements to buffer before
            performing all-reduce (default: 256M).
    """

    def __init__(self, module, process_group, buffer_size=2 ** 28):
        super().__init__()


        self.module = module
        self.process_group = process_group

        # Never use a bigger buffer than the number of model params
        self.buffer_size = min(buffer_size, sum(p.numel() for p in module.parameters()))
        self.buffer = None

        # We can also forcibly accumulate grads locally and only do the
        # all-reduce at some later time
        self.accumulate_grads = False

        # make per-device lists of parameters
        paramlists = OrderedDict()
        for param in self.module.parameters():
            device = param.device
            if paramlists.get(device) is None:
                paramlists[device] = []
            paramlists[device] += [param]

        # split into expert and normal params
        per_device_params = list(paramlists.values())
        self.per_device_params_normal = [[k for k in t if not hasattr(k, 'expert')] for t in per_device_params]
        self.per_device_params_expert = [[k for k in t if hasattr(k, 'expert')] for t in per_device_params]

        assert all([len([k for k in t if hasattr(k, 'base_expert')]) == 0 for t in per_device_params])

        # assign local pg
        if hasattr(get_moe_group, "_moe_groups"): # need to init groups first
            _, self.local_pg = get_moe_group()
        else:
            self.local_pg = None

        #start_pdb_on_rank_zero()
        #print('hi')

    @contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        old_accumulate_grads = self.accumulate_grads
        self.accumulate_grads = True
        yield
        self.accumulate_grads = old_accumulate_grads

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def all_reduce_params(self, params, curr_buffer, curr_process_group):
        buffer = curr_buffer
        if len(params) > 1:
            offset = 0
            for p in params:
                sz = p.numel()
                if p.grad is not None:
                    buffer[offset : offset + sz].copy_(p.grad.data.view(-1))
                else:
                    buffer[offset : offset + sz].zero_()
                offset += sz
        else:
            # we only have a single grad to all-reduce
            p = params[0]
            if p.grad is not None:
                buffer = p.grad.data
            elif p.numel() <= curr_buffer.numel():
                buffer = buffer[: p.numel()]
                buffer.zero_()
            else:
                buffer = torch.zeros_like(p)

        utils.all_reduce(buffer, curr_process_group)

        # copy all-reduced grads back into their original place
        offset = 0
        for p in params:
            sz = p.numel()
            if p.grad is not None:
                p.grad.data.copy_(buffer[offset : offset + sz].view_as(p))
            else:
                p.grad = buffer[offset : offset + sz].view_as(p).clone()
            offset += sz



    def all_reduce_grads(self):
        """
        This function must be called explicitly after backward to reduce
        gradients. There is no automatic hook like c10d.
        """

        # This function only needs to be called once
        if self.accumulate_grads:
            return

        if self.buffer is None:
            self.buffer = next(self.module.parameters()).new(self.buffer_size)

        # reduce normal params
        curr_world_size = dist.get_world_size(self.process_group)
        self._all_reduce_grads(self.per_device_params_normal, self.buffer, self.process_group, curr_world_size)
        # reduce expert params
        if self.local_pg is not None:
            self._all_reduce_grads(self.per_device_params_expert, self.buffer, self.local_pg, curr_world_size)


    def _all_reduce_grads(self, current_params, curr_buffer, curr_process_group, curr_world_size):
        for params in current_params:
            # All-reduce the gradients in buckets
            offset = 0
            buffered_params = []
            for param in params:
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                else:
                    param.grad.data.div_(curr_world_size)

                if param.grad.requires_grad:
                    raise RuntimeError(
                        "DistributedDataParallel only works "
                        "with gradients that don't require "
                        "grad"
                    )
                sz = param.numel()
                if sz > curr_buffer.numel():
                    # all-reduce big params directly
                    self.all_reduce_params([param], curr_buffer, curr_process_group)
                else:
                    if offset + sz > curr_buffer.numel():
                        self.all_reduce_params(buffered_params, curr_buffer, curr_process_group)
                        offset = 0
                        buffered_params.clear()
                    buffered_params.append(param)
                    offset += sz

            if len(buffered_params) > 0:
                self.all_reduce_params(buffered_params, curr_buffer, curr_process_group)
