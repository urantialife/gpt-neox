# coding=utf-8
# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utilities."""
import os
import sys
import re
from typing import Dict, List

import requests
import torch
from deepspeed.launcher.runner import fetch_hostfile, parse_inclusion_exclusion

from megatron import get_args
from megatron import print_rank_0
from megatron import get_adlr_autoresume
from megatron import mpu
from megatron.fp16 import FP16_Optimizer


def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    reduced_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(reduced_losses)
    reduced_losses = reduced_losses / torch.distributed.get_world_size()

    return reduced_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    print_rank_0(string)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, model-parallel,min, max, norm\n'
    optimizer_ = optimizer
    if isinstance(optimizer, FP16_Optimizer):
        optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = param.data.norm()
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, lr_scheduler):
    """Check for autoresume signal and exit if it is received."""
    # to prevent circular import
    from megatron.checkpointing import save_checkpoint
    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def local_rank():
    """ Local rank of process """
    return int(os.environ["LOCAL_RANK"])


def is_local_main():
    """ True if is the local main process """
    return local_rank() == 0


def get_wandb_api_key():
    """ Get Weights and Biases API key from ENV or .netrc file. Otherwise return None """
    if 'WANDB_API_KEY' in os.environ:
        return os.environ['WANDB_API_KEY']

    wandb_token = requests.utils.get_netrc_auth('https://api.wandb.ai')

    if wandb_token is not None:
        return wandb_token[1]


def neox_args(parser):
    group = parser.add_argument_group(title='Weights and Biases monitoring args')

    group.add_argument('--wandb_group', type=str, default=None,
                       help='Weights and Biases group name - used to group together "runs".')
    group.add_argument('--wandb_team', type=str, default=None,
                       help='Team name for Weights and Biases.')
    group.add_argument('--git_hash', type=str, default=None,
                       help='current git hash of repository')
    return parser


def obtain_resource_pool(hostfile_path, include_arg, exclude_arg) -> Dict[str, List[int]]:
    """
    Get dict of `resource_pool[hostname] = [list of GPU ranks]` using hostfile, include and exclude args.
    Modified from: `deepspeed.launcher.runner.main`
    """
    resource_pool = fetch_hostfile(hostfile_path)
    if not resource_pool:
        resource_pool = {}
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool['localhost'] = device_count

    active_resources = parse_inclusion_exclusion(resource_pool,
                                                 include_arg,
                                                 exclude_arg)
    return active_resources


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_total_params(model):
    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        params = sum([p.nelement() for p in model.parameters()])
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(), params), flush=True)
    else:
        params = 0

    total_n_parameters = torch.tensor([params]).cuda(torch.cuda.current_device())
    torch.distributed.all_reduce(total_n_parameters)
    total_n_parameters = total_n_parameters.item()
    return total_n_parameters


def human_readable_flops(n):
    for unit in ['', 'KFLOPS', 'MFLOPS', 'GFLOPS', 'TFLOPS', 'PFLOPS', 'EFLOPS', 'ZFLOPS']:
        if abs(n) < 1000.0:
            return "%3.1f%s" % (n, unit)
        n /= 1000.0
    return "%.1f%s" % (n, 'Yi')


def get_global_batch_size(args):
    return args.batch_size * mpu.get_data_parallel_world_size() * args.gas


def get_flops(iter_time_s):
    args = get_args()

    world_size = torch.distributed.get_world_size()
    global_batch_size = get_global_batch_size(args)
    global_flops_per_iteration = flops_per_iteration(args.hidden_size, args.num_layers, global_batch_size,
                                                     args.seq_length, args.padded_vocab_size)
    return global_flops_per_iteration / (iter_time_s * world_size)


def flops_per_iteration(hidden_size, num_layers, batch_size, seq_len, vocab_size):
    """Flops formula from https://arxiv.org/pdf/2104.04473.pdf"""
    return (96 * batch_size * seq_len * num_layers * hidden_size ** 2) * (1 + (seq_len /
                                                                               (6 * hidden_size)) + (vocab_size / (
            16 * num_layers * hidden_size)))


def get_deepspeed_config():
    # Determine if deepspeed config is JSON or filepath.
    # If JSON then directly load it
    args = get_args()
    deepspeed_conf = None
    if hasattr(args, 'deepspeed_config'):
        if not os.path.exists(args.deepspeed_config):
            # If it's not a path trying parsing as a JSON string
            deepspeed_json_conf = args.deepspeed_config
            if len(deepspeed_json_conf) > 2 and deepspeed_json_conf[0] == "'" and deepspeed_json_conf[-1] == "'":
                deepspeed_json_conf = deepspeed_json_conf[1:-1]  # Remove shell quotes
            try:
                deepspeed_conf = json.loads(deepspeed_json_conf)
                args.deepspeed_config = None  # Pass directly as dictionary to deepspeed
            except JSONDecodeError:
                # Not a path or a string
                raise ValueError(
                    f'The parameter `deepspeed_config` is neither a file path that exists or a JSON string:'
                    f' {args.deepspeed_config}')

    return deepspeed_conf
