# coding=utf-8
# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
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

import sys
import torch
import wandb

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import print_rank_0
from megatron.global_vars import get_use_wandb
from megatron.utils import report_memory, get_flops, get_global_batch_size, human_readable_flops


class Tee:
    """ Duplicate output to both stdout/err and file """

    def __init__(self, file, err=False):
        self.file = open(file, 'w')
        self.err = err
        if not err:
            self.std = sys.stdout
            sys.stdout = self
        else:
            self.std = sys.stderr
            sys.stderr = self

    def __del__(self):
        if not self.err:
            sys.stdout = self.std
        else:
            sys.stderr = self.std
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.std.write(data)

    def flush(self):
        self.file.flush()


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter, model):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Update losses.
    skipped_iters_key = 'skipped iterations'
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    got_nan_key = 'got nan'

    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(key, 0.) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan

    total_loss_dict[got_nan_key] = total_loss_dict.get(
        got_nan_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)

    if args.pipe_parallel_size <= 0:
        add_to_logging('forward')
        add_to_logging('backward')
        add_to_logging('backward-backward')
        add_to_logging('backward-allreduce')
        add_to_logging('backward-master-grad')
        add_to_logging('backward-clip-grad')
        add_to_logging('optimizer')
        add_to_logging('batch generator')
    else:
        # with pipeline parallel, the megatron timers are overridden by the deepspeed ones.
        # Try to grab timer values from model engine. Only recently added to deeperspeed, so check that the engine
        # has that attribute first
        if hasattr(model, 'timer_values') and model.timer_values is not None:
            if model.wall_clock_breakdown() and model.global_steps % model.steps_per_print() == 0:
                timer_values = model.timer_values
                # deepspeed already logs to tensorboard / prints values, so just log to wandb
                if get_use_wandb() and torch.distributed.get_rank() == 0:
                    for key in timer_values:
                        wandb.log({key: timer_values[key]}, step=iteration)

    # Log timer info to tensorboard and wandb
    normalizer = iteration % args.log_interval
    if normalizer == 0:
        normalizer = args.log_interval
    if torch.distributed.get_rank() == 0:
        timers.write(names=timers_to_log, iteration=iteration, normalizer=normalizer)

    # wandb writer
    if get_use_wandb() and torch.distributed.get_rank() == 0:
        wandb.log({'learning_rate': learning_rate}, step=iteration)
        for key in loss_dict:
            wandb.log({key: loss_dict[key]}, step=iteration)
        if args.fp16:
            wandb.log({'loss_scale': loss_scale}, step=iteration)

    # Tensorboard values.
    if writer and torch.distributed.get_rank() == 0:
        writer.add_scalar('learning_rate', learning_rate, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
        if args.fp16:
            writer.add_scalar('loss_scale', loss_scale, iteration)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval time').elapsed()
        iteration_time = elapsed_time / args.log_interval
        samples_per_sec = get_global_batch_size(args) / iteration_time
        log_string = ' samples/sec: {:.3f} |'.format(samples_per_sec)
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('samples/sec', samples_per_sec, iteration)
            writer.add_scalar('iteration_time', iteration_time, iteration)
        if get_use_wandb() and torch.distributed.get_rank() == 0:
            wandb.log({'samples/sec': samples_per_sec}, step=iteration)
            wandb.log({'iteration_time': iteration_time}, step=iteration)
        log_string += ' iteration {:8d}/{:8d} |'.format(iteration,
                                                        args.train_iters)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time * 1000.0 / args.log_interval)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        num_iterations = max(
            1, args.log_interval - total_loss_dict[skipped_iters_key])

        # calculate tflop / gpu
        flops_per_s_per_gpu = get_flops(iteration_time)
        log_string += f' approx flops per GPU: {human_readable_flops(flops_per_s_per_gpu)} |'
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('flops/s/gpu', flops_per_s_per_gpu, iteration)
        if get_use_wandb() and torch.distributed.get_rank() == 0:
            wandb.log({'flops/s/gpu': flops_per_s_per_gpu}, step=iteration)

        for key in total_loss_dict:
            if key not in [skipped_iters_key, got_nan_key]:
                v = total_loss_dict[key].item() if hasattr(total_loss_dict[key], 'item') else total_loss_dict[key]
                avg = v / float(num_iterations)
                log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = 0.0
        if args.fp16:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[got_nan_key])
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[got_nan_key] = 0
        print_rank_0(log_string)
        if report_memory_flag:
            report_memory('after {} iterations'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag
