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
#
# This file has been modified from its original version
#

"""Pretrain utilities."""
import socket

import math
import sys
import torch
import wandb
import deepspeed
import os

from wandb import UsageError
from apex.optimizers import FusedAdam as Adam
from datetime import datetime

from megatron import get_tokenizer, get_args, get_timers, get_tensorboard_writer, mpu, print_rank_0
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.data_loading import build_train_valid_test_data_iterators
from megatron.fp16 import fp32_to_fp16
from megatron.global_vars import get_use_wandb
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.model import get_params_for_weight_decay_optimization
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron.global_vars import set_use_wandb
from megatron.model import GPT2Model, GPT2ModelPipe
from megatron.utils import get_ltor_masks_and_position_ids, is_local_main, local_rank, get_wandb_api_key, \
    get_total_params, check_adlr_autoresume_termination, human_readable_flops, get_global_batch_size, get_deepspeed_config, get_flops, report_memory


def pretrain(train_valid_test_dataset_provider, model_provider,
             forward_step_func, extra_args_provider=None, args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # Data stuff.
    timers('train/valid/test data iterators').start()
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
        train_valid_test_dataset_provider)
    timers('train/valid/test data iterators').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator)

    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   0, True)


def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = model_provider_func()

    if args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_optimizer(model):
    """Set up the optimizer."""
    args = get_args()
    param_groups = get_params_for_weight_decay_optimization(model, args)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay)
    elif args.onebitadam:
        assert args.deepspeed
        optimizer = None
        # onebitadam needs to be instantiated within the deepspeed engine to work :|
    elif args.sm3:
        from .optimizers import SM3
        optimizer = SM3(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            beta=args.adam_beta1,
            eps=args.adam_eps,
        )
    else:
        # Use Adam
        optimizer = Adam(param_groups,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.adam_beta1, args.adam_beta2),
                         eps=args.adam_eps,
                         adam_w_mode=not args.no_adamw)
    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer, param_groups
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()
    if args.deepspeed and args.onebitadam:
        print_rank_0("WARNING: onebitadam requires the lr scheduler be built by deepspeed - "
                     "Make sure one is added to your deepspeed config")
        return None

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)
    optimizer, param_groups = get_optimizer(model)
    lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu if args.pipe_parallel_size == 0 else None,
            dist_init_required=False,
            model_parameters=param_groups if optimizer is None else None,
            config_params=get_deepspeed_config(args),
        )

        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        if args.pipe_parallel_size > 0:
            model.set_batch_fn(model.module._megatron_batch_fn)
    else:
        raise ValueError("Must be using deepspeed to run neox")

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def backward_step(optimizer, model, loss):
    """Backward step."""
    args = get_args()
    timers = get_timers()

    # Backward pass.
    timers('backward-backward').start()
    if args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers('backward-backward').stop()

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('backward-allreduce').reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")

    if not args.deepspeed:
        # Update master gradients.
        timers('backward-master-grad').start()
        if args.fp16:
            optimizer.update_master_grads()
        timers('backward-master-grad').stop()

        # Clipping gradients helps prevent the exploding gradient.
        timers('backward-clip-grad').start()
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)
        timers('backward-clip-grad').stop()


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Pipeline parallelism schedules forward/backward/step
    if args.pipe_parallel_size > 0:
        return train_step_pipe(model, data_iterator)

    # Forward model for one step.
    timers('forward').start()
    loss, loss_reduced = forward_step_func(data_iterator, model)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, model, loss)
    timers('backward').stop()

    # Update parameters.
    skipped_iter = 0
    timers('optimizer').start()
    if args.deepspeed:
        model.step()
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers('optimizer').stop()

    return loss_reduced, skipped_iter


def train_step_pipe(model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine. """
    args = get_args()
    timers = get_timers()

    assert args.deepspeed
    loss = model.train_batch(data_iter=data_iterator)
    loss_dict = {'lm loss': loss}
    if args.fp16 and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    # Don't break Megatron's timers because we changed code paths.
    for t in ['forward', 'backward', 'allreduce', 'optimizer', 'batch generator',
              'data loader']:
        timers(t).reset()
    return loss_dict, skipped_iter


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    timers('interval time').start()
    report_memory_flag = True
    while iteration < args.train_iters:
        loss_dict, skipped_iter = train_step(forward_step_func,
                                             train_data_iterator,
                                             model,
                                             optimizer,
                                             lr_scheduler)
        iteration += 1

        # Logging.
        loss_scale = None
        if args.fp16:
            loss_scale = optimizer.cur_scale if args.deepspeed else optimizer.loss_scale
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter, model)

        # Autoresume
        if args.adlr_autoresume and \
                (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Checkpointing
        if args.save and args.save_interval and \
                iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
                args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_rank_0('rank: {} | time: {} | exiting the program at '
                         'iteration {}'.format(rank, time_str, iteration))
            sys.exit()

    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))
            # Forward evaluation.
            _, loss_dict = forward_step_func(data_iterator, model)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            # Reduce across processes.
            for key in loss_dict:
                total_loss_dict[key] = total_loss_dict.get(key, 0.) + \
                                       loss_dict[key]
    # Move model back to the train mode.
    model.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters

    return total_loss_dict


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    writer = get_tensorboard_writer()

    # Pipeline parallelism needs eval_batch() instead of a simple forward().
    args = get_args()
    if args.pipe_parallel_size > 0:
        def _eval_helper(data_iter, pipe_model):
            loss = model.eval_batch(data_iter)
            return None, {'lm loss': loss}

        forward_step_func = _eval_helper

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('{} value'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} ppl'.format(key), ppl, iteration)

        if get_use_wandb() and torch.distributed.get_rank() == 0:
            wandb.log({
                'validation {} value'.format(key): total_loss_dict[key].item(),
                'validation {} ppl'.format(key): ppl
            }, step=iteration)

    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)


def model_provider():
    """Build the model."""

    args = get_args()

    print_rank_0('building GPT2 model ...')
    if args.pipe_parallel_size == 0:
        model = GPT2Model(num_tokentypes=0, parallel_output=True)
    else:
        model = GPT2ModelPipe(num_tokentypes=0, parallel_output=True, topology=mpu.get_topology())
        # This is a hack to give us a reference to get_batch_pipe from within training.py
        # We need to call model.set_batch_fn after deepspeed.initialize
        model._megatron_batch_fn = get_batch_pipe

    ## Wandb. (one worker per machine)
    use_wandb = is_local_main() and (get_wandb_api_key() is not None)
    set_use_wandb(use_wandb)
    args_dict = vars(args)
    if use_wandb:
        group_name = args_dict.get('wandb_group')
        name = f'{socket.gethostname()}-{local_rank()}' if group_name else None

        try:
            wandb.init(project="neox", group=group_name, name=name, save_code=False,
                       force=False, entity=args_dict.get('wandb_team'))
        except UsageError as e:
            set_use_wandb(False)
            print(e)
            print('Skipping wandb. Execute `wandb login` on local or main node machine to enable.')

    if use_wandb:
        wandb.config.update(args_dict)

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data):
    """A modification of get_batch() to work with the latest batch instead of an iterator. """
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    # unpack data
    if args.fp16:
        # cast to fp16 because pipeline parallelism skips the FP16 wrapper.
        return fp32_to_fp16((tokens, position_ids, attention_mask)), fp32_to_fp16((labels, loss_mask))
    else:
        return (tokens, position_ids, attention_mask), (labels, loss_mask)


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch generator').stop()
    # Forward model.
    losses = model(tokens, position_ids, attention_mask, labels=labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT2 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT2 datasets ...")

    return train_ds, valid_ds, test_ds


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
        flops_per_s_per_gpu = get_flops(args, iteration_time)
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