import argparse
import os
import time
import queue
import sys
import warnings

sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler
from utils.progressbar import ProgressBar

warnings.filterwarnings("ignore")
from utils.misc import (
    load_config,
    make_model_dir,
    make_logger, make_writer, make_wandb,
    set_seed,
    is_main_process, init_DDP,
    synchronize
)
from dataset.Dataloader import build_dataloader
from prediction import evaluation
import wandb
import errno


def build_optimizer_with_llm_lr(config, model, logger=None):

    lr_config = config.get('learning_rate', {})
    lr_default = lr_config.get('default', 1e-3)
    lr_recognition = lr_config.get('recognition', lr_default)
    lr_translation = lr_config.get('translation', lr_default)
    lr_mapper = lr_config.get('mapper', lr_default)
    lr_llm = lr_config.get('llm_prefix', 2e-4)

    llm_keywords = [
        'llm_prefix_generator',
        'prior_llm_prefix_generator',
        'posterior_llm_prefix_generator',
        'upscaler',
        'downscaler',
        'prefix_queries',
        'condition_embedding',
        'lora',
    ]

    param_groups = []
    param_stats = {
        'recognition': {'params': 0, 'trainable': 0},
        'mapper':      {'params': 0, 'trainable': 0},
        'translation': {'params': 0, 'trainable': 0},
        'llm_prefix':  {'params': 0, 'trainable': 0},
    }

    if hasattr(model, 'recognition_network'):
        rec_params = []
        for name, param in model.recognition_network.named_parameters():
            param_stats['recognition']['params'] += param.numel()
            if param.requires_grad:
                rec_params.append(param)
                param_stats['recognition']['trainable'] += param.numel()
        if rec_params:
            param_groups.append({'params': rec_params, 'lr': lr_recognition, 'name': 'recognition_network'})

    if hasattr(model, 'vl_mapper'):
        mapper_params = []
        for name, param in model.vl_mapper.named_parameters():
            param_stats['mapper']['params'] += param.numel()
            if param.requires_grad:
                mapper_params.append(param)
                param_stats['mapper']['trainable'] += param.numel()
        if mapper_params:
            param_groups.append({'params': mapper_params, 'lr': lr_mapper, 'name': 'vl_mapper'})

    if hasattr(model, 'translation_network'):
        translation_params = []
        llm_params = []

        for name, param in model.translation_network.named_parameters():
            if not param.requires_grad:
                continue
            is_llm_param = any(kw in name.lower() for kw in llm_keywords)
            if is_llm_param:
                llm_params.append(param)
                param_stats['llm_prefix']['trainable'] += param.numel()
            else:
                translation_params.append(param)
                param_stats['translation']['trainable'] += param.numel()

        for name, param in model.translation_network.named_parameters():
            is_llm_param = any(kw in name.lower() for kw in llm_keywords)
            if is_llm_param:
                param_stats['llm_prefix']['params'] += param.numel()
            else:
                param_stats['translation']['params'] += param.numel()

        if translation_params:
            param_groups.append({'params': translation_params, 'lr': lr_translation, 'name': 'translation_network'})
        if llm_params:
            param_groups.append({'params': llm_params, 'lr': lr_llm, 'name': 'llm_prefix_generator'})

    if logger:
        logger.info("=" * 60)
        logger.info("优化器参数分组统计:")
        logger.info("=" * 60)
        lr_map = {'recognition': lr_recognition, 'mapper': lr_mapper,
                  'translation': lr_translation, 'llm_prefix': lr_llm}
        for group_name, stats in param_stats.items():
            if stats['params'] > 0:
                logger.info(f"  [{group_name}]")
                logger.info(f"    - 总参数: {stats['params']:,}")
                logger.info(f"    - 可训练: {stats['trainable']:,}")
                logger.info(f"    - 学习率: {lr_map[group_name]:.2e}")
        logger.info("=" * 60)

    optimizer_type = config.get('optimizer', 'Adam')
    weight_decay = config.get('weight_decay', 0.001)
    betas = config.get('betas', [0.9, 0.998])

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(param_groups, betas=tuple(betas), weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, betas=tuple(betas), weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    if logger:
        logger.info("优化器参数组学习率:")
        for i, group in enumerate(optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            logger.info(f"  learning rate {name}={group['lr']}")

    return optimizer



def save_model(model, optimizer, scheduler, output_file, epoch=None, global_step=None,
               current_score=None, scaler=None):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_score': best_score,
        'current_score': current_score,
        'scaler_state': scaler.state_dict() if scaler is not None else None,
    }

    start_time = time.time()
    logger.info("Saving model state as " + output_file)
    torch.save(state, output_file)
    logger.info("Save model takes {:.2f} seconds.".format(time.time() - start_time))
    return output_file


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def evaluate_and_save(
        model, optimizer, scheduler, val_dataloader, cfg, tb_writer,
        wandb_run=None, epoch=None, global_step=None, generate_cfg={},
        do_recognition=True, do_translation=True, scaler=None,
):
    tag = 'epoch_{:02d}'.format(epoch) if epoch is not None else 'step_{}'.format(global_step)
    global best_score, ckpt_queue

    eval_results = evaluation(
        model=model, val_dataloader=val_dataloader, cfg=cfg, tb_writer=tb_writer,
        wandb_run=wandb_run, epoch=epoch, global_step=global_step, generate_cfg=generate_cfg,
        save_dir=os.path.join(cfg['training']['model_dir'], 'validation', tag),
        do_recognition=do_recognition, do_translation=do_translation,
    )

    metric = 'bleu4' if '2T' in cfg['task'] else 'wer'
    sort_key = lambda x: x
    if metric == 'bleu4':
        score = eval_results['bleu']['bleu4']
        best_score = max(best_score, score)
    elif metric == 'wer':
        score = eval_results['wer']
        best_score = min(best_score, score)
        sort_key = lambda x: -x
    logger.info('best_score={:.2f}'.format(best_score))

    output_file = os.path.join(
        cfg['training']['model_dir'], 'ckpts', "{}_{:.2f}_{}.ckpt".format(metric, score, tag)
    )

    if ckpt_queue.full():
        last_score, to_delete = ckpt_queue.get()
        if sort_key(last_score) <= sort_key(score):
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                logger.warning("Wanted to delete old checkpoint %s but file does not exist.", to_delete)
            ckpt_file = save_model(model=model, epoch=epoch, global_step=global_step,
                                   optimizer=optimizer, scheduler=scheduler,
                                   output_file=output_file, current_score=score, scaler=scaler)
            if best_score == score:
                symlink_update("./" + os.path.basename(ckpt_file),
                               os.path.join(cfg['training']['model_dir'], 'ckpts', 'best.ckpt'))
            ckpt_queue.put((sort_key(score), ckpt_file))
        else:
            ckpt_queue.put((sort_key(last_score), to_delete))
    else:
        ckpt_file = save_model(model=model, epoch=epoch, global_step=global_step,
                               optimizer=optimizer, scheduler=scheduler,
                               output_file=output_file, current_score=score, scaler=scaler)
        if best_score == score:
            symlink_update("./" + os.path.basename(ckpt_file),
                           os.path.join(cfg['training']['model_dir'], 'ckpts', 'best.ckpt'))
        ckpt_queue.put((sort_key(score), ckpt_file))

def main():
    parser = argparse.ArgumentParser("SLT Training")
    parser.add_argument("--config", default="configs/phoenix-2014t.yaml", type=str,
                        help="Training configuration file (yaml).")
    parser.add_argument("--wandb", action="store_true", help='turn on wandb')

    args = parser.parse_args()
    cfg = load_config(args.config)

    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    set_seed(seed=cfg["training"].get("random_seed", 42))

    model_dir = make_model_dir(
        model_dir=cfg['training']['model_dir'],
        overwrite=cfg['training'].get('overwrite', False)
    )

    global logger
    logger = make_logger(model_dir=model_dir, log_file='train.rank{}.log'.format(cfg['local_rank']))
    tb_writer = make_writer(model_dir=model_dir)
    wandb_run = make_wandb(model_dir=model_dir, cfg=cfg) if args.wandb else None

    if is_main_process():
        os.system('cp {} {}/'.format(args.config, model_dir))
    synchronize()

    device = cfg['device']

    model = build_model(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info('# Total parameters = {}'.format(sum(p.numel() for p in model.parameters())))
    logger.info('# Trainable parameters = {}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model = DDP(model, device_ids=[cfg['local_rank']], output_device=cfg['local_rank'],
                find_unused_parameters=True)

    train_dataloader, train_sampler = build_dataloader(
        cfg, 'train',
        model.module.text_tokenizer,
        model.module.gloss_tokenizer
    )
    dev_dataloader, dev_sampler = build_dataloader(
        cfg, 'dev',
        model.module.text_tokenizer,
        model.module.gloss_tokenizer
    )

    if is_main_process():
        tb_writer = SummaryWriter(log_dir=os.path.join(model_dir, "tensorboard"))

    has_llm_prefix = (
        hasattr(model.module, 'translation_network') and
        (hasattr(model.module.translation_network, 'llm_prefix_generator') or
         hasattr(model.module.translation_network, 'prior_llm_prefix_generator') or
         hasattr(model.module.translation_network, 'posterior_llm_prefix_generator'))
    )

    if has_llm_prefix:
        logger.info("检测到 LLM Prefix Generator，使用分组学习率优化器")
        optimizer = build_optimizer_with_llm_lr(
            config=cfg['training']['optimization'],
            model=model.module,
            logger=logger
        )
    else:
        logger.info("使用标准优化器")
        optimizer = build_optimizer(config=cfg['training']['optimization'], model=model.module)

    scheduler, scheduler_type = build_scheduler(
        config=cfg['training']['optimization'], optimizer=optimizer)

    scaler = torch.cuda.amp.GradScaler() if cfg['training']['amp'] else None

    total_epoch = cfg['training']['total_epoch']
    global_step = 0

    global ckpt_queue, best_score
    ckpt_queue = queue.PriorityQueue(maxsize=cfg['training']['keep_last_ckpts'])
    best_score = -100 if '2T' in cfg['task'] else 10000

    val_unit = cfg['training']['validation']['unit']
    val_freq = cfg['training']['validation']['freq']
    if val_unit == "epoch":
        val_freq = 1

    do_recognition = cfg['task'] not in ['G2T', 'S2T_glsfree'] and cfg['model']['recognition_weight'] > 0.
    do_translation = cfg['task'] != 'S2G' and cfg['model']['translation_weight'] > 0.

    logger.info("=" * 60)
    logger.info(f"开始训练: Epoch 0 -> {total_epoch}")
    if scaler is not None:
        logger.info(f"AMP 混合精度训练已启用, 初始 scale: {scaler.get_scale()}")
    logger.info("=" * 60)

    for epoch_no in range(total_epoch):
        train_sampler.set_epoch(epoch_no)
        scheduler.step()

        num_training_steps = len(train_dataloader)
        logger.info('Epoch {}, Training steps {}'.format(epoch_no, num_training_steps))
        logger.info(f'Current learning rate: '
                    f'{scheduler.optimizer.param_groups[0]["lr"]:.2e}, '
                    f'Global step: {global_step}')
        if scaler is not None:
            logger.info(f'AMP Scaler scale: {scaler.get_scale()}')

        pbar = ProgressBar(n_total=num_training_steps, desc='Training') if is_main_process() else None

        for step, batch in enumerate(train_dataloader):

            if cfg['training']['amp']:
                with torch.cuda.amp.autocast():
                    model.module.set_train()
                    output = model.forward(global_step=global_step, is_train=True, **batch)
                scaler.scale(output['total_loss']).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                model.module.set_train()
                output = model.forward(global_step=global_step, is_train=True, **batch)
                with torch.autograd.set_detect_anomaly(True):
                    output['total_loss'].backward()
                optimizer.step()

            model.zero_grad()
            optimizer.zero_grad()

            if is_main_process() and tb_writer:
                for k, v in output.items():
                    if '_loss' in k:
                        tb_writer.add_scalar('train/' + k, v, global_step)
                    if 'factor' in k:
                        tb_writer.add_scalar('train/' + k, v, global_step)
                if scaler is not None:
                    tb_writer.add_scalar('train/amp_scale', scaler.get_scale(), global_step)
                if wandb_run is not None:
                    wandb.log({k: v for k, v in output.items() if '_loss' in k})
                    if scaler is not None:
                        wandb.log({'amp_scale': scaler.get_scale()})

            if (
                    is_main_process() and val_unit == 'step' and global_step % val_freq == 0
                    and global_step > cfg['training']['validation']['valid_start_step']
            ):
                evaluate_and_save(
                    cfg=cfg, model=model.module, optimizer=optimizer, scheduler=scheduler,
                    val_dataloader=dev_dataloader, tb_writer=tb_writer, wandb_run=wandb_run,
                    global_step=global_step, generate_cfg=cfg['training']['validation']['cfg'],
                    do_recognition=do_recognition, do_translation=do_translation, scaler=scaler,
                )

            global_step += 1
            if pbar:
                pbar(step)

        if (
                is_main_process() and val_unit == 'epoch' and epoch_no % val_freq == 0
                and epoch_no >= cfg['training']['validation']['valid_start_epoch']
        ):
            evaluate_and_save(
                cfg=cfg, model=model.module, optimizer=optimizer, scheduler=scheduler,
                val_dataloader=dev_dataloader, tb_writer=tb_writer, wandb_run=wandb_run,
                epoch=epoch_no, generate_cfg=cfg['training']['validation']['cfg'],
                do_recognition=do_recognition, do_translation=do_translation, scaler=scaler,
            )
        print()

    if is_main_process():
        load_model_path = os.path.join(cfg['training']['model_dir'], 'ckpts', 'best.ckpt')
        state_dict = torch.load(load_model_path, map_location='cuda', weights_only=False)
        model.module.load_state_dict(state_dict['model_state'])
        epoch = state_dict.get('epoch', 0)
        global_step = state_dict.get('global_step', 0)
        logger.info('Load model ckpt from ' + load_model_path)

        for split in ['dev', 'test']:
            logger.info('Evaluate on {} set'.format(split))
            dataloader, sampler = build_dataloader(
                cfg, split,
                model.module.text_tokenizer,
                model.module.gloss_tokenizer
            )
            evaluation(
                cfg=cfg, model=model.module, val_dataloader=dataloader,
                epoch=epoch, global_step=global_step,
                generate_cfg=cfg['testing']['cfg'],
                save_dir=os.path.join(model_dir, split),
                do_translation=do_translation, do_recognition=do_recognition,
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
