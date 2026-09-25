import math
import os
import time
from collections import OrderedDict, defaultdict
from logging import getLogger

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.evaluator import Evaluator
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_file_name, get_total_steps, log


class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel,
                 tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(config['result_dir'],
                                             config['ckpt_dir'],
                                             f"{config['run_time']}.pth")

        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    def fit(self, train_dataloader, val_dataloader, test_dataloader=None):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
            test_dataloader: Optional. If ``test_eval_interval`` is set in config,
                run test evaluation every N epochs.
        """
        resume_ckpt = self.config.get('resume_ckpt')
        resume_epoch = int(self.config.get('resume_epoch') or 0)
        if resume_ckpt:
            if not os.path.isfile(resume_ckpt):
                raise FileNotFoundError(
                    f'resume_ckpt not found: {resume_ckpt}')
            state = torch.load(resume_ckpt, map_location='cpu')
            if (isinstance(state, dict) and 'state_dict' in state
                    and not any(k in self.model.state_dict() for k in state)):
                state = state['state_dict']
            cleaned = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in state.items()
            }
            self.model.load_state_dict(cleaned, strict=True)
            self.log(f'Loaded resume checkpoint from {resume_ckpt}')

        optimizer = AdamW(self.model.parameters(),
                          lr=self.config['lr'],
                          weight_decay=self.config['weight_decay'])

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        # Prefer absolute warmup_steps (DiffGRM-style) when provided; else ratio.
        if self.config.get('warmup_steps') is not None:
            warmup_steps = int(self.config['warmup_steps'])
        else:
            warmup_steps = math.floor(
                total_n_steps * float(self.config.get('warmup_ratio', 0.0)))
        warmup_steps = max(0, min(warmup_steps, total_n_steps))
        self.log(f"Total steps: {total_n_steps}, warmup steps: {warmup_steps}")

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler)

        n_epochs = np.ceil(total_n_steps /
                           (len(train_dataloader) *
                            self.accelerator.num_processes)).astype(int)
        best_epoch = int(self.config.get('resume_best_epoch') or 0)
        resume_best_val = self.config.get('resume_best_val')
        best_val_score = (-1 if resume_best_val is None else
                          float(resume_best_val))
        skip_val = bool(self.config.get('skip_validation', False))
        last_epoch_ran = resume_epoch
        test_eval_interval = self.config.get('test_eval_interval')
        if test_eval_interval is not None:
            test_eval_interval = int(test_eval_interval)

        if resume_epoch > 0:
            skip_steps = resume_epoch * len(train_dataloader)
            for _ in range(skip_steps):
                scheduler.step()
            self.log(
                f'Resuming at epoch {resume_epoch + 1}/{n_epochs} '
                f'(skipped {skip_steps} scheduler steps); '
                f'best_epoch={best_epoch}, best_val={best_val_score}')

        for epoch in range(resume_epoch, n_epochs):
            start_time = time.time()
            self.model.train()

            total_loss = 0.0
            total_ghost_loss = 0.0
            total_ghost_weighted = 0.0
            total_dual_mix = 0.0
            total_dual_tok = 0.0
            total_dual_item = 0.0
            last_dual_alpha = None

            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
                disable=True,
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(),
                                    self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

                def _term(name):
                    v = getattr(outputs, name, None)
                    return 0.0 if v is None else float(v.item())

                total_ghost_loss += _term('ghost_loss')
                total_ghost_weighted += _term('ghost_weighted')
                total_dual_mix += _term('dual_mix_loss')
                total_dual_tok += _term('dual_tok_loss')
                total_dual_item += _term('dual_item_loss')
                if getattr(outputs, 'dual_alpha', None) is not None:
                    last_dual_alpha = float(outputs.dual_alpha.item())

            n_batches = len(train_dataloader)
            avg_loss = total_loss / n_batches
            avg_ghost = total_ghost_loss / n_batches
            avg_ghost_w = total_ghost_weighted / n_batches
            avg_dual_mix = total_dual_mix / n_batches
            avg_dual_tok = total_dual_tok / n_batches
            avg_dual_item = total_dual_item / n_batches
            log_scalars = {
                "Loss/train_loss": avg_loss,
                "Loss/ghost_loss": avg_ghost,
                "Loss/ghost_weighted": avg_ghost_w,
            }
            if last_dual_alpha is not None:
                log_scalars.update({
                    "Loss/dual_mix": avg_dual_mix,
                    "Loss/dual_tok": avg_dual_tok,
                    "Loss/dual_item": avg_dual_item,
                    "Loss/dual_alpha": last_dual_alpha,
                })
            self.accelerator.log(log_scalars, step=epoch + 1)

            self.log(
                "[Epoch {}] Train Loss: {:.4f} | Cost: {:.2f}s  lr: {:.6f}"
                .format(epoch + 1, avg_loss, time.time() - start_time,
                        scheduler.get_last_lr()[0]))

            last_epoch_ran = epoch + 1

            if not skip_val and (epoch + 1) % self.config['eval_interval'] == 0:

                all_results = self.evaluate(val_dataloader,
                                            split='val',
                                            epoch=epoch)

                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log(
                            {f"Val_Metric/{key}": all_results[key]},
                            step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')

                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        self.save_model()
                        self.log(
                            f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}'
                        )

                if self.config[
                        'patience'] is not None and epoch + 1 - best_epoch >= self.config[
                            'patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    break

            if (test_dataloader is not None and len(test_dataloader) > 0
                    and test_eval_interval and test_eval_interval > 0
                    and (epoch + 1) % test_eval_interval == 0):
                test_results = self.evaluate(test_dataloader,
                                             split='test',
                                             epoch=epoch)
                if self.accelerator.is_main_process:
                    for key in test_results:
                        self.accelerator.log(
                            {f'Test_Metric/{key}': test_results[key]},
                            step=epoch + 1)
                    self.log(
                        f'[Epoch {epoch + 1}] Test Results: {test_results}')

        if skip_val and last_epoch_ran > 0:
            best_epoch = last_epoch_ran
            best_val_score = None
            if self.accelerator.is_main_process:
                self.save_model()
                self.log(
                    f'[Final epoch {last_epoch_ran}] Saved model checkpoint to '
                    f'{self.saved_model_ckpt} (no validation)')
            self.log('Training finished without validation / early stopping.')
        elif not skip_val:
            self.log(
                f'Best epoch: {best_epoch}, Best val score: {best_val_score}')

        return best_epoch, best_val_score

    def save_model(self, ):
        if self.config['use_ddp']:  # unwrap model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
        else:
            torch.save(self.model.state_dict(), self.saved_model_ckpt)

    def evaluate(self, dataloader, split='test', epoch=-1):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()
        # Catalog item embeddings are cached across batches during eval;
        # drop the stale one so it is rebuilt from the current weights.
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, 'invalidate_catalog_emb_cache'):
            unwrapped.invalidate_catalog_emb_cache()

        all_results = defaultdict(list)
        sid_valid_n = 0
        sid_total_n = 0
        eval_start_time = time.time()
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
            disable=True,
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {
                    k: v.to(self.accelerator.device)
                    for k, v in batch.items()
                }
                batch['split'] = split
                batch['epoch'] = epoch

                if self.config[
                        'use_ddp']:  # ddp, gather data from all devices for evaluation

                    if split == 'val':
                        preds = self.model.module.generate(
                            batch, n_return_sequences=self.evaluator.maxk_eval)
                    else:
                        preds = self.model.module.generate(
                            batch, n_return_sequences=self.evaluator.maxk)

                    all_preds, all_labels = self.accelerator.gather_for_metrics(
                        (preds, batch['labels']))

                    results = self.evaluator.calculate_metrics(
                        all_preds, all_labels, split)
                    eval_preds = all_preds
                else:
                    if split == 'val':
                        preds = self.model.generate(
                            batch, n_return_sequences=self.evaluator.maxk_eval)
                    else:
                        preds = self.model.generate(
                            batch, n_return_sequences=self.evaluator.maxk)

                    results = self.evaluator.calculate_metrics(
                        preds, batch['labels'], split)
                    eval_preds = preds

                for key, value in results.items():
                    all_results[key].append(value)

                # Log fraction of top-10 predicted SIDs that exist in the catalog.
                if split == 'test' and self.accelerator.is_main_process:
                    model_ref = self.accelerator.unwrap_model(self.model)
                    if hasattr(model_ref, '_sid_is_in_catalog'):
                        k10 = min(10, eval_preds.shape[1])
                        valid = model_ref._sid_is_in_catalog(
                            eval_preds[:, :k10])
                        sid_valid_n += int(valid.sum().item())
                        sid_total_n += int(valid.numel())

        output_results = OrderedDict()
        for metric in self.config['metrics']:

            topk_list = self.config['topk'] if (
                split == 'test') else self.config['val_topk']
            # for k in self.config['topk']:
            for k in topk_list:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()

        # Model-selection score: 0.8 * NDCG@10 + 0.2 * Recall@10
        if 'ndcg@10' in output_results and 'recall@10' in output_results:
            output_results['score@10'] = (
                0.8 * output_results['ndcg@10']
                + 0.2 * output_results['recall@10'])

        eval_cost = time.time() - eval_start_time
        self.log(f'[{split.capitalize()}] Eval Cost: {eval_cost:.2f}s')
        if split == 'test' and sid_total_n > 0:
            sid_valid_rate = sid_valid_n / sid_total_n
            output_results['sid_valid@10'] = sid_valid_rate
            self.log(
                f'[{split.capitalize()}] Top-10 SID valid rate: '
                f'{sid_valid_rate:.4f} ({sid_valid_n}/{sid_total_n})')

        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)
