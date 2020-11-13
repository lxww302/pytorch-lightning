from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    Adafactor,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    T5ForConditionalGeneration
)
import GPUtil
import os
from time import time


class GLUEDataModule(pl.LightningDataModule):

    task_text_field_map = {
        'cola': ['sentence'],
        'sst2': ['sentence'],
        'mrpc': ['sentence1', 'sentence2'],
        'qqp': ['question1', 'question2'],
        'stsb': ['sentence1', 'sentence2'],
        'mnli': ['premise', 'hypothesis'],
        'qnli': ['question', 'sentence'],
        'rte': ['sentence1', 'sentence2'],
        'wnli': ['sentence1', 'sentence2'],
        'ax': ['premise', 'hypothesis']
    }

    glue_task_num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'qqp': 2,
        'stsb': 1,
        'mnli': 3,
        'qnli': 2,
        'rte': 2,
        'wnli': 2,
        'ax': 3
    }

    loader_columns = [
        'datasets_idx',
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'start_positions',
        'end_positions',
        'labels'
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = 'mrpc',
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.num_workers = 96
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if 'gpt' in model_name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage):
        self.dataset = datasets.load_dataset('glue', self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=['label'],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if 'validation' in x]

    def prepare_data(self):
        datasets.load_dataset('glue', self.task_name)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size, num_workers=self.num_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.num_workers) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size, num_workers=self.num_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.num_workers) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if True:
            texts_or_text_pairs = []
            for sent1, sent2 in zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]):
                texts_or_text_pairs.append(F"{sent1} </s> {sent2}")
        elif len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length'
        )

        # Rename label to labels to make it easier to pass to model forward
        if 't5' in self.model_name_or_path:
            labels = []
            decoder_attention_mask = []
            for l in example_batch['label']:
                word = 'positive' if l == 0 else 'negative'
                result = self.tokenizer(word)
                labels.append(result['input_ids'])
                decoder_attention_mask.append(result['attention_mask'])
            features['labels'] = labels
            features['decoder_attention_mask'] = decoder_attention_mask
        else:
            features['labels'] = example_batch['label']
        return features


class GLUETransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        if 'gpt' in model_name_or_path:
            self.config.pad_token_id = self.config.eos_token_id
        if 't5' in model_name_or_path:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            'glue',
            'sst2',
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        labels = batch["labels"]

        if 't5' in self.hparams.model_name_or_path:
            preds = torch.argmax(logits, axis=-1)[:, 0]
            labels = labels[:, 0]
        elif self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == 'mnli':
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split('_')[-1]
                preds = torch.cat([x['preds'] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x['labels'] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x['loss'] for x in output]).mean()
                self.log(f'val_loss_{split}', loss, prog_bar=True)
                split_metrics = {f"{k}_{split}": v for k, v in self.metric.compute(
                    predictions=preds, references=labels).items()}
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = self.metric.compute(predictions=preds, references=labels)
        print('metrics', metrics)
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus))) //
                self.hparams.accumulate_grad_batches *
                float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                              relative_step=False, warmup_init=False, scale_parameter=False)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-4, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser


def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_argparse_args(parser)
    parser = GLUETransformer.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


def main(args):
    pl.seed_everything(args.seed)
    dm = GLUEDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup('fit')
    model = GLUETransformer(num_labels=dm.num_labels, eval_splits=dm.eval_splits, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    return dm, model, trainer


def prepare_inputs(tokenizer,
                   input_file='/data00/wuwei.ai/data/OpenSubtitles/en_tiny.txt',
                   batch_size=8,
                   max_length=128):
    lines = open(input_file).readlines()
    batch_lines = [lines[i: i + batch_size] for i in range(0, len(lines) - batch_size, batch_size)][:1000]
    outputs = []
    for batch in batch_lines:
        batch_tokens = tokenizer(batch, max_length=max_length, truncation=True,
                                 padding='max_length', return_tensors="pt")['input_ids']
        outputs.append(batch_tokens)
    return outputs


def run(model_name='distilbert-base-uncased'):
    mocked_args = F"""
    --model_name_or_path {model_name}
    --task_name mrpc
    --max_epochs 10
    --gpus 1
    --log_gpu_memory min_max"""
    args = parse_args(mocked_args.split())
    dm, model, trainer = main(args)
    train_start = time()
    trainer.fit(model, dm)
    train_duration = time() - train_start

    # batch_inputs = prepare_inputs(dm.tokenizer)
    out_model = model.model.eval().to('cuda:0')
    # out_model.save_pretrained('new_' + model_name)
    # with torch.jit.optimized_execution(True):
    #     traced_model = torch.jit.trace(out_model, batch_inputs[0].to('cuda:1'))
    #     torch.jit.save(traced_model, model_name + '.pt')
    # del out_model, model
    # torch.cuda.empty_cache()
    # memoryUsed = GPUtil.getGPUs()[1].memoryUsed
    # eval_start = time()
    # for inputs in batch_inputs:
    #     out_model(inputs.to('cuda:1'))
    # eval_duration = time() - eval_start
    result = {
        'model_name': model_name,
        'train_duration': train_duration,
        # 'memoryUsed': memoryUsed,
        # 'eval_duration': eval_duration,
        'eval_accuracy': trainer.logged_metrics['accuracy']
    }
    print(result)
    return result


if __name__ == '__main__':
    results = []
    model_names = [
        't5-base',
        # 't5-small',
        # 't5-large',
    ]
    generator_names = [
        'facebook/bart-base',
        'facebook/bart-large',
        'gpt2',
        'gpt2-medium',
        'distilgpt2',
        'funnel-transformer/small',
        'funnel-transformer/medium',
        'funnel-transformer/intermediate',
        'funnel-transformer/large',
        'funnel-transformer/xlarge',
    ]
    done = [
        'distilbert-base-uncased',
        'distilbert-base-cased',
        'distilbert-base-multilingual-cased',
        'albert-base-v2',
        'albert-large-v2',
        'albert-xlarge-v2',
        'albert-xxlarge-v2',
        'xlm-roberta-base',
        'xlm-roberta-large',
        'roberta-base',
        'roberta-large',
        'distilroberta-base',
        'squeezebert/squeezebert-uncased',
        'bert-base-uncased',
        'bert-large-uncased',
        'bert-base-multilingual-uncased',
        'xlnet-base-cased',
        'xlnet-large-cased',
        'google/mobilebert-uncased',
        'xlm-mlm-en-2048',
        'google/electra-small-discriminator',
        'google/electra-base-discriminator',
        'google/electra-large-discriminator',
        'microsoft/deberta-base',
        'microsoft/deberta-large',
    ]
    for model_name in model_names:
        results.append(run(model_name))
    for r in results:
        print(r)
