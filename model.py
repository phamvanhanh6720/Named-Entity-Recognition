from typing import List

import torch
import datasets
from pytorch_lightning import LightningModule
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)


class NERModel(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        tags_list: List[str],
        use_crf: bool = False,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.tags_list = tags_list
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metrics = datasets.load_metric('seqeval')

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=-1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):

        predictions = []
        labels = []
        for x in outputs:
            predictions.extend(x['preds'].detach().cpu().numpy().tolist())
            labels.extend(x['labels'].detach().cpu().numpy().tolist())

        # print(predictions)
        # print(labels)

        true_predictions = [
            [self.tags_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.tags_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)

        results = self.metrics.compute(predictions=true_predictions, references=true_labels)
        refactor_results = {}
        for key in results.keys():
            if 'overall' in key:
                refactor_results['val_' + key] = results[key]
            else:
                refactor_results['val_' + key + '_f1'] = results[key]['f1']

        self.log_dict(refactor_results)

        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = tb_size * self.trainer.accumulate_grad_batches
        self.total_steps = int((len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]