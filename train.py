from argparse import ArgumentParser

import torch
import mlflow.pytorch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from model import NERModel
from dataset import NERDataModule


import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    seed_everything(43)

    tags_list = ["B-ADDRESS", "I-ADDRESS",
                 "B-SKILL", "I-SKILL",
                 "B-EMAIL", "I-EMAIL",
                 "B-PERSON", "I-PERSON",
                 "B-PHONENUMBER", "I-PHONENUMBER",
                 "B-QUANTITY", "I-QUANTITY",
                 "B-PERSONTYPE", "I-PERSONTYPE",
                 "B-ORGANIZATION", "I-ORGANIZATION",
                 "B-PRODUCT", "I-PRODUCT",
                 "B-IP", 'I-IP',
                 "B-LOCATION", "I-LOCATION",
                 "O",
                 "B-DATETIME", "I-DATETIME",
                 "B-EVENT", "I-EVENT",
                 "B-URL", "I-URL"]

    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)

    parser.add_argument('--model_name_or_path', type=str, default='xlm-roberta-base')
    parser.add_argument('--dataset_path', type=str, default='dataset/all_data_v1_02t03.jsonl')
    parser.add_argument('--label_all_tokens', type=bool, default=False)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=15)

    # model specific arguments
    parser.add_argument('--use_crf', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()
    # mlflow.pytorch.autolog()
    mlf_logger = MLFlowLogger(experiment_name="fpt_ner_logs",
                              tracking_uri="file:./mlruns",
                              run_name=parser.run_name)

    dm = NERDataModule(model_name_or_path=args.model_name_or_path,
                       dataset_path=args.dataset_path,
                       tags_list=tags_list,
                       label_all_tokens=args.label_all_tokens,
                       max_seq_length=args.max_seq_length,
                       train_batch_size=args.train_batch_size,
                       eval_batch_size=args.eval_batch_size)
    dm.setup(stage="fit")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_overall_f1',
        dirpath='checkpoints/' + args.run_name,
        filename='{epoch:02d}--{val_overall_f1:.2f}',
        save_top_k=3,
        mode="max",
        save_weights_only=True)

    model = NERModel(model_name_or_path=dm.model_name_or_path,
                     num_labels=dm.num_labels,
                     tags_list=dm.tags_list,
                     train_batch_size=dm.train_batch_size,
                     eval_batch_size=dm.eval_batch_size,
                     use_crf=args.use_crf,
                     learning_rate=args.learning_rate,
                     adam_epsilon=args.adam_epsilon,
                     warmup_steps=args.warmup_steps,
                     weight_decay=args.weight_decay)

    AVAIL_GPUS = min(1, torch.cuda.device_count())

    trainer = Trainer(max_epochs=args.num_epochs, gpus=AVAIL_GPUS, logger=mlf_logger)
    trainer.fit(model, datamodule=dm)

    mlf_logger.experiment.log_artifact(mlf_logger.run_id, 'checkpoints/' + args.run_name)