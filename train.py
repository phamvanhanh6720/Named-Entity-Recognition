import torch
import mlflow.pytorch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import MLFlowLogger

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
    from pytorch_lightning.loggers import MLFlowLogger

    # mlflow.pytorch.autolog()
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./mlruns")

    dm = NERDataModule(model_name_or_path='xlm-roberta-base',
                       dataset_path='dataset/all_data_v1_02t03.jsonl',
                       tags_list=tags_list,
                       label_all_tokens=False,
                       max_seq_length=128,
                       train_batch_size=32,
                       eval_batch_size=32)
    dm.setup(stage="fit")

    model = NERModel(model_name_or_path="xlm-roberta-base",
                     num_labels=dm.num_labels,
                     tags_list=dm.tags_list,
                     train_batch_size=32,
                     eval_batch_size=32)

    AVAIL_GPUS = min(1, torch.cuda.device_count())

    trainer = Trainer(max_epochs=20, gpus=AVAIL_GPUS, logger=mlf_logger)
    trainer.fit(model, datamodule=dm)