from typing import Optional

import pandas as pd
from typing import List

import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from pytorch_lightning.core import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from utils.dataset import NERDataSet


class CustomDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 model_name_or_path: str,
                 tags_list: List[str],
                 max_seq_length: int = 128,
                 label_all_tokens: bool = False):

        self.max_seq_length = max_seq_length
        self.label_all_tokens = label_all_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.tag2id = {}
        for i in range(len(tags_list)):
            self.tag2id[tags_list[i]] = i

        self.dataset = []
        for i in range(len(df)):
            conll_format = df.iloc[i]['conll_label']
            tokenized_inputs = self._tokenize_and_align_labels(data_point=conll_format)
            self.dataset.append(tokenized_inputs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        labels = item['labels']

        # return input_ids, attention_mask, labels
        return item

    def _tokenize_and_align_labels(self, data_point: List[str]):
        word_list = [word.split(' ')[0] for word in data_point]
        label_list = [word.split(' ')[-1] for word in data_point]

        tokenized_inputs = self.tokenizer(word_list,
                                          truncation=True,
                                          is_split_into_words=True,
                                          max_length=self.max_seq_length,
                                          padding='max_length')
        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(self.tag2id[label_list[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(self.tag2id[label_list[word_idx]] if self.label_all_tokens else -100)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = torch.LongTensor(label_ids)
        tokenized_inputs['input_ids'] = torch.LongTensor(tokenized_inputs['input_ids'])
        tokenized_inputs['attention_mask'] = torch.LongTensor(tokenized_inputs['attention_mask'])
        return tokenized_inputs


class NERDataModule(LightningDataModule):

    def __init__(self,
                 model_name_or_path: str,
                 dataset_path: str,
                 tags_list: List[str],
                 label_all_tokens: bool = False,
                 max_seq_length: int = 128,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 test_size: float = 0.1,
                 val_size: float = 0.2,
                 **kwargs):
        super().__init__()

        self.val_data = None
        self.train_data = None
        self.dataset_df = None
        self.test_data = None

        self.model_name_or_path = model_name_or_path
        self.dataset_path = dataset_path
        self.tags_list = tags_list
        self.num_labels = len(tags_list)
        self.label_all_tokens = label_all_tokens

        self.val_size = val_size
        self.test_size = test_size

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        ner_dataset = NERDataSet(jsonl_file=self.dataset_path)
        self.dataset_df = ner_dataset.dataset_df

        # TODO
        # Write to conll format file
        # train, val, test split
        df_train, df_rest = train_test_split(self.dataset_df,
                                             shuffle=True,
                                             random_state=43,
                                             stratify=self.dataset_df[['source']],
                                             train_size=1 - self.val_size - self.test_size)

        df_val, df_test = train_test_split(df_rest,
                                           shuffle=True,
                                           random_state=43,
                                           stratify=df_rest[['source']],
                                           train_size=self.val_size / (self.val_size + self.test_size))

        self.train_data = CustomDataset(df=df_train,
                                        model_name_or_path=self.model_name_or_path,
                                        tags_list=self.tags_list,
                                        max_seq_length=self.max_seq_length,
                                        label_all_tokens=self.label_all_tokens)

        self.val_data = CustomDataset(df=df_val,
                                      model_name_or_path=self.model_name_or_path,
                                      tags_list=self.tags_list,
                                      max_seq_length=self.max_seq_length,
                                      label_all_tokens=self.label_all_tokens)

        self.test_data = CustomDataset(df=df_test,
                                       model_name_or_path=self.model_name_or_path,
                                       tags_list=self.tags_list,
                                       max_seq_length=self.max_seq_length,
                                       label_all_tokens=self.label_all_tokens)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.eval_batch_size)

    def predict_dataloader(self):
        pass


if __name__ == '__main__':
    from dataset import NERDataModule

    # seed_everything(43)

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

    # mlflow.pytorch.autolog(log_every_n_epoch=1)

    dm = NERDataModule(model_name_or_path='xlm-roberta-base',
                       dataset_path='dataset/all_data_v1_02t03.jsonl',
                       tags_list=tags_list,
                       max_seq_length=128,
                       train_batch_size=32,
                       eval_batch_size=32)
    dm.setup(stage="fit")
    print(1)