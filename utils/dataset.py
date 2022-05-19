import json
import ast
import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import cohen_kappa_score


class NERDataSet:
    def __init__(self, jsonl_file, entity_names=[]):
        self.entity_names = ['PERSONTYPE', 'LOCATION', 'PHONENUMBER', 'EMAIL',
                             'PRODUCT', 'URL', 'ORGANIZATION', 'DATETIME',
                             'QUANTITY', 'ADDRESS', 'PERSON', 'SKILL',
                             'EVENT', 'MISCELLANEOUS', 'IP']

        self.json_list = read_annotation_file(jsonl_file=jsonl_file)
        self.all_entities = []
        for json in self.json_list:
            self.all_entities.append(self.extract_entities(data_point=json))

        self.labels_dict = {'O': ['O', 'O']}
        for name in self.entity_names:
            self.labels_dict[name] = ['B-{}'.format(name), 'I-{}'.format(name)]

        self._to_conll_list()
        self._build_df()

    def _build_df(self):
        dataset = []
        for i in range(len(self.conll_list)):
            conll_label = self.conll_list[i]

            json_data = self.json_list[i]
            id_sen = json_data['id']
            text = json_data['data']
            ref_id = json_data['RED_ID'] if 'RED_ID' in json_data.keys() else None
            source = json_data['SOURCE'] if 'SOURCE' in json_data.keys() else None

            dataset.append([id_sen, ref_id, source, text, conll_label])

        self.dataset_df = pd.DataFrame(data=dataset,
                                       columns=['id', 'ref_id', 'source', 'text', 'conll_label'])

    def _to_conll_list(self):

        self.conll_list: List[List[str, str]] = []
        for entities_list in self.all_entities:
            syllable_level = []
            for entity in entities_list:
                text = entity[0]
                label = entity[1]

                if label not in self.entity_names:
                    label = 'O'

                words = text.split(' ')
                words = [word for word in words if word != '']

                syllable_level.append(words[0].strip(' \n') + ' ' + self.labels_dict[label][0])
                for word in words[1:]:
                    syllable_level.append(word.strip(' \n') + ' ' + self.labels_dict[label][1])

            self.conll_list.append(syllable_level)

    @staticmethod
    def extract_spans(labels_list: list, text_len):
        spans_list = []

        if not len(labels_list):
            spans_list.append([0, text_len, 'O'])
            return spans_list

        if len(labels_list) == 1:
            first_point = labels_list[0][0]
            second_point = labels_list[0][1]
            if first_point > 0:
                spans_list.append([0, first_point, 'O'])
            if second_point < text_len:
                spans_list.append(labels_list[0])
                spans_list.append([second_point, text_len, 'O'])
            else:
                spans_list.append(labels_list[0])

            return spans_list

        for i in range(0, len(labels_list) - 1):
            span = labels_list[i]
            next_label = labels_list[i + 1]

            first_point = span[0]
            second_point = span[1]
            third_point = next_label[0]

            if first_point > 0 and i == 0:
                spans_list.append([0, first_point, 'O'])

            spans_list.append(span)
            if third_point <= second_point:
                continue
            else:
                spans_list.append([second_point, third_point, 'O'])

        last_tag = labels_list[-1]
        if last_tag[1] < text_len:
            spans_list.append(last_tag)
            spans_list.append([last_tag[1], text_len, 'O'])
        else:
            spans_list.append(last_tag)

        return spans_list

    def extract_entities(self, data_point: dict) -> List[List[str]]:
        entities_list = []
        text = data_point['data']
        labels_list: list = data_point['label']
        sen_id = data_point['id']
        source = data_point['SOURCE'] if 'SOURCE' in data_point.keys() else None

        labels_list.sort(key=lambda x: x[0], reverse=False)

        spans_list = self.extract_spans(labels_list=labels_list,
                                        text_len=len(text))

        for span in spans_list:
            entity = text[span[0]: span[1]]
            label = span[2]
            entity = entity.strip(' \n')
            if entity in ['', ' ']:
                continue
            else:
                entities_list.append([entity, label, sen_id, source])

        return entities_list


def read_annotation_file(jsonl_file: str) -> List[dict]:
    json_list = []

    with open(jsonl_file, 'r') as file:
        lines_list = file.readlines()
        for line in lines_list:
            item = json.loads(line)
            if isinstance(item, dict):
                if 'ID' in list(item.keys()):
                    item['id'] = item['ID']
                    item.pop('ID', None)
                json_list.append(item)
            elif isinstance(item, str):
                item = ast.literal_eval(item)
                if 'ID' in list(item.keys()):
                    item['id'] = item['ID']
                    item.pop('ID', None)
                json_list.append(ast.literal_eval(item))

    json_list.sort(key=lambda x: x['id'], reverse=False)

    return json_list


def compare_annotations(jsonl_file1, jsonl_file2):
    json_list1 = read_annotation_file(jsonl_file1)
    json_list2 = read_annotation_file(jsonl_file2)

    id_list = []

    if len(json_list1) != len(json_list2):
        raise Exception('The number of sentences in 2 files is different')

    for i in range(len(json_list1)):
        sen1: dict = json_list1[i]
        sen2: dict = json_list2[i]

        if sen1['ID'] != sen2['ID']:
            raise Exception('The content of 2 files is difference. File {} vs {}'.format(sen1['ID'], sen2['ID']))

        id_sen = sen1['ID']

        labels_1: list = sen1['label']
        labels_2: list = sen2['label']

        # Sort following start idx of span
        labels_1.sort(key=lambda x: x[0], reverse=False)
        labels_2.sort(key=lambda x: x[0], reverse=False)

        if len(labels_1) != len(labels_2):
            id_list.append(id_sen)
            continue

        for span_id in range(len(labels_1)):
            span1 = labels_1[span_id]
            span2 = labels_2[span_id]

            entity_type1 = span1[2]
            entity_type2 = span2[2]
            if entity_type1 != entity_type2:
                id_list.append(id_sen)
                break

            start_idx1 = span1[0]
            start_idx2 = span2[0]
            if start_idx1 != start_idx2:
                id_list.append(id_sen)
                break

            end_idx1 = span1[1]
            end_idx2 = span2[1]
            if end_idx1 != end_idx2:
                id_list.append(id_sen)
                break

    return id_list


def cal_cohen_kappa(jsonl_file1, jsonl_file2):
    dataset1 = NERDataSet(jsonl_file=jsonl_file1)
    dataset2 = NERDataSet(jsonl_file=jsonl_file2)

    labels_dict = dataset1.labels_dict
    unique_labels = []
    for key in labels_dict.keys():
        unique_labels.extend(labels_dict[key])

    unique_labels = list(set(unique_labels))
    labels2id = {}
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        labels2id[label] = i

    conll_list1 = dataset1.to_conll_list()
    conll_list2 = dataset2.to_conll_list()

    if len(conll_list1) != len(conll_list2):
        raise Exception('2 datasets have different size')

    count = 0

    scores_list = []
    for i in range(len(conll_list1)):
        labels1 = [word.split(' ')[-1] for word in conll_list1[i]]
        labels1 = [labels2id[tag] for tag in labels1]
        labels2 = [word.split(' ')[-1] for word in conll_list2[i]]
        labels2 = [labels2id[tag] for tag in labels2]

        if len(labels1) != len(labels2):
            print("{}th sentences is not valid".format(i))
            count += 1
            continue

        if len(labels1) == 0:
            continue

        cohen_score = cohen_kappa_score(np.array(labels1), np.array(labels2))
        scores_list.append(cohen_score)

    avg_cohen_kappa = sum(scores_list) / len(scores_list)

    return avg_cohen_kappa, count
