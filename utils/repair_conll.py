import string
from argparse import ArgumentParser


def write_conll(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for sq in data:
            for txt in sq:
                ww, tt = txt
                f.write(ww + '\t' + tt)
                f.write('\n')
            f.write('\n')


def convert_to_BItag(sq):
    out = []
    for data in sq:
        newtag = []
        for i in range(len(data)):
            if data[i][1] != 'O':
                if i == 0 or data[i - 1][1] != data[i][1]:
                    temp = 'B-' + data[i][1]
                elif i == len(data) - 1 or data[i + 1][1] != data[i][1]:
                    temp = 'I-' + data[i][1]
                else:
                    temp = 'I-' + data[i][1]
            else:
                temp = data[i][1]
            newtag.append((data[i][0], temp))
        out.append(newtag)
    return out


def is_beginning_end(word):
    if word[0] in string.punctuation:
        return 0, True
    if word[-1] in string.punctuation:
        return -1, True
    return "", False


def repair_setence(list_word_label):
    special_char_in_url = ["/", "?", "%", ".", "="]
    for idx, wd_lb in enumerate(list_word_label):
        wd = wd_lb[0]
        lb = wd_lb[1]
        special_char_idx, check = is_beginning_end(wd)
        if wd == "," and lb == "ADDRESS":
            continue
        if len(wd) >= 2:
            if check:
                if ((wd[special_char_idx] == "(" or wd[special_char_idx] == ")") and (
                        lb == "ORGANIZATION" or lb == "PRODUCT" or lb == "PHONENUMBER")) \
                        or ((wd[special_char_idx] == '"' or wd[special_char_idx] == '#') and lb == "EVENT") \
                        or (wd[special_char_idx] == "%" and lb == "QUANTITY") \
                        or (wd[special_char_idx] in special_char_in_url and lb == "URL") \
                        or (wd[special_char_idx] in ['(', ')', ','] and lb == "DATETIME") \
                        or (wd[special_char_idx] in [','] and lb == "ORGANIZATION") \
                        or (lb == "ADDRESS"):
                    continue
                elif (wd[special_char_idx] in ['.'] and lb == "PERSON"):
                    continue
                else:
                    split_list = []
                    del list_word_label[idx]
                    if special_char_idx == 0:
                        split_list.append([wd[special_char_idx], "O"])
                        split_list.append([wd[1:], lb])
                        list_word_label[idx:idx] = split_list
                    if special_char_idx == -1:
                        split_list.append([wd[0:-1], lb])
                        split_list.append([wd[-1], "O"])
                        list_word_label[idx:idx] = split_list

    return list_word_label


def convert(file_need_repair_path, output_file_path):
    with open(file_need_repair_path, "r") as f:
        data = f.read()

    extract_info_total_sentence = []
    total_sentence = data.split("\n\n")

    for sentence in total_sentence:
        extract_info_sentence = []
        word_label_sentence = sentence.split("\n")
        if sentence == "":
            continue
        for word_label in word_label_sentence:
            pair_word_label = word_label.split()
            word = pair_word_label[0]
            label = pair_word_label[1]
            label_without_bio = label if label == "O" else label.split("-")[1]
            extract_info_sentence.append([word, label_without_bio])

        extract_info_sentence = repair_setence(extract_info_sentence)
        extract_info_total_sentence.append(extract_info_sentence)

    result = convert_to_BItag(extract_info_total_sentence)
    write_conll(result, path=output_file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_old_data', type=str, default='old_data.txt')
    parser.add_argument('--path_new_data', type=str, default='new_data.txt')
    args = parser.parse_args()
    convert(
        file_need_repair_path=args.path_old_data,
        output_file_path=args.path_new_data
    )
