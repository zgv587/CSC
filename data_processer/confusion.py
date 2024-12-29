from collections import defaultdict


def load_confusion(filepath, tokenizer=None, mode="utf-8"):
    confusion_set = defaultdict(list)

    with open(filepath, "r", encoding=mode) as f:
        for line in f.readlines():
            char, confusion_char = line.strip().split(":")
            if tokenizer:
                convert = tokenizer.convert_tokens_to_ids
                confusion_set[convert(char)].extend(
                    convert(list(confusion_char)+[char])
                )
            else:
                confusion_set[char].extend(list(confusion_char)+[char])

    return confusion_set
