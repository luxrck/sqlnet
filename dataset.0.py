import csv
from functools import partial

from torchtext.data import *
from transformers import BertTokenizer


bert_tokenizer = None


def sst_dataset(root="SST-2", tokenizer_name="builtin", bert_pretrained_model="bert-base-uncased", batch_size=64, batch_first=False, padding_to=0, sort_within_batch=True, device="cpu"):
    def padding(batch, vocab, to=padding_to):
        # import pdb; pdb.set_trace()
        if not padding_to or padding_to <= 0 or tokenizer_name == "bert":
            return batch
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < padding_to:
                batch[idx] = ex + [pad_idx] * (padding_to - len(ex))
        return batch
    
    def tokenizer(text=""):
        name = tokenizer_name
        if name == "builtin":
            return text.strip().split()
        elif name == "bert":
            global bert_tokenizer
            if not bert_tokenizer:
                bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
            tokens = bert_tokenizer.encode_plus(text, add_special_tokens=True, return_attention_mask=True, max_length=padding_to, pad_to_max_length=True)
            # import pdb; pdb.set_trace()
            return [tokens["input_ids"], tokens["attention_mask"]]

    # import pdb; pdb.set_trace()
    if tokenizer_name == "bert":
        global bert_tokenizer
        if not bert_tokenizer:
            bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        # eos_token = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.eos_token)
        eos_token = None
        pad_token = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.pad_token)
        unk_token = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.unk_token)
        print(f"unk: {unk_token} pad: {pad_token}")
        use_vocab = False
    else:
        eos_token = None
        pad_token = "<pad>"
        unk_token = '<unk>'
        use_vocab = True
    
    TEXT = Field(include_lengths=True,
                init_token=None,
                eos_token=None,
                pad_token=pad_token,
                unk_token=unk_token,
                use_vocab=use_vocab,
                lower=None,
                batch_first=batch_first,
                postprocessing=partial(padding, to=padding_to),
                tokenize=tokenizer)
    LABEL = Field(sequential=False, unk_token=None)
    _train, _test = TabularDataset.splits(path="data/" + root, root="data", train="train.tsv", test="test.tsv",
                                    format='csv', skip_header=False, fields=[("label", LABEL), ("text", TEXT)],
                                    csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})
    # import pdb; pdb.set_trace()
    if tokenizer_name == "builtin":
        TEXT.build_vocab(_train.text, _train.label, min_freq=1)
    LABEL.build_vocab(_train)

    sort_key = lambda x: len(x.text)
    train_iter = BucketIterator(_train,
                                 batch_size=batch_size,
                                 train=True,
                                 repeat=False,
                                 shuffle=True,
                                 sort_within_batch=sort_within_batch,
                                 sort_key=(sort_key if sort_within_batch else None),
                                 device=device)
    test_iter = BucketIterator(_test, batch_size=1, train=False, repeat=False, shuffle=True,
            sort_within_batch=False, sort_key=lambda x: len(x.text), device=device)

    return train_iter, test_iter, TEXT, LABEL
