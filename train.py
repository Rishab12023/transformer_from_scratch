import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]
    
def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    src_language = config['language_src']
    tgt_language = config['language_tgt']
    ds_raw = load_dataset("opus_books", f"{src_language}-{tgt_language}", split="train")

    ## Build Tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, src_language)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, tgt_language)

    ## Spliting Dataset
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len((ds_raw)) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw,[train_ds_size, val_ds_size])