import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_language, tgt_language, seq_len)-> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_language = src_language
        self.tgt_language = tgt_language

        self.sos_token = torch.Tensor(tokenizer_src.token_to_id(['[SOS]']), dtype=torch.int64)
        self.eos_token = torch.Tensor(tokenizer_src.token_to_id(['[EOS]']), dtype=torch.int64)
        self.pad_token = torch.Tensor(tokenizer_src.token_to_id(['[PAD]']), dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_language]
        tgt_text = src_target_pair['translation'][self.tgt_language]

        