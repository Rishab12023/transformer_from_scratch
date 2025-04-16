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
        self.max_seq_len = seq_len
        self.sos_token = torch.tensor(tokenizer_src.token_to_id('[SOS]'))
        self.eos_token = torch.tensor(tokenizer_src.token_to_id('[EOS]'))
        self.pad_token = torch.tensor(tokenizer_src.token_to_id('[PAD]'))
       
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_language]
        tgt_text = src_target_pair['translation'][self.tgt_language]

        enc_input_token = self.tokenizer_src.encode(src_text).ids
        dec_input_token = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_token = self.max_seq_len - len(enc_input_token) - 2   ## Using 2 here since we have a SOS and EOS token in encoder input
        dec_num_padding_token = self.max_seq_len - len(dec_input_token) - 1   ## Using 1 here since we will have SOS Token but not EOS token in decoder input
 
        if enc_num_padding_token < 0 or dec_num_padding_token < 0:
            raise ValueError("Sentence is too long. Excceds Context Window of the Model")
        
        encoder_input = torch.cat(
            [
                torch.tensor([self.sos_token], dtype=torch.int64),
                torch.tensor(enc_input_token, dtype=torch.int64),
                torch.tensor([self.eos_token], dtype=torch.int64),
                torch.tensor([self.pad_token] * enc_num_padding_token, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token.unsqueeze(0),
                torch.tensor(dec_input_token, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_token, dtype=torch.int64)   
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_token, dtype=torch.int64),
                self.eos_token.unsqueeze(0),
                torch.tensor([self.pad_token] * dec_num_padding_token,  dtype=torch.int64)
            ]
        )
      
        assert encoder_input.size(0) == self.max_seq_len
        assert decoder_input.size(0) == self.max_seq_len
        assert label.size(0) == self.max_seq_len


        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  ## 1 to Seq_len
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    

def causal_mask(size: int):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int64)
    return mask == 0
    
