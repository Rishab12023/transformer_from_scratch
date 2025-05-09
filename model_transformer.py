import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        # print("🚨 Embedding input dtype:", x.dtype)
        # assert x.dtype == torch.long, f"Expected torch.long but got {x.dtype}"
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(seq_len, d_model)
        position =  torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        dinominator_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.pe[:,0::2] = torch.sin(position*dinominator_term)
        self.pe[:,1::2] = torch.cos(position*dinominator_term)

        self.pe = self.pe.unsqueeze(0) #--> this make it (1,seq_len, d_model)
        self.register_buffer('Positional_Encodings', self.pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, d_model: int=512, esp: float = 10**-6) -> None:
        super().__init__()

        self.esp = esp
        self.alpha = nn.Parameter(torch.ones(d_model))  ## Multiplied to scale the distribution
        self.bias = nn.Parameter(torch.zeros(d_model))  ## Added to shift the distribution

    def forward(self, x):
        mean = x.mean(dim= -1, keepdim=True)
        std = x.std(dim= -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.esp) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        o1 = torch.relu(self.linear_1(x))  ## this will give the dimension (batch, d_model(512), d_ll(1024))
        d1 = self.dropout(o1)
        o2 = self.linear_2(d1)
        return o2
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float)->None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # print(f"here are the dimension of model {d_model} {h}")
        assert d_model %h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  ## Query Matrix
        self.w_k = nn.Linear(d_model, d_model)  ## Key Matrix
        self.w_v = nn.Linear(d_model, d_model)  ## Value Matrix

        self.w_o = nn.Linear(d_model, d_model)   ## --> since MultiHead(Q,K,V) = Concat( head1, head2...., head_h) * ( W_o )
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def compute_attention_scr(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        ## Key Dimension is (Batch, h, Seq_Len, d_k) ---> and when we multiply q*k then we get a matrix of seq * seq (Attention Score that each word give to every other word in the sequnece)
        ## Therefore we transpose the key and then multiply.

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask ==0, -1e9) ## Mask that will be used in case of Decoder Part.
        if dropout is not None:
            attention_score = dropout(attention_score)
        
        return (attention_score @ value), attention_score
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)  ## (Batch, Seq_Len, d_model) --After Matrix Multiplication--> (Batch, Seq_Len, d_model)
        key = self.w_k(k) ## (Batch, Seq_Len, d_model) --After Matrix Multiplication--> (Batch, Seq_Len, d_model)
        value = self.w_v(v) ## (Batch, Seq_Len, d_model) --After Matrix Multiplication--> (Batch, Seq_Len, d_model)


        # (Batch, Seq_Len, d_model) --Dividing Each Embedding into h for h heads--> (Batch, Seq_Len, h, d_k) --Rearranging Dimensions--> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) 

        x, self.attention_scores = MultiHeadAttentionBlock.compute_attention_scr(query, key, value, mask, self.dropout) ## Since compute_attention_scr is a static method its called using Class Name

        ## Since we have (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len,h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        ## In the paper its the output of each sub-layer is LayerNorm(x + Sublayer(x))
        ## We have taken Sublayer first and then LayerNormed it
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block(x))
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x , encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward_block(x))
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_maks, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_maks, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        ## Since the output from the decoder will be (batch, Seq_Len, d_model) --to project to vocab size--> (batch, Seq_Len ,vocab_size)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):

    def __init__(self, encoder:Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int=2048
)->Transformer:
    
    ## Input Token Embeddings
    src_embed = InputEmbeddings(d_model, src_vocab_size) ## Since in machine translation task there could be different number of words in the vocab of src & tgt language
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    ## Positional Encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    ## Creates the Encoder Block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    ## Creates the Decoder Block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    ## Create the Encoder and the Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    ## Create the Projection Layer 
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    ## Creates The TRANSFORMER
    transfomer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)


    ## Initilize with Xavier Initialization
    for p in transfomer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transfomer

