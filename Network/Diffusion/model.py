import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

from typing import Dict

from nn.prenet import EncoderEmbed, DecoderEmbed, PredictionHead


def masked_softmax(scores, mask=None):
    scores = torch.clamp(scores, min=-50, max=50)
    exp_scores = torch.exp(scores)
    if mask != None:
        exp_scores = exp_scores.masked_fill(mask == False, 0.0)
    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
    sum_exp_scores = sum_exp_scores.masked_fill(sum_exp_scores <= 0.0, 1.0)
    softmaxed = exp_scores / sum_exp_scores
    return softmaxed


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, token_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert token_dim % num_heads == 0
        self.d_head = token_dim // num_heads
        self.num_heads = num_heads

        # Q, K, V linear layers
        self.W_q = nn.Linear(token_dim, token_dim, bias=False)
        self.W_k = nn.Linear(token_dim, token_dim, bias=False)
        self.W_v = nn.Linear(token_dim, token_dim, bias=False)
        
        # Output linear layer
        self.W_o = nn.Linear(token_dim, token_dim, bias=False)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear transformations
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.d_head**0.5
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(3).expand_as(scores)
        attn_weights = masked_softmax(scores, mask)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        # Linear transformation to output
        return self.W_o(attn_output)

class Decoder(nn.Module):
    def __init__(self, token_dim, num_heads, dropout, checkpoint):
        super(Decoder, self).__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadCrossAttention(token_dim, num_heads)
        # Multi-head cross-attention
        self.cross_attn = MultiHeadCrossAttention(token_dim, num_heads)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Linear(4 * token_dim, token_dim)
        )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.norm3 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

        self.checkpoint = checkpoint

    def forward(self, main_seq, condition_seq, mask):
        # Self Attention
        attn_output = self.self_attn(main_seq, main_seq, main_seq, mask)
        x = main_seq + self.dropout(attn_output)
        x = self.norm1(x)

        # Cross Attention
        attn_output = self.cross_attn(x, condition_seq, condition_seq)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)

        # Feed Forward Network
        if self.checkpoint: ff_output = checkpoint(self.feed_forward, x)
        else: ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, token_dim, num_heads, dropout, checkpoint):
        super(Encoder, self).__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadCrossAttention(token_dim, num_heads)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Linear(4 * token_dim, token_dim)
        )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

        self.checkpoint = checkpoint

    def forward(self, condition_seq, mask):
        # Self Attention
        attn_output = self.self_attn(condition_seq, condition_seq, condition_seq, mask)
        x = condition_seq + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed Forward Network
        if self.checkpoint: ff_output = checkpoint(self.feed_forward, x)
        else: ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x
    
class NaviNetwork(nn.Module):
    def __init__(self, num_encoder, num_decoder, **config):
        super(NaviNetwork, self).__init__()

        self.token_dim = config.get('token_dim', 512)
        self.encoder_embed = EncoderEmbed(self.token_dim)
        self.decoder_embed = DecoderEmbed(self.token_dim)

        self.encoders = nn.ModuleList()
        for _ in range(num_encoder): self.encoders.append(Encoder(**config))
        self.decoders = nn.ModuleList()
        for _ in range(num_decoder): self.decoders.append(Decoder(**config))

        self.prediction_head = PredictionHead(self.token_dim)

    @autocast()
    def encoder_forward(self, **input):
        tokens, mask = self.encoder_embed(**input)
        for i in range(len(self.encoders)):
            tokens = self.encoders[i](tokens, mask)
        return tokens
    
    @autocast()
    def decoder_forward(self, encoder_tokens, **input):
        tokens, mask = self.decoder_embed(**input)
        for i in range(len(self.decoders)):
            tokens = self.decoders[i](tokens, encoder_tokens, mask)
        return tokens
    
    @autocast()
    def forward(self, encoder_input : Dict[str, torch.Tensor], decoder_input : Dict[str, torch.Tensor]):
        encoder_tokens = self.encoder_forward(**encoder_input)
        decoder_tokens = self.decoder_forward(encoder_tokens, **decoder_input)
        return self.prediction_head(decoder_tokens)
    
    @autocast()
    def fix_encoder(self, path):
        checkpoint = torch.load(path)
        encoder_state_dicts = [{k[15:]: v for k, v in checkpoint.items() if k.startswith(f'net.encoders.{i}.')}for i in range(len(self.encoders))]
        for i in range(len(self.encoders)):
            self.encoders[i].load_state_dict(encoder_state_dicts[i])
            self.encoders[i].requires_grad_(False)
        encoder_embed_state_dicts = {k[18:]: v for k, v in checkpoint.items() if k.startswith('net.encoder_embed.')}
        self.encoder_embed.load_state_dict(encoder_embed_state_dicts)
        self.encoder_embed.requires_grad_(False)
    
class NaviDiffusion(nn.Module):
    def __init__(self, **config):
        super(NaviDiffusion, self).__init__()
        self.net = NaviNetwork(**config)
        self.cache_encoder_tokens = None

    def eval(self):
        self.cache_encoder_tokens = None
        super(NaviDiffusion, self).eval()

    @autocast()
    def forward(self, x, t, c, c_mask):
        decoder_input = x
        decoder_input['time'] = t
        decoder_input['condition_mask'] = c_mask
        encoder_input = c

        return self.net.forward(encoder_input, decoder_input)
        
    def sample(self, x, t, c, c_mask):
        decoder_input = x
        decoder_input['time'] = t
        decoder_input['condition_mask'] = c_mask
        encoder_input = c

        if self.cache_encoder_tokens is None:
            self.cache_encoder_tokens = self.net.encoder_forward(**encoder_input)
        decoder_tokens = self.net.decoder_forward(self.cache_encoder_tokens, **decoder_input)
        return self.net.prediction_head(decoder_tokens)
    
    @autocast()
    def fix_encoder(self, path):
        return self.net.fix_encoder(path)
    

