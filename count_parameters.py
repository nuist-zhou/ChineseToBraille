#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

d_model = 768
nhead = 8
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
moe_n_experts = 4

class MoEGate(nn.Module):
    def __init__(self, d_model: int, n_experts: int, noisy: bool = True):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts, bias=False)
        self.noisy = noisy

    def forward(self, x):
        scores = self.proj(x)
        if self.noisy and self.training:
            scores = scores + torch.randn_like(scores) * 1e-2
        prob = torch.softmax(scores, dim=-1)
        top1_idx = prob.argmax(-1)
        top1_prob = prob.gather(-1, top1_idx.unsqueeze(-1)).squeeze(-1)
        return top1_idx, top1_prob, prob


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, aux_loss_coef: float = 1e-2):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(n_experts)])
        self.gate = MoEGate(d_model, n_experts, noisy=True)
        self.aux_loss_coef = aux_loss_coef
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        S, N, D = x.shape
        top1_idx, top1_prob, prob = self.gate(x)
        usage = prob.mean(dim=(0, 1))
        self.last_aux_loss = self.aux_loss_coef * (usage * usage.numel()).var()

        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = (top1_idx == e)
            if mask.any():
                xe = x[mask]
                ye = expert(xe)
                y[mask] = ye * top1_prob[mask].unsqueeze(-1)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        T = x.size(0)
        return x + self.pe[:T]


class BrailleTranslatorMoE(nn.Module):
    def __init__(self, braille_vocab_size: int, n_experts: int = 4):
        super().__init__()
        bert_config = BertConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12, 
                                 num_attention_heads=12, intermediate_size=3072)
        self.bert = BertModel(bert_config)
        self.moe = MoEFFN(d_model=d_model, d_ff=dim_feedforward, n_experts=n_experts, aux_loss_coef=1e-2)

        self.braille_embedding = nn.Embedding(braille_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=128)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Linear(d_model, braille_vocab_size)

    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, input_ids, attention_mask, braille_ids):
        enc = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        enc = enc.transpose(0, 1)
        enc = self.moe(enc)

        tgt = self.braille_embedding(braille_ids).transpose(0, 1)
        tgt = self.positional_encoding(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
        pad_id = 0
        tgt_key_padding_mask = (braille_ids == pad_id)
        mem_key_padding_mask = ~attention_mask.bool()

        dec = self.decoder(
            tgt=tgt,
            memory=enc,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_padding_mask
        )
        out = self.output_layer(dec).transpose(0, 1)
        return out


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_parameters_by_component(model):
    components = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        components[name] = params
    return components


def main():
    print("Loading BERT model...")
    braille_vocab_size = 100
    model = BrailleTranslatorMoE(braille_vocab_size=braille_vocab_size, n_experts=moe_n_experts)
    model = model.to(device)
    
    print("\n" + "="*60)
    print("MODEL PARAMETER COUNT")
    print("="*60)
    
    total, trainable = count_parameters(model)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    
    print("\n" + "-"*60)
    print("PARAMETERS BY COMPONENT")
    print("-"*60)
    
    components = count_parameters_by_component(model)
    for name, params in components.items():
        print(f"{name:25s}: {params:>15,} parameters ({params/total*100:5.2f}%)")
    
    print("\n" + "-"*60)
    print("MoE EXPERTS DETAILS")
    print("-"*60)
    
    for i, expert in enumerate(model.moe.experts):
        expert_params = sum(p.numel() for p in expert.parameters())
        print(f"Expert {i}: {expert_params:>12,} parameters")
    
    gate_params = sum(p.numel() for p in model.moe.gate.parameters())
    print(f"Gate: {gate_params:>15,} parameters")
    
    print("\n" + "-"*60)
    print("TRANSFORMER DECODER DETAILS")
    print("-"*60)
    
    for i, layer in enumerate(model.decoder.layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"Decoder Layer {i}: {layer_params:>12,} parameters")
    
    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    print(f"d_model: {d_model}")
    print(f"nhead: {nhead}")
    print(f"num_decoder_layers: {num_decoder_layers}")
    print(f"dim_feedforward: {dim_feedforward}")
    print(f"moe_n_experts: {moe_n_experts}")
    print(f"dropout: {dropout}")
    print(f"braille_vocab_size: {braille_vocab_size}")
    
    print("\n" + "="*60)
    print(f"Total Model Parameters: {total:,}")
    print(f"Model Size (FP32): {total * 4 / 1024 / 1024:.2f} MB")
    print("="*60)


if __name__ == "__main__":
    main()
