import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MBartConfig
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.mbart.modeling_mbart import MBartAttention, MBartPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

class AttentionLayer(nn.Module):
    
    def __init__(self, config: MBartConfig, variational_cfg: dict):
        super().__init__()
        self.embed_dim = config.d_model
        self.dropout = variational_cfg.get('dropout', config.dropout)#config.dropout
        
        self.attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Cross-Attention
        self.attn = MBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )#config.attention_dropout
        
        self.ffn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Feed-Forward Network
        ffn_dim = variational_cfg.get('ffn_dim', self.embed_dim * 4)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(ffn_dim, self.embed_dim),
            nn.Dropout(self.dropout)
        )
    
    def forward(
        self,
        query_states: torch.Tensor,           
        key_value_states: torch.Tensor,       
        query_mask: Optional[torch.Tensor] = None,
        key_value_mask: Optional[torch.Tensor] = None,
    ):
        # Expand masks
        if key_value_mask is not None:
            key_value_mask = _expand_mask(
                key_value_mask, 
                query_states.dtype, 
                tgt_len=query_states.shape[1]
            )
    
        residual = query_states
        
        normed_query = self.attn_layer_norm(query_states)
       
        attn_output, _, _ = self.attn(
            hidden_states=normed_query,  
            key_value_states=key_value_states,
            attention_mask=key_value_mask,
            output_attentions=False,
        )
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        
        query_states = residual + attn_output
        
        residual = query_states
        
        normed_query = self.ffn_layer_norm(query_states)
        
        ffn_output = self.ffn(normed_query)
        
        query_states = residual + ffn_output
        
        return query_states


class MultiLayerAttention(nn.Module):

    def __init__(self, config: MBartConfig, variational_cfg: dict, num_layers: int = 5):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            AttentionLayer(config, variational_cfg)
            for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
        print(f"   MultiLayerAttention初始化: {num_layers}层 (Pre-LN架构)")
    
    def forward(
        self,
        query_embeds: torch.Tensor,
        key_value_embeds: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_value_mask: Optional[torch.Tensor] = None,
    ):

        hidden_states = query_embeds
        
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                query_states=hidden_states,
                key_value_states=key_value_embeds,
                query_mask=query_mask,
                key_value_mask=key_value_mask,
            )
        
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states



class TwoLayerTransformerNet(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_proj = nn.Linear(embed_dim, latent_dim * 2)
        
        print(f"      TwoLayerTransformerNet初始化: 2层Transformer Block + 投影层")
        print(f"      输入维度: {embed_dim}, 输出维度: {latent_dim}*2 (mean+logvar)")
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        transformer_out = self.transformer(
            hidden_states,
            src_key_padding_mask=src_key_padding_mask
        )
        
        params = self.output_proj(transformer_out)  
        mean, logvar = params.chunk(2, dim=-1)      
        
        return mean, logvar

class GaussianNet(nn.Module):
    
    
    def __init__(self, variational_cfg, variator_cfg):
        super().__init__()

        self.in_out_dim = variational_cfg['input_embed_dim']
        self.latent_dim = variational_cfg['latent_dim']
        
        self.num_layers = variational_cfg.get('num_attention_layers', 5)
        
        self.prior_net = TwoLayerTransformerNet(
            embed_dim=self.in_out_dim,
            latent_dim=self.latent_dim,
            num_heads=variational_cfg.get('num_heads', 8),
            dropout=variational_cfg.get('dropout', 0.1)
        )
        
        self.posterior_net = TwoLayerTransformerNet(
            embed_dim=self.in_out_dim,
            latent_dim=self.latent_dim,
            num_heads=variational_cfg.get('num_heads', 8),
            dropout=variational_cfg.get('dropout', 0.1)
        )
        

        self.recover_layer = nn.Linear(self.latent_dim, self.in_out_dim)
        self.norm = variational_cfg.get("norm", "prefix")
        self.ln = nn.LayerNorm(self.in_out_dim)


        print(f"\n 初始化GaussianNet（5层版本）:")
        print(f"   阶段1 - 特征增强: {self.num_layers}层Pre-LN Attention")
        print(f"   阶段2 - 参数生成: 2层Transformer (替代原2层MLP)")
        print(f"   共享参数: {variational_cfg.get('attention_shared', True)}")
        
        if variational_cfg.get("attention_shared", True):
            
            self.shared_variator = MultiLayerAttention(
                variator_cfg, 
                variational_cfg,
                num_layers=self.num_layers
            )
            self.prior_variator = self.shared_variator
            self.posterior_variator = self.shared_variator
            print(f"    先验和后验共享{self.num_layers}层Pre-LN参数")
        else:

            self.prior_variator = MultiLayerAttention(
                variator_cfg, variational_cfg, num_layers=self.num_layers
            )
            self.posterior_variator = MultiLayerAttention(
                variator_cfg, variational_cfg, num_layers=self.num_layers
            )
            print(f"   先验和后验使用独立Pre-LN参数")

    def forward(
            self,
            prior_sign_encoder_out,
            posterior_sign_encoder_out,
            posterior_text_encoder_out,
            prior_sign_attnention_mask,
            posterior_sign_attnention_mask,
            posterior_text_attnention_mask,
    ):
        
        prior_sign_rep = self.prior_variator(
            query_embeds=prior_sign_encoder_out,
            key_value_embeds=prior_sign_encoder_out,
            query_mask=prior_sign_attnention_mask,
            key_value_mask=prior_sign_attnention_mask,
        )
        
       
        prior_residual = prior_sign_encoder_out
        prior_mean, prior_logvar = self.prior_net(prior_sign_rep, prior_sign_attnention_mask)
        
       
        prior_z = GaussianNet.reparameterize(
            prior_mean, prior_logvar, is_logv=True,
            temperature=1.0 if self.training else 0.0
        )
      
        prior_recover = self.combine(prior_residual, prior_z)

   
        posterior_rep = self.posterior_variator(
            query_embeds=posterior_sign_encoder_out,     
            key_value_embeds=posterior_text_encoder_out, 
            query_mask=posterior_sign_attnention_mask,
            key_value_mask=posterior_text_attnention_mask,
        )
        
        
        posterior_residual = posterior_sign_encoder_out
        delta_posterior_mean, delta_posterior_logvar = self.posterior_net(
            posterior_rep, posterior_sign_attnention_mask
        )
         
        posterior_mean = prior_mean + delta_posterior_mean
        posterior_logvar = prior_logvar + delta_posterior_logvar
       
        posterior_z = GaussianNet.reparameterize(
            posterior_mean, posterior_logvar, is_logv=True,
            temperature=1.0 if self.training else 0.0
        )
       
        posterior_recover = self.combine(posterior_residual, posterior_z)

        return {
            "posterior": {
                "mean": posterior_mean, 
                "logvar": posterior_logvar, 
                "z": posterior_z,
                "encoder_out": posterior_recover,
            } if posterior_sign_encoder_out is not None else None,
            "prior": {
                "mean": prior_mean, 
                "logvar": prior_logvar, 
                "z": prior_z,
                "encoder_out": prior_recover,
            } if prior_sign_encoder_out is not None else None,
        }

    def combine(self, inputs, z):
        z_recover = self.ln(self.recover_layer(z)) if self.norm == "prefix" else self.recover_layer(z)
        outputs = self.ln(inputs + z_recover) if self.norm == "postfix" else inputs + z_recover
        return outputs

    @staticmethod
    def reparameterize(mean, var, is_logv=False, sample_size=1, temperature=1.0):
        if sample_size > 1:
            mean = mean.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, mean.size(-1))
            var = var.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, var.size(-1))

        if not is_logv:
            sigma = torch.sqrt(var + 1e-10)
        else:
            sigma = torch.exp(0.5 * var)

        epsilon = torch.randn_like(sigma)
        z = mean + epsilon * sigma * temperature
        return z


def gaussian_kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar):
    kl_loss = -0.5 * torch.sum(
        1 + (posterior_logvar - prior_logvar)
        - torch.div(
            torch.pow(prior_mean - posterior_mean, 2) + posterior_logvar.exp(),
            prior_logvar.exp(),
        )
    )
    return kl_loss
