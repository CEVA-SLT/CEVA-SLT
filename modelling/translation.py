import math
import copy
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, MBartConfig
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

from utils.loss import XentLoss
from utils.misc import freeze_params, get_logger
from .Tokenizer import GlossTokenizer_G2T, TextTokenizer
from modelling.gaussian_net_5_t import GaussianNet, gaussian_kl_loss
from modelling.contrastive_loss import RecoverFeatureContrastiveLoss

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA will not be available.")

class DimensionUpscaler(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1280, output_dim=1536, dropout=0.1):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class DimensionDownscaler(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, output_dim=1024, dropout=0.1):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        #Learnable weights for residual connections
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, residual=None):
        out = self.layer1(x)
        out = self.layer2(out)
        
        #residual connection
        if residual is not None:
            if residual.shape[1] != out.shape[1]:
                min_len = min(residual.shape[1], out.shape[1])
                out[:, :min_len, :] = out[:, :min_len, :] + self.residual_weight * residual[:, :min_len, :]
            else:
                out = out + self.residual_weight * residual
        
        out = self.output_norm(out)
        return out

class LLMPrefixGenerator(nn.Module):
    def __init__(
        self,
        llm_path="autodl-tmp/CEVA-SLT/models/qwen2.5-1.5b/Qwen/Qwen2.5-1.5B",
        input_dim=1024,
        llm_hidden_dim=1536,
        output_dim=1024,
        prefix_length=64,
        num_llm_layers=6,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        dropout=0.1,
        use_lora=True,
        generator_name="default",
    ):
        super().__init__()
        
        self.prefix_length = prefix_length
        self.input_dim = input_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.output_dim = output_dim
        self.num_llm_layers = num_llm_layers
        self.generator_name = generator_name
        
        self.upscaler = DimensionUpscaler(
            input_dim=input_dim,
            hidden_dim=1280,
            output_dim=llm_hidden_dim,
            dropout=dropout
        )
        
        self.llm = self._load_truncated_llm(
            llm_path=llm_path,
            num_layers=num_llm_layers,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        self.downscaler = DimensionDownscaler(
            input_dim=llm_hidden_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.prefix_queries = nn.Parameter(
            torch.randn(1, prefix_length, llm_hidden_dim) * 0.02
        )
        
        self.use_extra_pos_embed = False
        
    def _load_truncated_llm(self, llm_path, num_layers, use_lora, lora_r, lora_alpha, lora_dropout):
        logger = get_logger()
        logger.info(f"[{self.generator_name}] Loading LLM from {llm_path}")
   
        try:
            from transformers import Qwen2ForCausalLM, Qwen2Config
            use_qwen2_class = True
            logger.info(f"[{self.generator_name}] Using Qwen2ForCausalLM class")
        except ImportError:
            use_qwen2_class = False
            logger.info(f"[{self.generator_name}] Qwen2 class not found, using AutoModelForCausalLM with trust_remote_code=True")
        
        if use_qwen2_class:
            config = Qwen2Config.from_pretrained(llm_path)
        else:
            config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
        
        original_num_layers = config.num_hidden_layers
        
        #Modify the configuration to only use the first N layers
        config.num_hidden_layers = num_layers
        logger.info(f"[{self.generator_name}] Truncating LLM from {original_num_layers} layers to {num_layers} layers")
        
        #Load the model
        if use_qwen2_class:
            llm = Qwen2ForCausalLM.from_pretrained(
                llm_path,
                config=config,
                torch_dtype=torch.float32,
            )
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
        
        #Frozen all LLM parameters
        for param in llm.parameters():
            param.requires_grad = False
        logger.info(f"[{self.generator_name}] Frozen all LLM parameters")
        
        if use_lora and PEFT_AVAILABLE:
            logger.info(f"[{self.generator_name}] Applying LoRA with r={lora_r}, alpha={lora_alpha}")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            llm = get_peft_model(llm, lora_config)
            llm.print_trainable_parameters()
        elif use_lora and not PEFT_AVAILABLE:
            logger.warning(f"[{self.generator_name}] LoRA requested but peft not installed. Using frozen LLM.")
        
        # Remove LM Head
        if hasattr(llm, 'lm_head'):
            llm.lm_head = nn.Identity()
        
        return llm
    
    def _create_causal_mask(self, seq_len, prefix_len, device):
        total_len = seq_len + prefix_len
    
        mask = torch.zeros(total_len, total_len, device=device)
        
        mask[:seq_len, :seq_len] = 0
        
        mask[seq_len:, :seq_len] = 0
        
        causal_mask = torch.triu(
            torch.ones(prefix_len, prefix_len, device=device) * float('-inf'),
            diagonal=1
        )
        mask[seq_len:, seq_len:] = causal_mask
        
        return mask
    
    def forward(self, encoder_out, attention_mask):

        batch_size, seq_len, _ = encoder_out.shape
        device = encoder_out.device
        
        #Save the original input for residual connection
        original_input = encoder_out
  
        upscaled = self.upscaler(encoder_out)
        prefix_queries = self.prefix_queries.expand(batch_size, -1, -1) 
        llm_input = torch.cat([upscaled, prefix_queries], dim=1)  
        
        prefix_mask = torch.ones(batch_size, self.prefix_length, dtype=torch.long, device=device)
        full_attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)  
        
        if hasattr(self.llm, 'base_model'):
            base_model = self.llm.base_model.model.model
        else:
            base_model = self.llm.model
        
        llm_outputs = base_model(
            inputs_embeds=llm_input,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        if hasattr(llm_outputs, 'last_hidden_state'):
            hidden_states = llm_outputs.last_hidden_state  
        else:
            hidden_states = llm_outputs.hidden_states[-1]
        
        prefix_hidden = hidden_states[:, -self.prefix_length:, :] 
        
        masked_input = original_input * attention_mask.unsqueeze(-1).float()
        pooled_input = masked_input.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled_input = pooled_input.unsqueeze(1).expand(-1, self.prefix_length, -1)  
        
        prefix = self.downscaler(prefix_hidden, residual=pooled_input)
        

        prefix = torch.clamp(prefix, min=-10.0, max=10.0)
        
        if torch.isnan(prefix).any() or torch.isinf(prefix).any():
            import warnings
            warnings.warn(f" [{self.generator_name}] Prefix contains NaN or Inf! Resetting to zeros.")
            prefix = torch.zeros_like(prefix)
        
        # Prefix mask
        output_prefix_mask = torch.ones(
            batch_size, self.prefix_length,
            dtype=torch.long,
            device=device
        )
        
        return prefix, output_prefix_mask

class TranslationNetwork(torch.nn.Module):
    def __init__(self, input_type, cfg, task) -> None:
        super().__init__()
        self.frozen_modules = []
        self.logger = get_logger()
        self.task = task
        self.input_type = input_type

        assert self.input_type in ['gloss', 'feature', 'text']

        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])

        if 'pretrained_model_name_or_path' in cfg:
            self.logger.info('Initialize translation network from {}'.format(cfg['pretrained_model_name_or_path']))

            self.model = MBartForConditionalGeneration.from_pretrained(
                cfg['pretrained_model_name_or_path'],
                **cfg.get('overwrite_cfg', {})
            )
        elif 'model_config' in cfg:
            self.logger.info('Train translation network from scratch using config={}'.format(cfg['model_config']))
            config = MBartConfig.from_pretrained(cfg['model_config'])
            for k, v in cfg.get('overwrite_cfg', {}).items():
                setattr(config, k, v)
                self.logger.info('Overwrite {}={}'.format(k, v))
            if cfg['TextTokenizer'].get('level', 'sentencepiece') == 'word':
                setattr(config, 'vocab_size', len(self.text_tokenizer.id2token))
                self.logger.info('Vocab_size {}'.format(config.vocab_size))
            self.model = MBartForConditionalGeneration(config=config)

            if 'pretrained_pe' in cfg:
                pe = torch.load(cfg['pretrained_pe']['pe_file'], map_location='cpu')
                self.logger.info('Load pretrained positional embedding from ', cfg['pretrained_pe']['pe_file'])
                with torch.no_grad():
                    self.model.model.encoder.embed_positions.weight = torch.nn.parameter.Parameter(
                        pe['model.encoder.embed_positions.weight']
                    )
                    self.model.model.decoder.embed_positions.weight = torch.nn.parameter.Parameter(
                        pe['model.decoder.embed_positions.weight']
                    )
                if cfg['pretrained_pe']['freeze']:
                    self.logger.info('Set positional embedding frozen')
                    freeze_params(self.model.model.encoder.embed_positions)
                    freeze_params(self.model.model.decoder.embed_positions)
                else:
                    self.logger.info('Set positional embedding trainable')
        else:
            raise ValueError

        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index,
            smoothing=cfg['label_smoothing']
        )
        self.input_dim = self.model.config.d_model
        self.input_embed_scale = cfg.get('input_embed_scale', math.sqrt(self.model.config.d_model))

        if self.task in ['S2T', 'G2T'] and 'pretrained_model_name_or_path' in cfg:
            self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg['GlossTokenizer'])
            self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])
            self.gls_eos = cfg.get('gls_eos', 'gls')
        elif self.task in ['S2T_glsfree']:
            self.gls_eos = None
            self.gloss_tokenizer, self.gloss_embedding = None, None
        elif 'pretrained_model_name_or_path' not in cfg:
            self.gls_eos = 'txt'
            self.gloss_tokenizer, self.gloss_embedding = None, None
        else:
            raise ValueError

        if cfg.get('from_scratch', False):
            self.model.init_weights()
            self.logger.info('Build Translation Network with scratch config!')
        if cfg.get('freeze_txt_embed', False):
            freeze_params(self.model.model.shared)
            self.logger.info('Set txt embedding frozen')

        if 'load_ckpt' in cfg:
            self.load_from_pretrained_ckpt(cfg['load_ckpt'])

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        logger = get_logger()
        logger.info(
            'Loading and Reinitializing Translation network from pretrained ckpt {}'.format(pretrained_ckpt)
        )
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k, v in checkpoint.items():
            if 'translation_network' in k:
                load_dict[k.replace('translation_network.', '')] = v
        self.load_state_dict(load_dict)

    def build_gloss_embedding(self, gloss2embed_file, from_scratch=False, freeze=False):
        gloss_embedding = torch.nn.Embedding(
            num_embeddings=len(self.gloss_tokenizer.id2gloss),
            embedding_dim=self.model.config.d_model,
            padding_idx=self.gloss_tokenizer.gloss2id['<pad>']
        )
        self.logger.info('gloss2embed_file ' + gloss2embed_file)
        if from_scratch:
            self.logger.info('Train Gloss Embedding from scratch')
            assert freeze is False
        else:
            gls2embed = torch.load(gloss2embed_file)
            self.gls2embed = gls2embed
            self.logger.info('Initialize gloss embedding from {}'.format(gloss2embed_file))
            with torch.no_grad():
                for id_, gls in self.gloss_tokenizer.id2gloss.items():
                    if gls in gls2embed:
                        assert gls in gls2embed, gls
                        gloss_embedding.weight[id_, :] = gls2embed[gls]
                    else:
                        self.logger.info('{} not in gls2embed train from scratch'.format(gls))

        if freeze:
            freeze_params(gloss_embedding)
            self.logger.info('Set gloss embedding frozen')
        return gloss_embedding

    def prepare_gloss_inputs(self, input_ids):
        input_emb = self.gloss_embedding(input_ids) * self.input_embed_scale
        return input_emb

    def prepare_feature_inputs(self, input_feature, input_lengths, gloss_embedding=None, gloss_lengths=None):
        if self.task == 'S2T_glsfree':
            suffix_len = 0
            suffix_embedding = None
        else:
            if self.gls_eos == 'gls':
                assert self.gloss_embedding is not None
                suffix_embedding = [self.gloss_embedding.weight[self.gloss_tokenizer.convert_tokens_to_ids('</s>'), :]]
            else:
                suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index, :]]
            if self.task in ['S2T', 'G2T']:
                if self.gls_eos == 'gls':
                    assert self.gloss_embedding is not None
                    src_lang_code_embedding = self.gloss_embedding.weight[
                                              self.gloss_tokenizer.convert_tokens_to_ids(self.gloss_tokenizer.src_lang),
                                              :]
                else:
                    src_lang_id = self.text_tokenizer.lang_index
                    src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
                suffix_embedding.append(src_lang_code_embedding)
            suffix_len = len(suffix_embedding)
            suffix_embedding = torch.stack(suffix_embedding, dim=0)

        max_length = torch.max(input_lengths) + suffix_len
        inputs_embeds = []
        attention_mask = torch.zeros(
            [input_feature.shape[0], max_length],
            dtype=torch.long,
            device=input_feature.device
        )
        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii]
            if 'gloss+feature' in self.input_type:
                valid_feature = torch.cat(
                    [gloss_embedding[ii, :gloss_lengths[ii], :], feature[:valid_len - gloss_lengths[ii], :]],
                    dim=0
                )
            else:
                valid_feature = feature[:valid_len, :]
            if suffix_embedding is not None:
                feature_w_suffix = torch.cat([valid_feature, suffix_embedding], dim=0)
            else:
                feature_w_suffix = valid_feature
            if feature_w_suffix.shape[0] < max_length:
                pad_len = max_length - feature_w_suffix.shape[0]
                padding = torch.zeros(
                    [pad_len, feature_w_suffix.shape[1]],
                    dtype=feature_w_suffix.dtype,
                    device=feature_w_suffix.device
                )
                padded_feature_w_suffix = torch.cat([feature_w_suffix, padding], dim=0)
                inputs_embeds.append(padded_feature_w_suffix)
            else:
                inputs_embeds.append(feature_w_suffix)
            attention_mask[ii, :valid_len + suffix_len] = 1
        transformer_inputs = {
            'inputs_embeds': torch.stack(inputs_embeds, dim=0) * self.input_embed_scale,
            'attention_mask': attention_mask
        }
        return transformer_inputs

    def forward(self, **kwargs):
        if self.input_type == 'gloss':
            kwargs.pop('text_length', None)
            input_ids = kwargs.pop('input_ids')
            kwargs['inputs_embeds'] = self.prepare_gloss_inputs(input_ids)
        elif self.input_type == 'feature':
            input_feature = kwargs.pop('input_feature')
            input_lengths = kwargs.pop('input_lengths')
            kwargs.pop('input_ids', None)
            kwargs.pop('text_length', None)
            kwargs.pop('gloss_ids', None)
            kwargs.pop('gloss_lengths', None)
            new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
            kwargs = {**kwargs, **new_kwargs}
        else:
            raise ValueError
        output_dict = self.model(**kwargs, output_hidden_states=None if self.training else True, return_dict=True)
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob, targets=kwargs['labels'])
        output_dict['translation_loss'] = batch_loss_sum / log_prob.shape[0]

        output_dict['transformer_inputs'] = kwargs
        return output_dict

    def generate(
            self,
            input_ids=None, attention_mask=None,
            inputs_embeds=None, input_lengths=None,
            num_beams=4, max_length=100, length_penalty=1, **kwargs
    ):
        assert attention_mask is not None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones(
            [batch_size, 1], dtype=torch.long,
            device=attention_mask.device
        ) * self.text_tokenizer.sos_index
        assert inputs_embeds is not None and attention_mask is not None
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_length,
            return_dict_in_generate=True
        )
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        return output_dict


class VariationalTranslationNetwork(TranslationNetwork):
    def __init__(self, input_type, translation_cfg, variation_cfg, task) -> None:
        super().__init__(input_type, translation_cfg, task)
        self.gls_eos = "txt"
        self.encoder = self.model.get_encoder()

        self.gaussian_net = GaussianNet(variation_cfg, self.model.config)

        self.prior_weight = variation_cfg.get('prior_weight', 1.0)
        self.gkl_factor = variation_cfg.get('gkl_factor', 1.0)
        self.gkl_step_scheduler = StepWarmUpScheduler(
            start_ratio=variation_cfg.get("gkl_start_ratio", 0.0),
            end_ratio=self.gkl_factor,
            warmup_start_step=variation_cfg.get("gkl_warmup_start", 0),
            warmup_step=variation_cfg.get("gkl_warmup_step", 4000),
        )
        self.kl_factor = variation_cfg.get('kl_factor', 1.0)
        self.kl_step_scheduler = StepWarmUpScheduler(
            start_ratio=variation_cfg.get("kl_start_ratio", 0.0),
            end_ratio=self.kl_factor,
            warmup_start_step=variation_cfg.get("kl_warmup_start", 0),
            warmup_step=variation_cfg.get("kl_warmup_step", 4000),
        )

        #Contrastive learning for feature recovery
        self.use_recover_contrastive = variation_cfg.get('use_recover_contrastive', False)
        if self.use_recover_contrastive:
            self.logger.info('Initializing Recover Feature Contrastive Learning Module')
            self.recover_contrastive_loss_fn = RecoverFeatureContrastiveLoss(
                temperature=variation_cfg.get('recover_contrast_temperature', 0.07)
            )
            
            self.recover_contrast_factor = variation_cfg.get('recover_contrast_factor', 0.5)
            self.recover_contrast_step_scheduler = StepWarmUpScheduler(
                start_ratio=variation_cfg.get("recover_contrast_start_ratio", 0.0),
                end_ratio=self.recover_contrast_factor,
                warmup_start_step=variation_cfg.get("recover_contrast_warmup_start", 1000),
                warmup_step=variation_cfg.get("recover_contrast_warmup_step", 4000),
            )
            self.logger.info(f'  - Temperature: {variation_cfg.get("recover_contrast_temperature", 0.07)}')
            self.logger.info(f'  - Factor: {self.recover_contrast_factor}')
            self.logger.info(f'  - Warmup: {variation_cfg.get("recover_contrast_warmup_start", 1000)} ~ '
                           f'{variation_cfg.get("recover_contrast_warmup_start", 1000) + variation_cfg.get("recover_contrast_warmup_step", 4000)}')
        else:
            self.logger.info('Recover Feature Contrastive Learning Module: Disabled')
        
        #Distillation
        self.use_distillation = variation_cfg.get('use_distillation', False)
        if self.use_distillation:
            self.logger.info('Initializing Knowledge Distillation')
            self.distill_factor = variation_cfg.get('distill_factor', 1.0)
            self.distill_temperature = variation_cfg.get('distill_temperature', 1.0)
            
            self.distill_step_scheduler = StepWarmUpScheduler(
                start_ratio=variation_cfg.get("distill_start_ratio", 0.0),
                end_ratio=self.distill_factor,
                warmup_start_step=variation_cfg.get("distill_warmup_start", 1000),
                warmup_step=variation_cfg.get("distill_warmup_step", 3000),
            )
            
            self.logger.info(f'  - Distillation Factor: {self.distill_factor}')
            self.logger.info(f'  - Distillation Temperature: {self.distill_temperature}')
        else:
            self.logger.info('Knowledge Distillation: Disabled')

        #Teacher Forcing Masking
        self.use_tf_masking = variation_cfg.get('use_tf_masking', False)
        if self.use_tf_masking:
            self.logger.info('Initializing Teacher Forcing Masking')
            
            #Masking parameters
            self.tf_mask_prob = variation_cfg.get('tf_mask_prob', 0.15)
            
            #Get the UNK token ID
            if hasattr(self.text_tokenizer, 'unk_index'):
                self.tf_mask_token_id = self.text_tokenizer.unk_index
            elif hasattr(self.text_tokenizer.tokenizer, 'unk_token_id') and self.text_tokenizer.tokenizer.unk_token_id is not None:
                self.tf_mask_token_id = self.text_tokenizer.tokenizer.unk_token_id
            else:
                self.logger.warning('UNK token not found, using PAD token for masking')
                self.tf_mask_token_id = self.text_tokenizer.pad_index
            
            #Special tokens that need to be protected
            self.tf_special_tokens = {
                self.text_tokenizer.pad_index,   
                self.text_tokenizer.sos_index,   
                self.text_tokenizer.eos_index,   
            }
            
            if hasattr(self.text_tokenizer, 'ignore_index'):
                self.tf_special_tokens.add(self.text_tokenizer.ignore_index)
            
            #Warmup scheduler
            self.tf_mask_scheduler = StepWarmUpScheduler(
                start_ratio=variation_cfg.get("tf_mask_start_ratio", 0.0),
                end_ratio=self.tf_mask_prob,
                warmup_start_step=variation_cfg.get("tf_mask_warmup_start", 2000),
                warmup_step=variation_cfg.get("tf_mask_warmup_step", 3000),
            )
            
            #Whether to mask the posterior network simultaneously
            self.tf_mask_posterior = variation_cfg.get('tf_mask_posterior', False)
            
            self.logger.info(f'  - Mask Probability: {self.tf_mask_prob}')
            self.logger.info(f'  - Mask Token ID: {self.tf_mask_token_id}')
            self.logger.info(f'  - Protected Token IDs: {self.tf_special_tokens}')
            self.logger.info(f'  - Mask Posterior: {self.tf_mask_posterior}')
            self.logger.info(f'  - Warmup: {variation_cfg.get("tf_mask_warmup_start", 2000)} ~ '
                           f'{variation_cfg.get("tf_mask_warmup_start", 2000) + variation_cfg.get("tf_mask_warmup_step", 3000)}')
        else:
            self.logger.info('Teacher Forcing Masking: Disabled')

        #LGPE
        self.use_prefix_tuning = variation_cfg.get('use_prefix_tuning', False)
        if self.use_prefix_tuning:
            self.logger.info('Initializing LGPE (Separate Prior & Posterior LLMs)')
            
            #LLM configuration parameters
            llm_path = variation_cfg.get(
                'llm_path', 
                'autodl-tmp/CEVA-SLT/models/qwen2.5-1.5b/Qwen/Qwen2.5-1.5B'
            )
            prefix_length = variation_cfg.get('prefix_length', 64)
            num_llm_layers = variation_cfg.get('num_llm_layers', 6)
            lora_r = variation_cfg.get('lora_r', 16)
            lora_alpha = variation_cfg.get('lora_alpha', 32)
            lora_dropout = variation_cfg.get('lora_dropout', 0.05)
            prefix_dropout = variation_cfg.get('prefix_dropout', 0.1)
            use_lora = variation_cfg.get('use_lora', True)
            
            #Prior LGPE
            self.logger.info('Creating Prior LGPE')
            self.prior_llm_prefix_generator = LLMPrefixGenerator(
                llm_path=llm_path,
                input_dim=variation_cfg.get('input_embed_dim', self.model.config.d_model),
                llm_hidden_dim=variation_cfg.get('llm_hidden_dim', 1536),
                output_dim=self.model.config.d_model,
                prefix_length=prefix_length,
                num_llm_layers=num_llm_layers,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                dropout=prefix_dropout,
                use_lora=use_lora,
                generator_name="Prior",
            )
            
            #Posterior LGPE
            self.logger.info('Creating Posterior LGPE')
            self.posterior_llm_prefix_generator = LLMPrefixGenerator(
                llm_path=llm_path,
                input_dim=variation_cfg.get('input_embed_dim', self.model.config.d_model),
                llm_hidden_dim=variation_cfg.get('llm_hidden_dim', 1536),
                output_dim=self.model.config.d_model,
                prefix_length=prefix_length,
                num_llm_layers=num_llm_layers,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                dropout=prefix_dropout,
                use_lora=use_lora,
                generator_name="Posterior",
            )
            
            self.logger.info(f'     LLM Prefix-Tuning Configuration:')
            self.logger.info(f'     - LLM Path: {llm_path}')
            self.logger.info(f'     - Prefix Length: {prefix_length}')
            self.logger.info(f'     - LLM Layers: {num_llm_layers}')
            self.logger.info(f'     - LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}')
            self.logger.info(f'     - Input Dim: {variation_cfg.get("input_embed_dim", 1024)}')
            self.logger.info(f'     - LLM Hidden Dim: {variation_cfg.get("llm_hidden_dim", 1536)}')
            self.logger.info(f'     - Output Dim: {self.model.config.d_model}')
            self.logger.info(f'     - Mode: Separate LLMs for Prior and Posterior')
        else:
            self.logger.info('LLM Prefix-Tuning Module: Disabled')

        if hasattr(self, "gloss_embedding"):
            delattr(self, "gloss_embedding")
        if hasattr(self, "gloss_tokenizer"):
            delattr(self, "gloss_tokenizer")
        
        self.forward_step_counter = 0

    def set_num_updates(self, num_updates):
        self.kl_factor = self.kl_step_scheduler.forward(num_updates)
        self.gkl_factor = self.gkl_step_scheduler.forward(num_updates)
        
        if self.use_recover_contrastive:
            self.recover_contrast_factor = self.recover_contrast_step_scheduler.forward(num_updates)
        if self.use_distillation:
            self.distill_factor = self.distill_step_scheduler.forward(num_updates)
        
        if self.use_tf_masking:
            self.tf_mask_prob = self.tf_mask_scheduler.forward(num_updates)

    def prepare_gaussian_net_feature_inputs(self, sign_embeds, sign_mask, text=None):
        text_embeds = self.model.model.shared(text) * self.input_embed_scale
        text_mask = text.ne(self.text_tokenizer.pad_index)
        transformer_inputs = {
            'inputs_embeds': torch.cat([sign_embeds, text_embeds], dim=1),
            'attention_mask': torch.cat([sign_mask, text_mask], dim=1),
        }
        return transformer_inputs

    def _compute_kl_loss(self, prior_out, posterior_out):
        kl_1 = F.kl_div(prior_out.log_softmax(-1), posterior_out.softmax(-1), reduction="sum")
        kl_2 = F.kl_div(posterior_out.log_softmax(-1), prior_out.softmax(-1), reduction="sum")
        kl_loss = (kl_1 + kl_2) / 2
        return kl_loss

    def _compute_gaussian_kl_loss(self, posterior_mean, posterior_logvar, prior_mean, prior_logvar):
        gkl_loss = gaussian_kl_loss(
            posterior_mean=posterior_mean, posterior_logvar=posterior_logvar,
            prior_mean=prior_mean, prior_logvar=prior_logvar,
        )
        return gkl_loss

    def _shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        
        return shifted_input_ids

    def _apply_teacher_forcing_mask(self, decoder_input_ids):

        if not self.training or not self.use_tf_masking:
            return decoder_input_ids
        
        batch_size, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        mask_matrix = torch.zeros_like(decoder_input_ids, dtype=torch.bool)
        
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = decoder_input_ids[i, j].item()
                
                #Skip special tokens and padding
                if token_id in self.tf_special_tokens:
                    continue
                
                #Randomly decide whether to mask
                if torch.rand(1).item() < self.tf_mask_prob:
                    mask_matrix[i, j] = True
        
        masked_decoder_input_ids = decoder_input_ids.clone()
        masked_decoder_input_ids[mask_matrix] = self.tf_mask_token_id
        
        return masked_decoder_input_ids

    def forward(self, **kwargs):
        kwargs.pop('gloss_ids', None)
        kwargs.pop('gloss_lengths', None)

        input_feature, input_lengths = kwargs.pop('input_feature'), kwargs.pop('input_lengths')
        kwargs.pop('text_length', None)
        kwargs.pop('input_ids', None)

        encoder_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
        kwargs = {**kwargs, **encoder_kwargs}

        posterior_encoder_kwargs = self.prepare_gaussian_net_feature_inputs(
            kwargs["inputs_embeds"], kwargs["attention_mask"], text=kwargs['labels']
        )
        prior_encoder_output_dict = self.encoder(**encoder_kwargs, return_dict=True,)
        posterior_encodet_output_dict = self.encoder(**posterior_encoder_kwargs, return_dict=True)
        sign_max_length = kwargs["inputs_embeds"].size(1)
        prior_sign_encoder_out = prior_encoder_output_dict["last_hidden_state"]
        posterior_sign_encoder_out, posterior_text_encoder_out = (
            posterior_encodet_output_dict["last_hidden_state"][:, :sign_max_length, :],
            posterior_encodet_output_dict["last_hidden_state"][:, sign_max_length:, :],
        )
        
        # gaussian net output
        gaussian_out = self.gaussian_net(
            prior_sign_encoder_out=prior_sign_encoder_out,
            posterior_sign_encoder_out=posterior_sign_encoder_out,
            posterior_text_encoder_out=posterior_text_encoder_out,
            prior_sign_attnention_mask=kwargs["attention_mask"],
            posterior_sign_attnention_mask=posterior_encoder_kwargs['attention_mask'][:, :sign_max_length],
            posterior_text_attnention_mask=posterior_encoder_kwargs["attention_mask"][:, sign_max_length:],
        )

        prior_features = gaussian_out["prior"]["encoder_out"]
        posterior_features = gaussian_out["posterior"]["encoder_out"]
        prior_mask = kwargs["attention_mask"]
        posterior_mask = kwargs["attention_mask"]

        #LGPE
        if self.use_prefix_tuning:
            prior_prefix, prior_prefix_mask = self.prior_llm_prefix_generator(
                prior_features, prior_mask
            )
            posterior_prefix, posterior_prefix_mask = self.posterior_llm_prefix_generator(
                posterior_features, posterior_mask
            )
            prior_features_with_prefix = torch.cat([prior_prefix, prior_features], dim=1)
            posterior_features_with_prefix = torch.cat([posterior_prefix, posterior_features], dim=1)
            
            prior_mask_with_prefix = torch.cat([prior_prefix_mask, prior_mask], dim=1)
            posterior_mask_with_prefix = torch.cat([posterior_prefix_mask, posterior_mask], dim=1)
            
            final_prior_features = prior_features_with_prefix
            final_posterior_features = posterior_features_with_prefix
            final_prior_mask = prior_mask_with_prefix
            final_posterior_mask = posterior_mask_with_prefix
        else:
            final_prior_features = prior_features
            final_posterior_features = posterior_features
            final_prior_mask = prior_mask
            final_posterior_mask = posterior_mask

        prior_encoder_outputs = BaseModelOutput(last_hidden_state=final_prior_features)
        posterior_encoder_outputs = BaseModelOutput(last_hidden_state=final_posterior_features)

        prior_kwargs = {**kwargs, 'attention_mask': final_prior_mask}
        posterior_kwargs = {**kwargs, 'attention_mask': final_posterior_mask}

        #Teacher Forcing Masking
        if 'decoder_input_ids' not in prior_kwargs:
            decoder_input_ids = self._shift_tokens_right(
                prior_kwargs['labels'],
                self.text_tokenizer.pad_index,
                self.text_tokenizer.eos_index
            )
            prior_kwargs['decoder_input_ids'] = decoder_input_ids
            posterior_kwargs['decoder_input_ids'] = decoder_input_ids.clone()

        if self.use_tf_masking and self.training:
            prior_kwargs['decoder_input_ids'] = self._apply_teacher_forcing_mask(
                prior_kwargs['decoder_input_ids']
            )
            
            if self.tf_mask_posterior:
                posterior_kwargs['decoder_input_ids'] = self._apply_teacher_forcing_mask(
                    posterior_kwargs['decoder_input_ids']
                )

        prior_output_dict = self.model(**prior_kwargs, return_dict=True, encoder_outputs=prior_encoder_outputs)
        posterior_output_dict = self.model(**posterior_kwargs, return_dict=True, encoder_outputs=posterior_encoder_outputs)

        sample_size = kwargs['labels'].size(0)
        
        prior_batch_loss_sum = self.translation_loss_fun(
            log_probs=prior_output_dict['logits'].log_softmax(-1), targets=kwargs['labels']
        )
        posterior_batch_loss_sum = self.translation_loss_fun(
            log_probs=posterior_output_dict['logits'].log_softmax(-1), targets=kwargs['labels']
        )

        # KL loss and GKL loss
        kl_loss = self._compute_kl_loss(
            prior_out=prior_output_dict["logits"], posterior_out=posterior_output_dict["logits"],
        )

        gkl_loss = self._compute_gaussian_kl_loss(
            posterior_mean=gaussian_out["posterior"]["mean"], posterior_logvar=gaussian_out["posterior"]["logvar"],
            prior_mean=gaussian_out["prior"]["mean"], prior_logvar=gaussian_out["prior"]["logvar"],
        )
        #CL loss
        if self.use_recover_contrastive:
            recover_contrastive_result = self.recover_contrastive_loss_fn(
                prior_recover=prior_features,
                posterior_recover=posterior_features,
                mask=kwargs["attention_mask"]
            )
            
            prior_output_dict["recover_contrastive_loss"] = (
                recover_contrastive_result['loss'] * self.recover_contrast_factor
            )
            prior_output_dict["recover_pos_sim"] = recover_contrastive_result['pos_sim']
            prior_output_dict["recover_neg_sim"] = recover_contrastive_result['neg_sim']
            prior_output_dict["recover_contrast_factor"] = self.recover_contrast_factor
        else:
            prior_output_dict["recover_contrastive_loss"] = torch.tensor(0.0, device=input_feature.device)
        
        #Distillation loss
        if self.use_distillation:
            T = self.distill_temperature
            
            distill_loss = F.kl_div(
                F.log_softmax(prior_output_dict['logits'] / T, dim=-1),
                F.softmax(posterior_output_dict['logits'] / T, dim=-1).detach(),
                reduction='batchmean'
            ) * (T ** 2)
            
            prior_output_dict["distillation_loss"] = distill_loss * self.distill_factor
            prior_output_dict["distill_factor"] = self.distill_factor
        else:
            prior_output_dict["distillation_loss"] = torch.tensor(0.0, device=input_feature.device)
        
        prior_output_dict['posterior_translation_loss'] = posterior_batch_loss_sum / sample_size
        prior_output_dict['prior_translation_loss'] = prior_batch_loss_sum / sample_size * self.prior_weight
        prior_output_dict["kl_loss"] = kl_loss / sample_size * self.kl_factor
        prior_output_dict["gkl_loss"] = gkl_loss / sample_size * self.gkl_factor

        #Total loss
        prior_output_dict['translation_loss'] = (
                prior_output_dict['posterior_translation_loss']
                + prior_output_dict['prior_translation_loss']
                + prior_output_dict['gkl_loss']
                + prior_output_dict['kl_loss']
                + prior_output_dict['recover_contrastive_loss']
                + prior_output_dict['distillation_loss']
        )
        prior_output_dict['gkl_factor'], prior_output_dict['kl_factor'] = self.gkl_factor, self.kl_factor

        if self.use_tf_masking and self.training:
            prior_output_dict['tf_mask_prob'] = self.tf_mask_prob
            
            if 'decoder_input_ids' in prior_kwargs:
                mask_count = (prior_kwargs['decoder_input_ids'] == self.tf_mask_token_id).sum().item()
                total_tokens = (prior_kwargs['decoder_input_ids'] != self.text_tokenizer.pad_index).sum().item()
                prior_output_dict['tf_actual_mask_ratio'] = mask_count / max(total_tokens, 1)

        kwargs["encoder_outputs"] = prior_encoder_outputs
        kwargs["attention_mask"] = final_prior_mask
        prior_output_dict['transformer_inputs'] = kwargs
        prior_output_dict['posterior_encoder_outputs'] = posterior_encoder_outputs

        return prior_output_dict

    def generate(
            self,
            input_ids=None, attention_mask=None,
            encoder_outputs=None,
            inputs_embeds=None, input_lengths=None,
            num_beams=4, max_length=100, length_penalty=1, **kwargs
    ):
        assert attention_mask is not None
        assert encoder_outputs is not None
        assert inputs_embeds is not None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones(
            [batch_size, 1], dtype=torch.long,
            device=attention_mask.device
        ) * self.text_tokenizer.sos_index
        output_dict = self.model.generate(
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_length,
            return_dict_in_generate=True,
        )
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])

        return output_dict


class StepWarmUpScheduler(object):
    def __init__(self, start_ratio, end_ratio, warmup_start_step, warmup_step):
        super().__init__()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_start_step = warmup_start_step
        self.warmup_step = warmup_step + int(warmup_step == 0)
        self.step_ratio = (end_ratio - start_ratio) / self.warmup_step

    def forward(self, step_num):
        if step_num < self.warmup_start_step:
            return self.start_ratio
        elif step_num >= self.warmup_start_step + self.warmup_step:
            return self.end_ratio
        else:
            ratio = self.start_ratio + self.step_ratio * (step_num - self.warmup_start_step)
            return ratio
