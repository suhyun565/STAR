# coding=utf-8
# copied from bart

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activations import ACT2FN
from models.model_utils import BaseModel
from transformers import TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration)
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput)
from .PARAMS import SAMPLE, TEMPERATURE
import numpy as np

import copy

class StrategyBlenderbotSmallConfig(BlenderbotSmallConfig):
    model_type = "strategy_blenderbot"

    def __init__(
        self,
        strategy_categories: list[str] = None,
        gate_dim=64,
        skip_gate_prob=0.0,
        strategy_dim=40,
        epoch=0,
        data_name = None,
        knowledge_name = None,
        **kwargs,
    ):
        self.strategy_categories = strategy_categories
        self.gate_dim = gate_dim
        self.skip_gate_prob = skip_gate_prob
        self.strategy_dim = strategy_dim
        self.epoch = epoch
        self.data_name = data_name
        self.knowledge_name = knowledge_name
        self.lora = kwargs.get('lora', False)
        super().__init__(**kwargs)


class FFN(nn.Module):
    def __init__(self, activation_function, activation_dropout, embed_dim, ffn_dim):
        super().__init__()
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = activation_dropout
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, hidden_states, activation_dropout, dropout, training):
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=activation_dropout, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=dropout, training=training)
        return hidden_states

    def set_from_pretrained(self, fc1, fc2, activation_fn):
        self.fc1 = copy.deepcopy(fc1)
        self.fc2 = copy.deepcopy(fc2)
        self.activation_fn = copy.deepcopy(activation_fn)

class AttentionPool1d(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):

        # x = x.permute(1, 0, 2)  # BLD -> LBD
        # x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC

        pooled = x.mean(dim=0, keepdim=True)
        x = torch.cat([pooled, x], dim=0)

        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x.squeeze(0)

class STAR(nn.Module):
    def __init__(
        self,
        config: BlenderbotSmallConfig,
        st_config,
        activation_function: str,
        activation_dropout: float,
        embed_dim: int,
        ffn_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.strategies_category = st_config.get('strategy_categories')
        print("---------------------------------------------------------------------------------")
        print("STRATEGY CATEGORIES")
        print(self.strategies_category)
        print("---------------------------------------------------------------------------------")
        self.strat_dict = {strategy: str(idx) for idx, strategy in enumerate(self.strategies_category)}
        self.strategies = list(self.strat_dict.values())
        self.skip_gate_prob = st_config.get('skip_gate_prob')
        self.gate_dim = st_config.get('gate_dim')

        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = activation_dropout
        self.ffn_dim = ffn_dim

        self.attention_pooling = AttentionPool1d(embed_dim, num_heads=8)

        ##################################
        #### SARA ####
        self.gate_in = nn.ModuleDict({strategy: nn.Linear(embed_dim, self.gate_dim, bias=True) for strategy in self.strategies})
        self.gate_out = nn.ModuleDict({strategy: nn.Linear(self.gate_dim, 1, bias=False) for strategy in self.strategies})
        ##################################
        
        ##################################
        #### Strategy Refinement (SR) ####
        self.proj = nn.ModuleDict({strategy: FFN(activation_function, activation_dropout, embed_dim, ffn_dim) for strategy in self.strategies})
        ##################################
        for gate_in in self.gate_in.values():
            nn.init.normal_(gate_in.weight, mean=0, std=0.01)
            nn.init.zeros_(gate_in.bias)
        for gate_out in self.gate_out.values():
            nn.init.zeros_(gate_out.weight)
        for proj in self.proj.values():
            nn.init.xavier_uniform_(proj.fc1.weight, mean=0, std=0.01)
            nn.init.zeros_(proj.fc1.bias)
            nn.init.xavier_uniform_(proj.fc2.weight)
            nn.init.zeros_(proj.fc2.bias)
        ##################################

    def forward(self, strategy: str, hidden_states: torch.Tensor, dropout: float, training: bool) -> torch.Tensor:
                
        batch_size = strategy.size(0)
        outputs = []
        gate_values = []
        
        for i in range(batch_size):
            # Choose the strategy for the current instance
            strategy_idx = strategy[i].item()
            strategy_key = self.strategies[strategy_idx]

            attn_output = self.attention_pooling(hidden_states[i])

            #### SARA - Calculate gate values ####
            gate_input = self.gate_in[strategy_key](nn.functional.dropout(attn_output, p=0.2, training=training))
            gate_input = torch.relu(gate_input)
            gate_value = torch.sigmoid(self.gate_out[strategy_key](gate_input))
            
            ### SR - Apply strategy refinement ###
            proj_output = self.proj[strategy_key].forward(
                        attn_output,
                        activation_dropout=self.activation_dropout,
                        dropout=dropout,
                        training=training
                    )
            
            #### Combine outputs based on gate values ####
            output = gate_value * proj_output + (1 - gate_value) * hidden_states[i]
            
            outputs.append(output)
            gate_values.append(gate_value)
        
        # 결과를 다시 텐서로 변환
        outputs = torch.stack(outputs, dim=0)
        gate_values = torch.stack(gate_values, dim=0)

        return outputs, gate_values

def concat_configs(config, st_config):
    config_dict = config.to_dict()
    config_dict.update(st_config)
    return StrategyBlenderbotSmallConfig(**config_dict)

class Model(BlenderbotSmallForConditionalGeneration):
    config_class = StrategyBlenderbotSmallConfig

    def __init__(self, config: StrategyBlenderbotSmallConfig, toker, **kwargs):
        super().__init__(config)
        self.config = config
        self.toker = toker
        self.st_config = kwargs.get('st_config')
        self.config = concat_configs(config, self.st_config)
        self.star = STAR(
            config,
            self.st_config,
            activation_function=config.activation_function,
            activation_dropout=config.activation_dropout,
            embed_dim=config.d_model,
            ffn_dim=config.decoder_ffn_dim,
            output_dim=config.strategy_dim,
        )
        self.gate_criteria = nn.CrossEntropyLoss()
        # self.hidden_states = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            labels[:, 0] = -100
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation: # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # self.hidden_states.append(outputs[0].cpu().numpy())

        strategy = self.predict_strategy(lm_logits, self.config.data_name, self.config.knowledge_name, encoded_info)

        hidden_states, gate_value = self.star(
            strategy,
            outputs[0],
            dropout=self.config.dropout,
            training=self.training
        )
        
        lm_logits = self.lm_head(hidden_states) + self.final_logits_bias
        
        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()
            encoded_info['epoch'] = self.config.epoch

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))
            strategy_loss = self.gate_criteria(encoded_info['strategy_logits'], encoded_info['strat_id'].long())

            scaled_epoch = np.log(encoded_info['epoch'] + 1)
            max_scaled_epoch = np.log(self.config.epoch)
            alpha = (max_scaled_epoch - scaled_epoch) / max_scaled_epoch
            alpha = max(alpha, 0)

            loss_weight = self.config.loss_weight if hasattr(self.config, 'loss_weight') else 1.0
            loss_weight = loss_weight * (1 - alpha)
            
        if masked_lm_loss is not None:
            masked_lm_loss =  (1 - loss_weight) * masked_lm_loss + loss_weight * strategy_loss

        if not self.training and not validation: # inference
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value}
            return res

        else: # validation
            assert not self.training
            return loss, label_size

    def predict_strategy(self, logits, data_name, knowledge_name, encoded_info, gate_value=None, flag=None):
        # assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        # logits = logits[:, 0, -8:]
        
        if knowledge_name == 'none':
            if data_name == 'esconv':
                logits = logits[:, 0, -8:]
            elif data_name == 'mi':
                logits = logits[:, 0, -10:]
        elif knowledge_name == 'basic':
            if data_name == 'esconv':
                logits = logits[:, 0, -13:-5]
            elif data_name == 'mi':
                logits = logits[:, 0, -15:-5]
        elif knowledge_name == 'bm25':
            if data_name == 'esconv':
                logits = logits[:, 0, -9:-1]
            elif data_name == 'mi':
                logits = logits[:, 0, -11:-1]
        elif knowledge_name == 'oracle':
            if data_name == 'esconv':
                logits = logits[:, 0, -14:-6]
            elif data_name == 'mi':
                logits = logits[:, 0, -16:-6]
        elif knowledge_name in ['sbert','graph']:
            if data_name == 'esconv':
                logits = logits[:, 0, -16:-8]
            elif data_name == 'mi':
                logits = logits[:, 0, -18:-8]
    
        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                # top_p_warper = TopPLogitsWarper(top_p=0.9)
                # filtered_logits = top_p_warper(None, logits / TEMPERATURE)
                filtered_logits = self.top_p_warper(None, logits)
                filtered_logits = self.top_k_warper(None, filtered_logits)
                filtered_logits = self.temp_warper(None, filtered_logits)
                print(filtered_logits)
                filtered_logits = self.repetition_penalty_processor(None, filtered_logits)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)
        
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
    
        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1),
            'strategy_logits': logits
        })

        if flag == 'before':

            encoded_info.update({
                'before_ffn' : pred,
            })

        elif flag == 'after':

            encoded_info.update({
                'after_ffn' : pred,
            })

            encoded_info['gate_values'].append(gate_value.detach().cpu().numpy())

        return pred
    
    @torch.no_grad()
    def generate(
        self,
        data_name,
        knowledge_name,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None
        
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # self.hidden_states = []

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias

        flag = 'before'
        strategy = self.predict_strategy(lm_logits,data_name, knowledge_name, encoded_info, flag=flag)

        hidden_states, gate_value = self.star(
            strategy,
            decoder_outputs.last_hidden_state,
            dropout=self.config.dropout,
            training=self.training
        )

        lm_logits = self.lm_head(hidden_states) + self.final_logits_bias
        flag = 'after'
        strategy = self.predict_strategy(lm_logits,data_name, knowledge_name, encoded_info, gate_value, flag=flag)
        
        if knowledge_name == 'none':
            if data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 8], dim=-1)
            elif data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 10], dim=-1)
        elif knowledge_name == 'basic':
            if data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 13], dim=-1)
            elif data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 15], dim=-1)
        elif knowledge_name == 'bm25':
            if data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 9], dim=-1)
            elif data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 11], dim=-1)
        elif knowledge_name == 'oracle':
            if data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 14], dim=-1)
            elif data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 16], dim=-1)
        elif knowledge_name in ['sbert','graph']:
            if data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 16], dim=-1)
            elif data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 18], dim=-1)
        
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids

        
        special_keys = ['other_res', 'pred_strat_id', 'pred_strat_id_top1', 'pred_strat_id_top3', 'pred_strat_id_dist', 'conv_id', 'strategy_logits', 'batch_size', 'gate_values', 'before_ffn', 'after_ffn']
        special_info = {key: encoded_info.pop(key, None) for key in special_keys}
        
        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )

        encoded_info.update(special_info)

        return encoded_info, generations[:, decoder_input_ids.size(1):]
    
    def load_blender_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)

        print(f"Checkpoint loaded with the following adjustments:")
        if missing_keys:
            print(f" - Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f" - Unexpected keys: {unexpected_keys}")

        print("BlenderBot weights loaded successfully.")
    
    def load_strategy_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        strategy_ffn_weights = {
                key.replace('strategy_ffn.', ''): value
                for key, value in checkpoint.items()
                if key.startswith('strategy_ffn.')
            }
        self.strategy_ffn.load_state_dict(strategy_ffn_weights, strict=True)
        print("Strategy weights loaded successfully.")
