# coding=utf-8

import argparse
import json
import logging
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import Tensor
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer
import openai

import time
import random

from inputters import inputters
from inputters.inputter_utils import _norm
from metric.myMetrics import Metric
from utils.building_utils import boolean_string, build_model, deploy_model
from utils.eval_utils import eval_model_loss
from models import models
import nltk

from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

from difflib import get_close_matches

import json
import random

# 데이터 읽기: 각 줄을 JSON 객체로 변환
with open("/home/station_04/Desktop/Emotional-Support-Conversation/codes_zcj/_reformat/mi/train.txt", 'r', encoding='utf-8') as f:
    train_data = [json.loads(line.strip()) for line in f if line.strip()]  # 공백 제거 후 JSON 로드

def get_example_dict(train_data):
    """
    주어진 데이터에서 각 전략에 따른 예제를 분류하여 반환하는 함수.
    """
    # 정의된 전략들
    strategies = [
        'GIV', 'QUEST', 'SEEK', 'AF', 'PWP', 'PWOP',
        'EMPH', 'CON', 'SR', 'CR'
    ]
    
    # 전략별 예제를 저장할 딕셔너리 초기화
    st_to_example = {strategy: [] for strategy in strategies}
    
    for conv in train_data:
        # 필수 키 존재 여부 확인
        if "dialog" not in conv or "strategy" not in conv or "target" not in conv:
            continue
        
        # 데이터 추출
        dialog = conv.get("dialog", [])
        strategy = conv.get("strategy", "").strip()
        target = conv.get("target", "").strip()
        
        # 유효한 전략인지 확인
        if strategy not in strategies:
            continue
        
        seekers = {}
        supporters = {}
        seeker_turn = 0
        supporter_turn = 0
        
        # 대화 순회
        for turn, content in enumerate(dialog):
            if turn % 2 == 0:  # 짝수 턴: seeker
                seekers[seeker_turn] = content.strip()
                seeker_turn += 1
            else:  # 홀수 턴: supporter
                if seeker_turn > 0:  # 이전에 seeker 발화가 있어야 함
                    supporters[supporter_turn] = {
                        "content": content.strip(),
                        "strategy": strategy,
                        "target": target
                    }
                    st_to_example[strategy].append({
                        "seeker": seekers.get(seeker_turn - 1, ""),
                        "supporter": content.strip(),
                        "target": target
                    })
                    supporter_turn += 1
    
    return st_to_example


def load_few_shot_exemplar(st_to_example, strategy=None):
    """
    주어진 전략의 몇 가지 예제를 샘플링하여 반환.
    """
    # 전략별 예제 가져오기
    if strategy not in st_to_example:
        return ""  # 유효하지 않은 전략이면 빈 문자열 반환

    turns = st_to_example.get(strategy, [])
    if not turns:
        return ""  # 예제가 없으면 빈 문자열 반환

    # 최대 20개의 샘플을 랜덤 선택
    turns = random.sample(turns, min(len(turns), 20))
    
    # 샘플을 대화 형식으로 병합
    few_shot_dialogue = []
    for t in turns:
        few_shot_dialogue.append(f"seeker: {t['seeker']}")
        few_shot_dialogue.append(f"supporter: {t['supporter']}")
        few_shot_dialogue.append(f"target: {t['target']}")  # 타겟 추가
    return "\n".join(few_shot_dialogue)


# 데이터에서 전략별 예제 생성
st_to_example = get_example_dict(train_data)

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary containing strategies and their definitions

strategy_dict = {
    "[GIV]": 'GIV',
    "[QUEST]": 'QUEST',
    "[SEEK]": 'SEEK',
    "[AF]": 'AF',
    "[PWP]": 'PWP',
    "[PWOP]": 'PWOP',
    "[EMPH]": 'EMPH',
    "[CON]": 'CON',
    "[SR]": 'SR',
    "[CR]": 'CR'
}

def classify_strategy_with_gpt(prompt):
    classification_prompt = f"""
    Based on the following conversation, classify which strategy the supporter should use to best respond to the seeker.
    The possible strategies are: {', '.join(st_to_example.keys())}.
    Conversation: {prompt}
    Output must be exactly one of the above keys (e.g., SEEK, GIV) with no additional text.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a classifier that identifies strategies in conversations."},
            {"role": "user", "content": classification_prompt}],
        max_tokens=20,
    )
    return response.choices[0].message.content.strip()

# GPT 모델을 사용하여 응답 생성
def generate_response_with_gpt4(strategy, prompt, example_context=None):
    gpt_prompt_few_shot = f"""
    1. Respond to the prompt using the strategy ‘{strategy}’
    2. The following examples demonstrate that the supporter is using the specified strategy. Consider them as a guide for your response:
    {example_context}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": gpt_prompt_few_shot},
            {"role": "user", "content": prompt}],
        max_tokens=40,
    )
    return response.choices[0].message.content

def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent

def transform_string(input_str: str) -> str:
    """
    주어진 문자열에서 모든 공백을 제거하고,
    대문자로 변환한 뒤,
    [ ... ] 형태로 감싸서 반환합니다.
    """
    # 1) 모든 공백 제거
    without_spaces = input_str.replace(" ", "")
    
    # 2) 대문자로 변환
    upper_str = without_spaces.upper()
    
    # 3) 대괄호로 감싸기
    if len(input_str) >= 2 and input_str.startswith('[') and input_str.endswith(']'):
        pass
    else:
        upper_str = f"[{upper_str}]"
    
    return upper_str


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--inputter_name', type=str, required=True)
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--knowledge_name', type=str, default=None)
parser.add_argument('--custom_name', type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_checkpoint", '-c', type=str, default=None)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--max_input_length", type=int, default=150)
parser.add_argument("--max_src_turn", type=int, default=None)
parser.add_argument("--max_decoder_input_length", type=int, default=50)
parser.add_argument("--max_knowledge_length", type=int, default=None)
parser.add_argument('--label_num', type=int, default=None)
parser.add_argument('--multi_knl', action='store_true', help='allow candidate knowledge items')
parser.add_argument('--only_encode', action='store_true', help='only do encoding')
parser.add_argument('--only_generate', action='store_true', help='do not conduct evaluations')
parser.add_argument('--add_nlg_eval', action='store_true', help='add nlg-eval')
parser.add_argument("--min_length", type=int, default=5)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--infer_batch_size", type=int, default=16)
parser.add_argument('--infer_input_file', type=str, nargs='+', required=True)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument("--length_penalty", type=float, default=1.0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
parser.add_argument('--orpo', type=boolean_string, default=False, help='use orpo')
parser.add_argument('--gate_loss', type=boolean_string, default=False, help='use gate loss')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

logger.info('initializing cuda...')
_ = torch.tensor([1.], device=args.device)

set_seed(args.seed)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))


names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
    'data_name': args.data_name,
    'knowledge_name': args.knowledge_name,
}

with open(f'./CONFIG/{args.config_name}.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    
if args.data_name == 'mi':
    sp_tokens = config['expanded_vocab']['mi']
elif args.data_name == 'esconv':
    sp_tokens = config['expanded_vocab']['esconv']

inputter = inputters[args.inputter_name]()
toker = AutoTokenizer.from_pretrained(config['pretrained_model_path'])
toker.add_tokens(sp_tokens, special_tokens=True)

config['st_config']['strategy_categories'] = sp_tokens
Model = models[config['model_name']]
device = "cuda"

if config['model_name'] in ['gate_blenderbot_small', 'gate_blenderbot_small_gate_loss']:
    model = Model.from_pretrained(config['pretrained_model_path'], toker, **config)
else:
    model = Model.from_pretrained(config['pretrained_model_path'], toker)

model.resize_token_embeddings(len(toker))
# model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
logger.info('Model loaded from {}'.format(args.load_checkpoint))

model.to(device)
model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

model.eval()

inputter = inputters[args.inputter_name]()
dataloader_kwargs = {
    'max_src_turn': args.max_src_turn,
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knowledge_length': args.max_knowledge_length,
    'label_num': args.label_num,
    'multi_knl': args.multi_knl,
    'only_encode': args.only_encode,
    'infer_batch_size': args.infer_batch_size,
}

pad = toker.pad_token_id if toker.pad_token_id is not None else toker.eos_token_id
bos = toker.bos_token_id if toker.bos_token_id is not None else toker.cls_token_id
eos = toker.eos_token_id if toker.eos_token_id is not None else toker.sep_token_id

generation_kwargs = {
    'max_length': args.max_length,
    'min_length': args.min_length,
    'do_sample': True if (args.top_k > 0 or args.top_p < 1) else False,
    'temperature': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'num_beams': args.num_beams,
    'num_return_sequences': args.num_return_sequences,
    'length_penalty': args.length_penalty,
    'repetition_penalty': args.repetition_penalty,
    'no_repeat_ngram_size': args.no_repeat_ngram_size,
    'encoder_no_repeat_ngram_size': args.no_repeat_ngram_size if model.config.is_encoder_decoder else None,
    'pad_token_id': pad,
    'bos_token_id': bos,
    'eos_token_id': eos,
}
print(json.dumps(generation_kwargs, indent=2, ensure_ascii=False))

for infer_idx, infer_input_file in enumerate(args.infer_input_file):
    set_seed(args.seed)
    infer_input_file += (args.data_name + '/' + args.knowledge_name + '/test.txt')
    infer_dataloader = inputter.infer_dataloader(
        infer_input_file,
        toker,
        args.data_name,
        args.knowledge_name,
        **dataloader_kwargs
    )
    metric_res = {}
    if not args.only_encode and not args.only_generate:
        loss_loader = inputter.valid_dataloader(
            corpus_file=infer_input_file,
            toker=toker,
            batch_size=args.infer_batch_size,
            data_name=args.data_name,
            knowledge_name=args.knowledge_name,
            **dataloader_kwargs
        )
        # infer_loss, _, infer_samples, pointwise_loss, pointwise_sample = eval_model_loss(
        #     model=model,
        #     eval_dataloader=loss_loader,
        #     epoch_id=0,
        #     infer=True,
        #     args=args,
        # )
        # assert len(pointwise_loss) == len(pointwise_sample)
        # metric_res['perplexity'] = float(np.exp(infer_loss))
        
        # ptr = 0
    
    if not args.only_generate:
        metric = Metric(toker)
    
    res = []
    other_res = {}
    gate_values = []
    decode = lambda x: _norm(toker.decode(x))

    conversation_history = str()

    for idx ,(batch, posts, references, sample_ids) in enumerate(infer_dataloader):

        if idx == 0:
            current_conv_id = sample_ids[0]
        
        new_conv_id = sample_ids[0]

        if new_conv_id != current_conv_id:
            conversation_history = str()

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch.update(generation_kwargs)
        batch.pop('batch_size')
        batch['gate_values'] = []
        
        # encoded_info, generations = model.generate(data_name=args.data_name, knowledge_name=args.knowledge_name, **batch)
        
        # batch_other_res = None
        # if 'other_res' in batch:
        #     batch_other_res = batch.pop('other_res')

        #     add_acc = 'acc_map' in batch_other_res and any(k in batch_other_res and v in encoded_info for k, v in batch_other_res['acc_map'].items())
        #     if add_acc:
        #         if 'acc' not in other_res:
        #             other_res['acc'] = {}
        #         if 'acc_map' not in other_res:
        #             other_res['acc_map'] = batch_other_res['acc_map']
                
        #         for k, v in batch_other_res['acc_map'].items():
        #             if k not in batch_other_res or v not in encoded_info: # TODO
        #                 continue # TODO
        #             batch_other_res[k] = batch_other_res[k].tolist()
        #             encoded_info[v] = encoded_info[v].tolist()
        #             if f'{v}_top1' in encoded_info:
        #                 encoded_info[f'{v}_top1'] = encoded_info[f'{v}_top1'].tolist()
        #             if f'{v}_top3' in encoded_info:
        #                 encoded_info[f'{v}_top3'] = encoded_info[f'{v}_top3'].tolist()
        #             if f'{v}_dist' in encoded_info:
        #                 encoded_info[f'{v}_dist'] = encoded_info[f'{v}_dist'].tolist()
                    
        #             if k not in other_res['acc']:
        #                 other_res['acc'][k] = []
        #             other_res['acc'][k].extend(batch_other_res[k])
                    
        #             if v not in other_res['acc']:
        #                 other_res['acc'][v] = []
        #             other_res['acc'][v].extend(encoded_info[v])
                    
        #             if f'{v}_top1' in encoded_info:
        #                 if f'{v}_top1' not in other_res['acc']:
        #                     other_res['acc'][f'{v}_top1'] = []
        #                 other_res['acc'][f'{v}_top1'].extend(encoded_info[f'{v}_top1'])
        #             if f'{v}_top3' in encoded_info:
        #                 if f'{v}_top3' not in other_res['acc']:
        #                     other_res['acc'][f'{v}_top3'] = []
        #                 other_res['acc'][f'{v}_top3'].extend(encoded_info[f'{v}_top3'])
                    
        #             if f'{v}_dist' in encoded_info:
        #                 if f'{v}_dist' not in other_res['acc']:
        #                     other_res['acc'][f'{v}_dist'] = []
        #                 other_res['acc'][f'{v}_dist'].extend(encoded_info[f'{v}_dist'])
        
        # if not args.only_encode:
        #     generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]

        
        for idx in range(len(sample_ids)):
            p = posts[idx]
            ref = references[idx]
            # cid = conv_ids[idx]
            strategy = classify_strategy_with_gpt(p)
            
            try:
                strategy = get_close_matches(input_key, strategy_dict.keys(), n=1, cutoff=0.5)
            except:
                strategy = random.choice(list(strategy_dict.keys()))

            time.sleep(random.random() * 3)
            
            conversation_history += p

            updated_post = conversation_history
            
            exemplar_context = str()
            # Load one exemplar dialogue
            st = strategy_dict[strategy]
            example = load_few_shot_exemplar(st_to_example,strategy=st)
            exemplar_context = "\n".join(example)
            
            # # Generate response using GPT-4 zero-shot
            # generated_response = generate_response_with_gpt4(strategy,updated_post)
            
            # Generate response using GPT-4 few-shot
            generated_response = generate_response_with_gpt4(strategy,updated_post,exemplar_context)
            
            tmp_res_to_append = {
                'sample_id': sample_ids[idx], 
                'post': p, 
                'response': ref, 
                'strategy_used': strategy,
                'generation': generated_response
            }

            conversation_history += "\n" + generated_response
            
            res.append(tmp_res_to_append)
            metric.forword([ref], generated_response, chinese=False)
        current_conv_id = new_conv_id

        if idx % 10 == 0:
            logger.info('infer batch %d' % idx)
            checkpoint_dir_path = '/'.join(args.load_checkpoint.split('/')[:-1])
            checkpoint_name = args.load_checkpoint.split('/')[-1]
            infer_input_file_name = infer_input_file.split('/')[-1]
            infer_input_file_name = '.'.join(infer_input_file_name.split('.')[:-1])
            if not args.only_encode:
                save_dir = f'{checkpoint_dir_path}/{args.custom_name}_res_{checkpoint_name}_{infer_input_file_name}_k.{args.top_k}' \
                        f'_p.{args.top_p}_b.{args.num_beams}_t.{args.temperature}_lp.{args.length_penalty}' \
                        f'_rp.{args.repetition_penalty}_ng.{args.no_repeat_ngram_size}'
            else:
                save_dir = f'{checkpoint_dir_path}/res_{checkpoint_name}_{infer_input_file_name}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            with open(os.path.join(save_dir, f'gen.json'), 'w') as f:
                json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=False)
            
            with open(os.path.join(save_dir, f'gen.txt'), 'w') as f:
                for line in res:
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    checkpoint_dir_path = '/'.join(args.load_checkpoint.split('/')[:-1])
    checkpoint_name = args.load_checkpoint.split('/')[-1]
    infer_input_file_name = infer_input_file.split('/')[-1]
    infer_input_file_name = '.'.join(infer_input_file_name.split('.')[:-1])
    if not args.only_encode:
        save_dir = f'{checkpoint_dir_path}/{args.custom_name}_res_{checkpoint_name}_{infer_input_file_name}_k.{args.top_k}' \
                f'_p.{args.top_p}_b.{args.num_beams}_t.{args.temperature}_lp.{args.length_penalty}' \
                f'_rp.{args.repetition_penalty}_ng.{args.no_repeat_ngram_size}'
    else:
        save_dir = f'{checkpoint_dir_path}/res_{checkpoint_name}_{infer_input_file_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    with open(os.path.join(save_dir, f'gen.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=False)
    
    with open(os.path.join(save_dir, f'gen.txt'), 'w') as f:
        for line in res:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    metric_res_list = None
    if not args.only_encode and not args.only_generate:
        metric_res_list = {}
        closed_res = metric.close()
        metric_res.update(closed_res[0])
        metric_res_list.update(closed_res[1])
    
    if not args.only_generate:
        if 'acc' in other_res:
            for k, v in other_res['acc_map'].items():
                if k not in other_res['acc'] or v not in other_res['acc']: # TODO
                    continue # TODO
                kk = np.array(other_res['acc'][k], dtype=int)
                vv = np.array(other_res['acc'][v], dtype=int)
                print(f'{k}: classification_report\n', classification_report(kk, vv))
                with open(os.path.join(save_dir, f'confusion_matrix_{k}.json'), 'w') as f:
                    json.dump(confusion_matrix(kk, vv).tolist(), f)
                    print(f'{k}: confusion_matrix\n', confusion_matrix(kk, vv))
                
                metric_res[f'acc_{k}'] = np.mean(kk == vv)
                metric_res[f'f1_micro_{k}'] = f1_score(kk, vv, average='micro')
                metric_res[f'f1_macro_{k}'] = f1_score(kk, vv, average='macro')
                if metric_res_list is None:
                    metric_res_list = {}
                metric_res_list[f'acc_{k}'] = (kk == vv).astype(int).tolist()
                
                if f'{v}_top1' in other_res['acc']:
                    vv_top1 = np.array(other_res['acc'][f'{v}_top1'], dtype=int)
                    metric_res[f'acc_{k}_top1'] = np.mean(np.sum((kk.reshape(-1, 1) - vv_top1) == 0, axis=-1) != 0)
                    metric_res_list[f'acc_{k}_top1'] = (np.sum((kk.reshape(-1, 1) - vv_top1) == 0, axis=-1) != 0).astype(int).tolist()
                if f'{v}_top3' in other_res['acc']:
                    vv_top3 = np.array(other_res['acc'][f'{v}_top3'], dtype=int)
                    metric_res[f'acc_{k}_top3'] = np.mean(np.sum((kk.reshape(-1, 1) - vv_top3) == 0, axis=-1) != 0)
                    metric_res_list[f'acc_{k}_top3'] = (np.sum((kk.reshape(-1, 1) - vv_top3) == 0, axis=-1) != 0).astype(int).tolist()
    
        with open(os.path.join(save_dir, f'metric.json'), 'w') as f:
            json.dump(metric_res, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        if metric_res_list is not None:
            with open(os.path.join(save_dir, f'metric_list.json'), 'w') as f:
                json.dump(metric_res_list, f)

    if args.add_nlg_eval:
        ref_list = []
        hyp_list = []
        for line in res:
            if isinstance(line['response'], list):
                ref = line['response'][0]
            else:
                ref = line['response']
            ref = ' '.join(nltk.word_tokenize(ref.lower()))
            
            if isinstance(line['generation'], list):
                hyp = line['generation'][0]
            else:
                hyp = line['generation']
            hyp = ' '.join(nltk.word_tokenize(hyp.lower()))
            
            ref_list.append(ref)
            hyp_list.append(hyp)
        
        from metric import NLGEval
        metric = NLGEval()
        metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
        with open(os.path.join(save_dir, f'metric_nlgeval.json'), 'w') as f:
            json.dump(metric_res, f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(os.path.join(save_dir, f'metric_nlgeval_list.json'), 'w') as f:
            json.dump(metric_res_list, f, ensure_ascii=False)


