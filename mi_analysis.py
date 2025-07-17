from metric.metric_utils import mi_metric
import os
import json
import nltk

def test_mi():
    infer_input_file = "./_reformat/esconv/none/test.txt"
    save_dir = "best_model_dir"

    mi_res = mi_metric(infer_input_file, os.path.join(save_dir, f'gen.json'), os.path.join(save_dir, f'mi_gen.json'), False, True)


def find_case():
    save_dir = "best_model_dir/mi_gen.json"
    
    highest_bleu = 0
    with open(save_dir, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        for d in data:
            dial = d['dialog']
            for utt in dial:
                if utt['speaker'] != 'sys':
                    continue
                if utt['strategy'] != 'Providing Suggestions':
                    continue
                reference = nltk.word_tokenize(utt['text'])
                references = [reference]
                try:
                    hypothesis = nltk.word_tokenize(utt['generated_respons'])
                except Exception as e:
                    continue
                bleu = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
                if bleu > highest_bleu:
                    highest_bleu = bleu
                    case = utt['text']
                    print(bleu, references, hypothesis)
    print(case)

find_case()