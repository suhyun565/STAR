import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import json
from tqdm import tqdm
import argparse

snowball = SnowballStemmer(language='english')

def ext_freq_term(sentence):
    words = nltk.word_tokenize(sentence)
    new_words= [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in new_words if not w in stop_words]
    tagged = nltk.pos_tag(filtered_words)
    final_words = [snowball.stem(w) for w,p in tagged if p.startswith('N') or p.startswith('V')]
    return final_words


def fingerprint(path, outpath):
    with open(path, 'r') as infile:
        data = json.load(infile)
    
    count = {'E': 0, 'A':0, 'F':0, 'R':0}
    metrics = {'pro':[], 'inf':{'I':[], 'N':[]}, 'rep':{'I':[], 'N':[]}, 'rel':{'I':[], 'N':[]}}
    print(len(data))
    for dial in tqdm(data):
        start = False
        end = False
        for turn in dial:
            if not start:
                if turn['label'] == 'I':
                    start = True
                    continue
                turn['label'] = 'H'
            if 'bye' in turn['text'].lower():
                end = True
            if end:
                turn['label'] = 'B'

        last_emo = None
        for i in range(len(dial)):
            if dial[i]['speaker'] == 'usr' and dial[i]['label'] != 'H':
                last_emo = dial[i]['emotion_intensity']
                continue
            next_emo = None
            for j in range(i+1, len(dial)):
                if dial[j]['speaker'] == 'usr' and dial[j]['label'] != 'B':
                    next_emo = dial[j]['emotion_intensity']
                    break
            if last_emo is not None and next_emo is not None:
                dial[i]['rel'] = next_emo - last_emo

        
        all_words = set()
        usr_words = set()
        for turn in dial:
            words = set(ext_freq_term(turn['text']))
            turn['words'] = list(words)

            inf = len(words - all_words)
            rep = len(words & usr_words)
            #inf = 0 if len(words - all_words) == 0 else len(words - all_words) / float(len(turn['words']))
            #rep = 0 if len(words & usr_words) == 0 else len(words & usr_words) / float(len(turn['words']))

            all_words = all_words | words
            if turn['speaker'] == 'usr':
                usr_words = usr_words | words
                if turn['label'] == 'I':
                    count['E'] += 1
                else:
                    count['F'] += 1
                continue
            if turn['label'] in ['H','B']:
                metrics['pro'].append(0)
                count['R'] += 1
                continue
            
            if turn['speaker'] == 'sys':
                turn['inf'] = inf
                turn['rep'] = rep

                if turn['label'] == 'I':
                    metrics['pro'].append(1)
                    count['A'] += 1
                else:
                    metrics['pro'].append(0)
                    count['R'] += 1
                metrics['inf'][turn['label']].append(inf)
                metrics['rep'][turn['label']].append(rep)
                if 'rel' in turn:
                    metrics['rel'][turn['label']].append(turn['rel'])
    
    print(float(sum(metrics['pro']))/len(metrics['pro']))
    print(float(sum(metrics['inf']['I']))/len(metrics['inf']['I']))
    print(float(sum(metrics['inf']['N']))/len(metrics['inf']['N']))
    print(float(sum(metrics['inf']['I'])+sum(metrics['inf']['N']))/len(metrics['inf']['I']+metrics['inf']['N']))
    print(float(sum(metrics['rep']['I']))/len(metrics['rep']['I']))
    print(float(sum(metrics['rep']['N']))/len(metrics['rep']['N']))
    print(float(sum(metrics['rep']['I'])+sum(metrics['rep']['N']))/len(metrics['rep']['I']+metrics['rep']['N']))
    print(float(sum(metrics['rel']['I']))/len(metrics['rel']['I']))
    print(float(sum(metrics['rel']['N']))/len(metrics['rel']['N']))
    print(float(sum(metrics['rel']['I'])+sum(metrics['rel']['N']))/len(metrics['rel']['I']+metrics['rel']['N']))
    print(count)

    with open(outpath,'w') as outfile:
        json.dump(data, outfile, indent=2)


fingerprint('esconv_all_emo.json','fingerprint_esconv.json')
fingerprint('emp_dial_emo.json','fingerprint_emp.json')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="converted_dataset")
    parser.add_argument("--out_path", type=str, default="fingerprint.json")

    args = parser.parse_args()

    fingerprint(args.data_path, args.out_path)
            
