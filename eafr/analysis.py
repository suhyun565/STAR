import json

def dialogue_flow(path):
    with open(path, 'r') as infile:
        data = json.load(infile)
    
    df_dict = {}
    label_dict = {'E':0, 'F':0, 'A':0, 'R':0}
    for dial in data:
        start = False
        last = 'H'
        for turn in dial:
            if turn['speaker'] == 'usr':
                if turn['label'] == 'I':
                    current = 'E'
                else:
                    current = 'F'
            else:
                if turn['label'] == 'I':
                    current = 'A'
                else:
                    current = 'R'
            if not start:
                if turn['label'] == 'I':
                    start = True
                    if last+current not in df_dict:
                        df_dict[last+current] = 0
                    df_dict[last+current] += 1
                    last = current
                    label_dict[current] += 1
                continue
            if 'bye' in turn['text'].lower():
                current = 'B'
                if last+current not in df_dict:
                    df_dict[last+current] = 0
                df_dict[last+current] += 1
                break
            label_dict[current] += 1
            if last+current not in df_dict:
                df_dict[last+current] = 0
            df_dict[last+current] += 1
            last = current
    print(df_dict)
    label_sum = sum(label_dict.values())
    for key in label_dict:
        label_dict[key] = label_dict[key]/float(label_sum)
    
    inter_dict = {}
    for key in ['ER','RE','EA','AE','RF','FR','AF','FA']:
        inter_dict[key] = df_dict[key]
    inter_sum = sum(inter_dict.values())
    for key in inter_dict:
        inter_dict[key] = inter_dict[key]/float(inter_sum)

    bye_dict = {}
    for key in ['EB','RB','AB','FB']:
        if key in df_dict:
            bye_dict[key] = df_dict[key]
    bye_sum = sum(bye_dict.values())
    for key in bye_dict:
        bye_dict[key] = bye_dict[key]/float(bye_sum)
    
    hi_dict = {}
    if 'HE' in df_dict and 'HA' in df_dict:
        for key in ['HE', 'HA']:
            hi_dict[key] = float(df_dict[key])/(df_dict['HE']+df_dict['HA'])

    return label_dict, inter_dict, hi_dict, bye_dict

print(dialogue_flow('fingerprint_esconv.json'))
print(dialogue_flow('fingerprint_emp.json'))


def progress(path):
    with open(path, 'r') as infile:
        data = json.load(infile)

    relax = {'I':[], 'N': []}
    eafr_stat = [{'I':0,'N':0,'H':0,'B':0}, {'I':0,'N':0,'H':0,'B':0}, {'I':0,'N':0,'H':0,'B':0}, {'I':0,'N':0,'H':0,'B':0}, {'I':0,'N':0,'H':0,'B':0}]
    emo_stat = [{'I':[], 'N':[]}, {'I':[], 'N':[]}, {'I':[], 'N':[]}, {'I':[], 'N':[]}, {'I':[], 'N':[]}]
    for dial in data:
        dlen = len(dial)
        d1 = dial[:int(0.2*dlen)]
        d2 = dial[int(0.2*dlen):int(0.4*dlen)]
        d3 = dial[int(0.4*dlen):int(0.6*dlen)]
        d4 = dial[int(0.6*dlen):int(0.8*dlen)]
        d5 = dial[int(0.8*dlen):]

        for utt in dial:
            if utt['speaker'] == 'sys':
                if 'rel' in utt and utt['label'] in ['I', 'N']:
                    relax[utt['label']].append(utt['rel'])

        for utt in d1:
            if utt['speaker'] == 'sys':
                eafr_stat[0][utt['label']] += 1
                if 'rel' in utt and utt['label'] in ['I', 'N']:
                    emo_stat[0][utt['label']].append(utt['rel'])
        
        for utt in d2:
            if utt['speaker'] == 'sys':
                eafr_stat[1][utt['label']] += 1
                if 'rel' in utt and utt['label'] in ['I', 'N']:
                    emo_stat[1][utt['label']].append(utt['rel'])

        for utt in d3:
            if utt['speaker'] == 'sys':
                eafr_stat[2][utt['label']] += 1
                if 'rel' in utt and utt['label'] in ['I', 'N']:
                    emo_stat[2][utt['label']].append(utt['rel'])

        for utt in d4:
            if utt['speaker'] == 'sys':
                eafr_stat[3][utt['label']] += 1
                if 'rel' in utt and utt['label'] in ['I', 'N']:
                    emo_stat[3][utt['label']].append(utt['rel'])

        for utt in d5:
            if utt['speaker'] == 'sys':
                eafr_stat[4][utt['label']] += 1
                if 'rel' in utt and utt['label'] in ['I', 'N']:
                    emo_stat[4][utt['label']].append(utt['rel'])
    
    for s in eafr_stat:
        s_sum = sum(s.values())
        for key in s:
            s[key] = s[key]/s_sum
        print(s)

    for s in emo_stat:
        for key in s:
            if len(s[key]) == 0:
                s[key] = 0
            else:
                s[key] = sum(s[key])/len(s[key])
        print(s)

    print(float(sum(relax['I']))/len(relax['I']))
    print(float(sum(relax['N']))/len(relax['N']))

progress('fingerprint_esconv.json')
progress('fingerprint_emp.json')