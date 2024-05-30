import json, os, random, ollama, re, datetime, argparse
from loguru import logger
from metric import Scores
from collections import OrderedDict

def init_figer_afet(path='./figer_afet'):
    # labels: 存储无重复所有标签
    labels = OrderedDict()
    with open(os.path.join(path, 'train.json'), 'r') as f:
        for line in f:
            data = json.loads(line)
            for entity in data['mentions']:
                for label in entity['labels']:
                    labels.setdefault(label, 0)
                    labels[label] += 1
    with open(os.path.join(path, 'labels.json'), 'w') as f:
        json.dump(labels, f)
        
    return labels

def make_tree(path):
    # 离散化标签
    # items 表示索引 -> 标签; mp 表示标签 -> 索引
    if os.path.exists(os.path.join(path, 'labels.json')):
        with open(os.path.join(path, 'labels.json'), 'r') as f:
            labels = json.load(f)
    else:
        labels = init_figer_afet(path)
    items, mp = ['/'], {'/':0}
    
    for i, label in enumerate(labels):
        items.append(label)
        mp[label] = i + 1
    # 建立邻接表
    # /person/artist 说明 /person/artist 是 /person 的子类
    # 一般来说，标签很多，而标签本身叠加的深度不会很深
    adj = [[] for _ in range(len(labels) + 1)]
    parent = {}
    # adj[i] 表示第 i 个标签的子类
    for x in labels:
        last = x.rfind('/')
        par = x[:last] if last != 0 else '/'
        adj[mp[par]].append(mp[x])
        parent[x] = par
        
    
    # 邻接表建树一大好处是，标签父子关系体现在子树中，可以用 dfs 序表征
    # x 是 y 的父类，那么 x 的 dfs 序号一定在 y 的 dfs 序号之前
    
    dfn, idx = [0] * (len(labels) + 1), 0
    def dfs(u):
        nonlocal idx
        dfn[u] = idx
        idx += 1
        for v in adj[u]:
            dfs(v)
    dfs(0) # 从 '/' 开始 dfs
    return items, mp, dfn, parent

def make_prompt(sentence, mentions, ord):
    # kind : 0 表示实体类别粗到细，1 表示实体类别细到粗，2 表示实体类别随机    
    prompt = f"""[Task]: Fine-grained entity classification
[sentence]: "Apple Inc. unveiled a new smartphone called iPhone 13."
[entity]: apple
[entity types]: ['musician', 'artist', 'person', 'organization', 'company']
[Fine-Grained Entity Classification Result]:  'organization', 'company' 

Now I will give you a problem as the above example, you just need to output the Fine-Grained Entity Classification Result

"""
    rprompt = prompt

    prompt += f"""[Task]: Fine-grained entity classification
[sentence]: "{sentence}"
[entity]: {mentions}
[entity types]: {ord}
[Fine-Grained Entity Classification Result]: 
"""
    rprompt += f"""[Task]: Fine-grained entity classification
[sentence]: "{sentence}"
[entity]: {mentions}
[entity types]: {ord[::-1]}
[Fine-Grained Entity Classification Result]:
"""

    prompt += f"""[Warning]: Just output nothing except entity types above, separate them by spaces, there may be more than one answer"""
    rprompt += f"""[Warning]: Just output nothing except entity types above, separate them by spaces, there may be more than one answer"""
    return prompt, rprompt

def get_suf(x):
    return x[x.rfind('/')+1:]

N = 1

class Figer_aFet():
    def __init__(self, path, kind=0) -> None:
        self.path = path
        if os.path.exists(os.path.join(path, 'labels.json')):
            with open(os.path.join(path, 'labels.json'), 'r') as f:
                self.labels = json.load(f)
        else:
            self.labels = init_figer_afet(path)
        self.items, self.mp, self.dfn, self.parent = make_tree(path)
        
        # mp : {"/person/arist": 1, "/person": 2, ...}
        # tot : {"person": 1, "artist": 2, ...}
        self.ord = self.items[1:]
        self.tot = {}
        self.ret = {}
        for x in self.items[1:]:
            self.ret[get_suf(x)] = x
        if kind == 0:
            self.ord.sort(key=lambda x: self.dfn[self.mp[x]])
        elif kind == 1:
            self.ord.sort(key=lambda x: self.dfn[self.mp[x]], reverse=True)
        elif kind == 2:
            random.shuffle(self.ord)
        else:
            self.ord = [get_suf(x) for x in self.ord]
            self.ord.sort() # 字典序排序
        if kind != 3:
            self.ord = [get_suf(x) for x in self.ord]
        for i in range(len(self.ord)):
            self.tot[self.ord[i]] = i + 1
        
    @property
    def get_data(self):
        with open(os.path.join(self.path, 'train.json'), 'r') as f:
            for line in f:
                data = json.loads(line)
                sentence = data['tokens']
                for entity in data['mentions']:
                    mentions = sentence[entity['start']:entity['end']]
                    true_labels = entity['labels']
                    prompts = []
                    for i in range(N):
                        prompt, rprompt = make_prompt(' '.join(sentence), ' '.join(mentions), self.ord)
                        prompts.extend([prompt, rprompt])
                    yield prompts, [get_suf(x) for x in true_labels]

limit = 200 

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--limit', type=int, default=200)
    paser.add_argument('--kind', type=int, default=1)
    paser.add_argument('--reverse', type=bool, default=False)
    paser.add_argument('--N', type=int, default=1)
    paser.add_argument('--model', type=str, default="llama3:8b")
    args = paser.parse_args()
    data = Figer_aFet('./figer_afet', kind=args.kind)
    sample = 0
    N = args.N
    limit = args.limit
    metric = Scores()
    logger.add('./logs1/test0_type_' + str(args.kind) + f"_{args.limit}_" + \
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.log')

    
    for li, true_labels in data.get_data:
        pred_info = ''
        for i in range(len(li)):     
            if args.reverse is False and i > 0:
                continue
            mes = [
                {"role": "system", "content": "You are a excellent linguist, you can finish the following task well! Also, you need to recognize some entity types are relative."},
                {"role": "user", "content": li[i]},
            ]
            response = ollama.chat(model=args.model, messages=mes)
            pred_info = pred_info + ' ' + response['message']['content']
        # print(rcontx)
        # print(response)
        # print(f"output : {pred_info}")
        # print(true_labels)
        # print(data.tot)
        # print(true_labels)
        try:
            li = pred_info.split(' ')
            pred_labels = {}
            for x in li:
                if x in data.tot.keys():
                    u = data.ret[x]
                    pred_labels[get_suf(u)] = 1
                    while u != '/':
                        u = data.parent[u]
                        if  u == '/':
                            break
                        pred_labels[get_suf(u)] = 1
            pred_labels = list(pred_labels.keys())
            if len(pred_labels) == 0:
                raise Exception('pred_labels is None')
            print(pred_labels, '\n')
            print(true_labels, '\n')
            logger.info(f"pred_labels: {pred_labels}, true_labels: {true_labels}")
        except:
            continue
        else:
            metric.update([data.tot[_] for _ in true_labels], [data.tot[_] for _ in pred_labels])
            info = metric.evaluate
            contents = f"accuracy: {info['accuracy']}, macro_f1: {info['macro_f1']}, micro_f1: {info['micro_f1']}\n\n"
            
            print(contents)
            logger.info(contents)
            
            sample += 1
            if (sample == limit):
                break

    
    
    