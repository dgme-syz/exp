import os
import json
from tqdm import tqdm

data_path = "./onenotes/"
output_path = "./OneNotes_JSON/"

def fetch_one_item(s: str) -> dict:
    string_list = s.split('\t')
    start_index, end_index = int(string_list[0]), int(string_list[1])
    
    sentence = string_list[2].split()
    labels = []
    for i in range(3, len(string_list)):
        if string_list[i][0] == '/':
            labels.append(string_list[i])
    return dict({
        "tokens": sentence,
        "mentions": [{
            "start": start_index,
            "labels": label,
            "end": end_index,
        } for label in labels],
    })
    
def main() -> None:
    for file in os.listdir(data_path):
        if file.endswith('.txt'): 
            path = os.path.join(data_path, file)
            pos = os.path.join(output_path, file.removesuffix('.txt') + '.json')
            
            print(pos)
            if os.path.exists(pos):
                with open(pos, 'w') as f:
                    pass
            else:
                os.makedirs(os.path.dirname(pos), exist_ok=True)
            with open(path, 'r') as f:
                for line in tqdm(f):
                    json.dump(fetch_one_item(line), open(pos, 'a'))


if __name__ == '__main__':
    main()