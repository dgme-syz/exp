import json, argparse
from tqdm import tqdm
from utils import FET
from config import *

def main(args):
    data = FET(args.path, kind=args.kind)
    name = args.path.split("/")[-1]
    output_file = f"./fine-tuning/tune_data/train_{args.kind}_{name}.json"

    with open(output_file, "w") as f:
        f.write("[\n")
        first_item = True
        i, x = 0, 0
        for li, true_labels in tqdm(data.get_data, total=train_sample_num):
            if x >= args.limit * 0.58 and len(true_labels) <= 3:
                continue
            if len(true_labels) <= 3:
                x += 1
            item = {
                "instruction": "You are an excellent linguist, you can finish the following task well! Also, you need to recognize some entity types are relative.",
                "input": li[0],
                "output": " ".join(true_labels)
            }
            if not first_item:
                f.write(",\n")
            json.dump(item, f, indent=4)
            first_item = False
            i = i + 1
            if i >= args.limit:
                break
        f.write("\n]")
    print(f"Write {i} items to {output_file}")
        
if __name__ == "__main__":
    # python -m fine-tuning.get_data --kind 0 --limit 1000
    paser = argparse.ArgumentParser()
    paser.add_argument('--path', type=str, default='./figer_afet')
    paser.add_argument('--kind', type=int, default=0)
    paser.add_argument('--limit', type=int, default=5000)
    args = paser.parse_args()
    main(args)


