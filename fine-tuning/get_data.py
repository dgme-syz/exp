import json, argparse
from tqdm import tqdm
from utils import Figer_aFet
from config import *

def main(args):
    data = Figer_aFet('./figer_afet', kind=args.kind)
    output_file = f"./fine-tuning/tune_data/train_{args.kind}.json"

    with open(output_file, "w") as f:
        f.write("[\n")
        first_item = True
        i = 0
        for li, true_labels in tqdm(data.get_data, total=train_sample_num):
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
        
if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument('--kind', type=int, default=0)
    paser.add_argument('--limit', type=int, default=100)
    args = paser.parse_args()
    main(args)


