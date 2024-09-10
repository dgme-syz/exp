import yaml
import os
import argparse

def main(args):
    # train
    with open("./train/template.yaml", "r") as f:
        temp_train = yaml.load(f, Loader=yaml.FullLoader)
    temp_train['dataset'] = args.name
    temp_train['output_dir'] = f"saves/llama3-8b/lora/sft_0_{args.name}"
    with open(f"./train/llama3_exp_{args.name}.yaml", "w") as f:
        yaml.dump(temp_train, f)
    
    # interface
    with open("./interface/template.yaml", "r") as f:
        temp_interface = yaml.load(f, Loader=yaml.FullLoader)
    temp_interface['adapter_name_or_path'] = f"saves/llama3-8b/lora/sft_0_{args.name}"
    with open(f"./interface/llama3_exp_{args.name}.yaml", "w") as f:
        yaml.dump(temp_interface, f)
    
    # api
    with open(f"./api/llama3_lora_{args.name}.sh", "w") as f:
        f.write(f'''CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api \
--model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
--adapter_name_or_path saves/llama3-8b/lora/sft_0_{args.name} \
--template llama3 \
--finetuning_type lora''')

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--name', type=str)
    args = paser.parse_args()
    main(args)