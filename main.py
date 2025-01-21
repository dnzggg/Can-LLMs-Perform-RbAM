import argparse
import json
import os
import time
from glob import glob

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import prompt
from llm_managers import HuggingFaceLlmManager, OpenAiLlmManager

if __name__ == "__main__":
    rbam_prompt_class = prompt.RbAMPrompts()
    rbam_prompts = [func for func in dir(rbam_prompt_class) if "__" not in func]

    datasets = list(map(os.path.basename, glob("Datasets/*")))

    parser = argparse.ArgumentParser(description="CanLLMsPerformRbAM")
    # model related args
    parser.add_argument(
        "--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    parser.add_argument(
        "--dataset-name", type=str, default=["all"], action="extend", nargs="*", choices=datasets + ["all"]
    )
    parser.add_argument("--save-loc", type=str, default="Results/")
    parser.add_argument(
        "--cache-dir", type=str, default="/vol/bitbucket/dg1822/cache"
    )
    parser.add_argument("--cuda-visible-devices", type=str, default='-1')
    # model parameter args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"]
    )
    parser.add_argument("--apply-template", action=argparse.BooleanOptionalAction, default=False)
    # generation related args
    parser.add_argument(
        "--rbam-prompt", type=str, choices=rbam_prompts + ["all"], default="all"
    )
    parser.add_argument("--nr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--instruction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-shot", type=str, default="0")
    parser.add_argument("--primer", type=str, default="seed_1.csv")
    # experiment type
    parser.add_argument("--prompt-experiment", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    print("Loading model...")
    if args.cuda_visible_devices != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if "openai" in args.model_name:
        llm_manager = OpenAiLlmManager(
            model_name=args.model_name,
        )
    else:
        llm_manager = HuggingFaceLlmManager(
            model_name=args.model_name,
            quantization=args.quantization,
        )
    
    generation_args = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
    }

    if args.rbam_prompt != "all":
        rbam_prompts = [args.rbam_prompt]

    if (len(args.dataset_name) > 1 and args.dataset_name[1] != "all") or (len(args.dataset_name) == 1 and args.dataset_name[0] != "all"):
        datasets = args.dataset_name
        datasets.pop(0)

    for rbam_prompt in rbam_prompts:
        print(
            f"Prompt experiment with RbAM prompt: {rbam_prompt}"
        )
        generate_prompt = getattr(rbam_prompt_class, rbam_prompt)

        for dataset_name in datasets:
            filename = os.path.join(args.save_loc, f"{args.model_name.split('/')[1]}_{dataset_name}_{rbam_prompt}_{args.n_shot}-shot_Ins{args.instruction}.json")
            
            if os.path.exists(filename):
                print(f"Results exist for {dataset_name} with {rbam_prompt}. So passing.")
                continue
            
            if "NR" in dataset_name and not args.nr:
                continue
            elif "NR" not in dataset_name and args.nr:
                continue

            if not args.prompt_experiment and "AllDataset" in dataset_name:
                continue
            elif args.prompt_experiment and "AllDataset" not in dataset_name:
                continue

            print("Loading dataset " + dataset_name)
            dataset = load_from_disk("Datasets/" + dataset_name)
            
            start = time.time()
            results = []
            
            # This is for having a checkpoint
            if not os.path.exists(args.save_loc):
                os.makedirs(args.save_loc)
            
            log = os.path.join(args.save_loc, f"{args.model_name.split('/')[1]}_{dataset_name}_{rbam_prompt}_{args.n_shot}-shot_Ins{args.instruction}.log")
            
            if os.path.exists(log):
                with open(log) as file:
                    data = json.load(file)
                    results = data["results"]
                    start = start - data["time"]["total"]
            
            progress_bar = tqdm(total=len(dataset), initial=len(results))
            
            for i, data in enumerate(dataset):
                if i < len(results):
                    continue
                if i % 100 == 0:
                    with open(log, "w") as file:
                        json.dump({"results": results, "time": {"total": time.time() - start}}, file)

                generated_prompt, constraints, format_fun = generate_prompt(
                    data["arg1"], data["arg2"], n_shot=args.n_shot, instruction=args.instruction, nr=args.nr,
                    primer=args.primer
                )
                prediction = format_fun(
                    llm_manager.chat_completion(
                        generated_prompt,
                        print_result=False,
                        trim_response=True,
                        apply_template=args.apply_template,
                        **constraints,
                        **generation_args,
                    )
                )
                progress_bar.update(1)
                results.append(prediction)

            os.remove(log)
            progress_bar.close()
            end = time.time()

            metrics = {
                "accuracy": (accuracy_score, {}),
                "f1": (f1_score, {"average": None}),
                "f1_b": (f1_score, {"average": "weighted"})
            }

            eval_results = {}
            for metric, (metric_fun, parameters) in metrics.items():
                result = metric_fun(dataset["support"], results, **parameters)
                if isinstance(result, np.ndarray):
                    for i, res in enumerate(result):
                        eval_results[metric + "_" + str(i)] = res
                else:
                    eval_results[metric] = result

            experiment_summary = {
                    "arguments": vars(args),
                    "results": results,
                    "labels": dataset["support"],
                    "eval_results": eval_results,
                    "time": {"total": end - start, "average": (end - start) / len(dataset)}
            }
            
            with open(filename, "w") as f:
                json.dump(experiment_summary, f, indent=4)

