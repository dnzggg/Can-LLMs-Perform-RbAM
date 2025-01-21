from glob import glob
import argparse
import json
import os

from sklearn.metrics import f1_score, accuracy_score


parser = argparse.ArgumentParser(description="Analyse")
parser.add_argument(
    "--nr", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "--n-shot", type=str, default="0"
)
parser.add_argument(
    "--instruction", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--apply-template", action=argparse.BooleanOptionalAction, default=False
)
args = parser.parse_args()


datasets = list(map(os.path.basename, glob("Datasets/*")))
models = {"llama70b": [0] * (4 if args.nr else 3), "mistral": [0] * (4 if args.nr else 3), "mixtral": [0] * (4 if args.nr else 3), "gpt35": [0] * (4 if args.nr else 3)}

results = {"llama70b": [], "mistral": [], "mixtral": [], "gpt35": []}
labels = {"llama70b": [], "mistral": [], "mixtral": [], "gpt35": []}

i = 0
for dataset_name in datasets:
    if "NR" in dataset_name and not args.nr:
            continue
    elif "NR" not in dataset_name and args.nr:
            continue
    if "All" in dataset_name: #or "MArg" in dataset_name:
        continue
    print(dataset_name, "&")
    i += 1
    for model in models:
        t = f"{'old_' if args.apply_template else ''}{'nr_' if args.nr else ''}results/{model}/*{dataset_name}*deniz*{args.n_shot}*{args.instruction}.json"
        if filename := glob(t):
            file = open(filename[0])
            d = json.load(file)
            res = d["eval_results"]
            results[model] += d["results"]
            labels[model] += d["labels"]
            models[model][0] += int(round(res["f1_1"]*100))
            models[model][1] += int(round(res["f1_0"]*100))
            res["f1_b"] = f1_score(d["results"], d["labels"], average="micro")
            if args.nr:
                models[model][2] += int(round(res["f1_2"]*100))
                models[model][3] += int(round(res["f1_b"]*100))
                print(int(round(res["f1_1"]*100)), int(round(res["f1_0"]*100)), int(round(res["f1_2"]*100)), int(round(res["f1_b"]*100)), sep=" / ", end="")
            else:
                models[model][2] += int(round(res["accuracy"]*100))
                print(int(round(res["f1_1"]*100)), int(round(res["f1_0"]*100)), int(round(res["f1_b"]*100)), sep=" / ", end="")
            print(" &")
        else:
            print(t)
            print("- \\hspace{0.15em} / \\hspace{0.15em} - \\hspace{0.15em} / \\hspace{0.15em} -&")

    print(" \\\\")

#print("Avg. &")
#for model in models:
#    #if model == "gpt35":
#    #    i -= 1
#    models[model][0] = int(models[model][0] / i)
#    models[model][1] = int(models[model][1] / i)
#    models[model][2] = int(models[model][2] / i)
#    if args.nr:
#        models[model][3] = int(models[model][3] / i)
#
#    print(*models[model], sep=" / ", end=" &\n")
#print(" \\\\")

print("Mic Avg. &")
for model in models:
    f1 = f1_score(results[model], labels[model], average=None)
    f1 = [f1[1], f1[0], f1[2]] if args.nr else [f1[1], f1[0]] 
    macro_f1 = [*f1, f1_score(results[model], labels[model], average="micro")]
    macro_f1 = [int(round(u*100)) for u in macro_f1]
    print(*macro_f1, sep=" / ", end= " &\n")
print(" \\\\")

