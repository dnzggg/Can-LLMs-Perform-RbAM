from glob import glob
import argparse
import os

from sklearn.metrics import f1_score, classification_report, accuracy_score
from datasets import load_from_disk


parser = argparse.ArgumentParser(description="Analyse")
parser.add_argument(
    "--nr", action=argparse.BooleanOptionalAction, default=False
)

args = parser.parse_args()

datasets = list(map(os.path.basename, glob("/vol/bitbucket/dg1822/CanLLMsPerformRbAM/Datasets/*")))
datasets.remove("AllDataset")
datasets.remove("NR_AllDataset")

models = {dataset.replace("Dataset", ""): [0] * (4 if args.nr else 3) for dataset in datasets if (args.nr and "NR" in dataset) or (not args.nr and "NR" not in dataset)}

results = {dataset.replace("Dataset", ""): [] for dataset in datasets if (args.nr and "NR" in dataset) or (not args.nr and "NR" not in dataset)}
labels = {dataset.replace("Dataset", ""): [] for dataset in datasets if (args.nr and "NR" in dataset) or (not args.nr and "NR" not in dataset)}

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
        t = f"Results/Result-Roberta-{model}-{dataset_name}"
        if filename := glob(t):
            d = load_from_disk(filename[0])
            results[model] += d["result"]
            labels[model] += d["support"]
            res = {}
            for y, x in enumerate(f1_score(d["result"], d["support"], average=None)):
                models[model][y] += int(round(x*100))
                res["f1_" + str(y)] = x
            res["f1_b"] = f1_score(d["result"], d["support"], average="micro") 
            if args.nr:
                models[model][3] += int(round(res["f1_b"]*100))
                print(int(round(res["f1_1"]*100)), int(round(res["f1_0"]*100)), int(round(res["f1_2"]*100)), int(round(res["f1_b"]*100)), sep=" / ", end="")
            else:
                models[model][2] += int(round(res["f1_b"]*100))
                print(int(round(res["f1_1"]*100)), int(round(res["f1_0"]*100)), int(round(res["f1_b"]*100)), sep=" / ", end="")
            print(" &")
        else:
            print("- \\hspace{0.15em} / \\hspace{0.15em} - \\hspace{0.15em} / \\hspace{0.15em} -&")

    print(" \\\\")

#i -= 1
#print("Avg. &")
#for model in models:
#    models[model][0] = int(models[model][1] / i)
#    models[model][1] = int(models[model][0] / i)
#    models[model][2] = int(models[model][2] / i)
#    if args.nr:
#        models[model][3] = int(models[model][3] / i)
#
#    print(*models[model], sep=" / ", end=" &\n")
#print(" \\\\")

print("Micro Avg. &")
for model in models:
    f1 = f1_score(results[model], labels[model], average=None)
    if args.nr:
        f1 = f1[1], f1[0], f1[2]
    else:
        f1 = f1[1], f1[0]
    micro_f1 = [*f1, f1_score(results[model], labels[model], average="micro")]
    micro_f1 = [int(round(u*100)) for u in micro_f1]
    # print(classification_report(results[model], labels[model]))
    print(*micro_f1, sep=" / ", end= " &\n")
print(" \\\\")

