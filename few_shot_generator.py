import random

from datasets import load_from_disk
import csv

dataset = load_from_disk('Datasets/DebatepediaProconDataset')

a = 1
s = 2
file = open("few_shot/1A2S/seed_2.csv", "w")
writer = csv.writer(file, delimiter='#', quotechar='|', quoting=csv.QUOTE_MINIMAL)
while a > 0 or s > 0:
    data = dataset.select([random.randint(0, len(dataset))])

    if data[0]["support"] == 0 and a > 0:
        a -= 1
        writer.writerow([data[0]["arg1"], data[0]["arg2"], data[0]["support"]])
    if data[0]["support"] == 1 and s > 0:
        s -= 1
        writer.writerow([data[0]["arg1"], data[0]["arg2"], data[0]["support"]])

    
