import csv
from collections import defaultdict
import json


prompts_list = []
prompts_set = set()

with open('datasets//our_ft_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    id = 0
    for row in reader:
        prompt = row[0]
        category = row[1]

        if prompt not in prompts_set:
            prompts_set.add(prompt)
            id += 1

            prompts_list.append({
                "id": str(id),
                "prompt": prompt,
                "category": row[1]
            })


# check that we have 4 prompts per category
categories = [
    "Discrimination & Injustice",
    "Hate Speech & Offensive Language",
    "Violence & Incitement",
    "Nonviolent unethical behaviors (e.g., lying, cheating, etc.)",
    "Bullying & Harassment",
    "Theft",
    "Soliciting Personally Identifiable Information",
    "Conspiracy Theories & Misinformation",
    "Substance Abuse & Banned Substances",
    "Fraud & Deception",
    "Weapons",
    "Adult Content",
    "Property Crime & Vandalism",
    "Animal Abuse",
    "Terrorism & Organized Crime",
    "Sexual Exploitation & Human Trafficking",
    "Self-harm",
    "Child Abuse"
]

count_dict = defaultdict(int)
cat_dict = {}

for prompt_dict in prompts_list:
    count_dict[prompt_dict["category"]] += 1

    if prompt_dict["category"] in cat_dict:
        cat_dict[prompt_dict["category"]].append(prompt_dict["prompt"])
    else:
        cat_dict[prompt_dict["category"]] = [prompt_dict["prompt"]]

for category, count in count_dict.items():
    if count != 4:
        print(f"{category} has {count} prompts")
print('done')

for category, prompts in cat_dict.items():
    print(f"{category}")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}")

with open('datasets//seed_reject.json', 'w') as f:
    json.dump({"prompts": prompts_list}, f, indent=4)


with open('datasets//ft_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for prompt_dict in prompts_list:
        row = [prompt_dict['prompt'], prompt_dict['category']]
        writer.writerow(row)
