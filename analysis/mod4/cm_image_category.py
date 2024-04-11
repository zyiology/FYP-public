import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

target_attrib = "category"
cm_file = "mod4_category_cm.txt"

with open("../attrib_mappings.json", "r") as f:
    attrib_mappings = json.load(f)

labels = list(attrib_mappings[target_attrib].keys())
labels = labels[:-3]

num_labels = [str(i) for i in range(1, len(labels) + 1)]
n = len(labels)
cm = np.zeros((n, n))

with open(cm_file, 'r') as f:
    for i, line in enumerate(f):
        data = line.split()
        # print(data)
        cm[i] = data

correct = 0
for i in range(n):
    correct += cm[i, i]

print(correct/cm.sum())
exit()

plt.figure(figsize=(10, 7))
sns.set(font_scale=1) # for label size
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', cmap='Blues', xticklabels=num_labels, yticklabels=num_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Building Category')
plt.yticks(rotation=0)

plt.figtext(0.5, -0.12, "\n".join([f"{num}: {label}" for num, label in zip(num_labels, labels)]), ha="center", fontsize=10)

plt.tight_layout()

filepath = 'category_cm.png'
plt.savefig(filepath, bbox_inches='tight', pad_inches=0.25)