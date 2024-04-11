import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

labels = ["Unknown", "Agriculture", "Commercial", "Education", "Industry",
          "Government", "Religious", "Residential", "Others"]

target_attrib = "occupancy_group"
cm_file = "baseline_occupancy_cm.txt"


num_labels = [str(i) for i in range(1, len(labels) + 1)]
n = len(labels)
cm = np.zeros((n, n))

with open(cm_file, 'r') as f:
    for i, line in enumerate(f):
        data = line.split()
        # print(data)
        cm[i] = data

print(cm)

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
plt.title('Confusion Matrix for Occupancy Group')
plt.yticks(rotation=0)

legend = [f"{num}: {label}" for num, label in zip(num_labels, labels)]
legend_text = ""
for i in range(len(legend)):
    if i % 2 == 0:
        legend_text += "\n"
        legend_text += legend[i] + '    '
    else:
        legend_text += legend[i]

plt.figtext(0.5, -0.1, legend_text, ha="center", fontsize=10)

plt.tight_layout()

filepath = 'baseline_occupancy_cm.png'
plt.savefig(filepath, bbox_inches='tight', pad_inches=0.25)