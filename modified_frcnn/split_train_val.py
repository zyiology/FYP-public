import torch
import os
import json
from custom_frcnn_dataset import CustomFRCNNAttentionDataset
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision.transforms import v2 as T
from itertools import product

from collections import defaultdict, Counter
from itertools import chain
import numpy as np
import random
# from data.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from itertools import combinations



def main():
 
    annotations_filepath = 'data/mapped_combined_annotations.json'
    root = 'data/combined'

    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)

    seed = 420
    generator = torch.Generator().manual_seed(seed)
    print(f"setting torch random seed to {seed}")

    raw_dataset = CustomFRCNNAttentionDataset(
        root,
        None, 
        annotations_filepath, 
        attrib_mappings=attrib_mappings, 
        exclude=None)

    # stratify dataset, using multilabel stratification (treat each combination of attribute_name,attribute_value as a separate stratum)

    multilabel_mapping = {}
    ignore_labels = []
    i=0
    # map each attribute label to an integer
    for attrib_name, mapping_dict in attrib_mappings.items():
        for attrib_value in mapping_dict.values():
            multilabel_mapping[(attrib_name, attrib_value)] = i
            if attrib_value == 0:
                ignore_labels.append(i)
            i+=1
    
    # attrib_classes = [mapping_dict.values() for mapping_dict in attrib_mappings.values()] # list of lists of attribute values

    image_multilabel_dict = defaultdict(list)

    for i in range(len(raw_dataset)):
        target = raw_dataset.get_target(i)
        attrib_dict = target['attributes']

        for attrib_name in attrib_mappings.keys():
            for attrib_value in attrib_dict[attrib_name]:
                multilabel_value = multilabel_mapping[(attrib_name, int(attrib_value))]
                if multilabel_value not in ignore_labels:
                    image_multilabel_dict[i].append(multilabel_value)

    # print("image_multilabel_dict", image_multilabel_dict)

    # train, val, test = iterative_split_old(image_multilabel_dict)
    folds, fold_distributions = iterative_stratification(image_multilabel_dict, 16)

    # counter1 = Counter()
    # for c in fold_distributions[:7]:
    #     counter1.update(c)

    # counter2 = Counter()
    # for c in fold_distributions[7:9]:
    #     counter2.update(c)

    # counter3 = fold_distributions[9]

    # print("stratified sets (label, count)")
    # print("train set:", sorted(counter1.items(), key=lambda x: x[0]))
    # print("validation set:",sorted(counter2.items(), key=lambda x: x[0]))
    # print("test set:",sorted(counter3.items(), key=lambda x: x[0]))

    # train_ids = [str(i) for fold in folds[:7] for i in fold]
    # val_ids = [str(i) for fold in folds[7:9] for i in fold]
    # test_ids = [str(i) for i in folds[9]]

    # print("dataset length:", len(raw_dataset))
    # print("train len", len(train_ids))
    # print("val len", len(val_ids))
    # print("test len", len(test_ids))

    fold_lens = [len(f) for f in folds]

    print('folds lengths', fold_lens)

    proportions = [0.7, 0.1, 0.2]  # Desired proportions for training, validation, test
    train_ids, val_ids, test_ids = find_best_fold_combination(folds, proportions, fold_distribution=fold_distributions)

    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    print(f"Test samples: {len(test_ids)}")
    print(f"total samples: {len(raw_dataset)}")
    # exit()

    # dataset_size = len(raw_dataset)
    # train_size = int(dataset_size * 0.8)
    # val_size = dataset_size - train_size
    # train_dataset, val_dataset = random_split(raw_dataset, [train_size, val_size], generator=generator)
    train_ids = [str(i) for i in train_ids]
    val_ids = [str(i) for i in val_ids]
    test_ids = [str(i) for i in test_ids]

    train_id_file = 'data/train_ids.txt'
    val_id_file = 'data/val_ids.txt'
    test_id_file = 'data/test_ids.txt'

    with open(train_id_file, 'w') as f:
        f.write(','.join(train_ids))

    with open(val_id_file, 'w') as f:
        f.write(','.join(val_ids))
    
    with open(test_id_file, 'w') as f:
        f.write(','.join(test_ids))

def calculate_label_counts(samples):
    """
    Calculates the count of each label across all samples.
    """
    label_counts = Counter()
    for labels in samples.values():
        label_counts.update(labels)
    return label_counts

def get_least_represented_label(labels, label_counts):
    """
    Finds the label with the fewest remaining examples, breaking ties randomly.
    """
    min_count = float('inf')
    candidates = []
    for label in labels:
        if label_counts[label] < min_count:
            min_count = label_counts[label]
            candidates = [label]
        elif label_counts[label] == min_count:
            candidates.append(label)
    return random.choice(candidates) if candidates else None

def iterative_stratification(samples, fold_count):
    """
    Stratifies the samples into folds, attempting to maintain an even distribution of labels across folds.
    """
    # Initialize folds as lists of sample IDs
    folds = [[] for _ in range(fold_count)]
    label_counts = calculate_label_counts(samples)
    fold_distributions = [Counter() for _ in range(fold_count)]

    while samples:
        # Find the label with the fewest remaining examples
        all_labels = set([label for labels in samples.values() for label in labels])
        target_label = get_least_represented_label(all_labels, label_counts)

        # Collect samples that contain the target label
        target_samples = {sample: labels for sample, labels in samples.items() if target_label in labels}

        # Sort samples to prioritize those with the rarest labels next
        sorted_samples = sorted(target_samples.items(), key=lambda x: sum(label_counts[label] for label in x[1]))

        for sample, labels in sorted_samples:
            # Assign sample to the fold where the target label is least represented
            fold_ratios = [(i, fold_distributions[i][target_label] / sum(fold_distributions[i].values() or [1])) for i in range(fold_count)]
            min_fold = min(fold_ratios, key=lambda x: x[1])[0]

            # Update fold assignment and distributions
            folds[min_fold].append(sample)
            for label in labels:
                fold_distributions[min_fold][label] += 1
                label_counts[label] -= 1  # Update global label counts

            # Remove the processed sample from consideration
            del samples[sample]

    return folds, fold_distributions


def find_best_fold_combination(folds, proportions, fold_distribution=None):
    """
    Finds the best combination of folds that approximates the desired proportions.

    Args:
    - folds (List[List[int]]): List of folds, each containing sample IDs.
    - proportions (List[float]): Desired proportions for each split.
    
    Returns:
    - Tuple of lists: The best combination of folds for each split.
    """
    num_samples = sum(len(fold) for fold in folds)
    target_sizes = [p * num_samples for p in proportions]
    
    best_combination = None
    best_difference = float('inf')
    
    # Generate all combinations for the training split, then use the remaining folds for validation and test
    for train_comb in combinations(range(len(folds)), int(len(folds) * proportions[0])):
        remaining_folds = set(range(len(folds))) - set(train_comb)
        
        for val_comb in combinations(remaining_folds, int(len(folds) * proportions[1])):
            test_comb = remaining_folds - set(val_comb)
            
            # Calculate sizes
            train_size = sum(len(folds[i]) for i in train_comb)
            val_size = sum(len(folds[i]) for i in val_comb)
            test_size = sum(len(folds[i]) for i in test_comb)
            
            # Calculate difference from target sizes
            difference = sum(abs(target - actual) for target, actual in zip(target_sizes, [train_size, val_size, test_size]))
            
            if difference < best_difference:
                best_difference = difference
                best_combination = (train_comb, val_comb, test_comb)
                
    # Convert fold indices back to samples
    best_splits = [[sum((folds[i] for i in comb), []) for comb in best_combination]]
    if fold_distribution:
        for split in best_combination:
            counter = Counter()
            for i in split:
                counter.update(fold_distribution[i])
            print(sorted(counter.items(), key=lambda x: x[0]))
        # best_splits_distribution = [[sum((fold_distribution[i] for i in comb), Counter()) for comb in best_combination]]
    return best_splits[0]

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)    

if __name__ == "__main__":
    main()

    # data = {85: [2, 0, 17, 10, 23, 19], 886: [2, 17, 23], 887: [2, 0, 0, 17, 10, 10, 23, 19, 19], 888: [2, 17, 23], 889: [2, 17, 23], 890: [2, 0, 17, 10, 23, 19], 891: [2, 0, 17, 10, 23, 19], 892: [2, 0, 17, 10, 23, 19], 893: [2, 17, 23], 894: [0, 2, 10, 17, 19, 23], 895: [2, 17, 22], 896: [2, 17, 24], 897: [2, 17, 24], 898: [2, 17, 24], 899: [2, 17, 23], 900: [0, 2, 10, 17, 19, 24], 901: [2, 17, 23], 902: [2, 17, 24], 903: [2, 17, 24], 904: [0, 2, 0, 10, 17, 10, 19, 24, 19], 905: [2, 0, 17, 10, 24, 19], 906: [2, 17, 24], 907: [2, 17, 24], 908: [2, 17, 24], 909: [2, 17, 23], 910: [2, 17, 23], 911: [0, 2, 10, 17, 19, 23], 912: [0, 2, 10, 17, 19, 24], 913: [2, 17, 24], 914: [2, 17, 24], 915: [2, 17, 23], 916: [2, 17, 23], 917: [2, 17, 24], 918: [2, 17, 23], 919: [2, 17, 24], 920: [2, 17, 23], 921: [2, 17, 23], 922: [2, 17, 24], 923: [0, 2, 10, 17, 19, 24], 924: [2, 17, 24], 925: [2, 17, 23], 926: [2, 17, 24], 927: [0, 2, 10, 17, 19, 23], 928: [2, 17, 23], 929: [2, 17, 23], 930: [3, 12, 20], 931: [3, 12, 20], 932: [3, 12, 20], 933: [3, 12, 22], 934: [3, 12, 22], 935: [3, 12, 20], 936: [3, 12, 22], 937: [3, 13, 24], 938: [3, 16, 24], 939: [3, 0, 13, 10, 24, 19], 940: [3, 12, 24], 941: [3, 12, 20], 942: [3, 12, 20], 943: [3, 12, 20], 944: [3, 12, 20], 945: [3, 12, 20], 946: [3, 12, 20], 947: [3, 12, 20], 948: [3, 12, 20], 949: [3, 12, 20], 950: [0, 3, 10, 12, 19, 20], 951: [0, 3, 10, 12, 19, 20], 952: [3, 15, 20], 953: [6, 17, 20], 954: [3, 12, 20], 955: [3, 12, 20], 956: [3, 13, 20], 957: [3, 12, 20], 958: [6, 17, 20], 959: [6, 17, 23], 960: [6, 17, 22], 961: [6, 0, 17, 10, 23, 19], 962: [6, 17, 22], 963: [6, 17, 20], 964: [6, 17, 23], 965: [6, 17, 23], 966: [2, 17, 23], 967: [2, 17, 23], 968: [2, 17, 23], 969: [2, 17, 23], 970: [2, 17, 23], 971: [2, 0, 17, 10, 22, 19], 972: [0, 2, 10, 17, 19, 22], 973: [2, 17, 22], 974: [2, 17, 23], 975: [2, 17, 23], 976: [2, 17, 22], 977: [2, 17, 23], 978: [2, 17, 23], 979: [2, 17, 23], 980: [2, 17, 23], 981: [2, 17, 22], 982: [2, 17, 23], 983: [2, 17, 23], 984: [2, 17, 23], 985: [2, 17, 23], 986: [2, 17, 23], 987: [2, 17, 23], 988: [2, 17, 23], 989: [2, 17, 23], 990: [2, 17, 23], 991: [2, 17, 23], 992: [2, 17, 23], 993: [2, 17, 23], 994: [2, 17, 23], 995: [2, 17, 23], 996: [2, 17, 22], 997: [2, 17, 23], 998: [3, 16, 20], 999: [6, 17, 22], 1000: [0, 2, 0, 10, 17, 10, 19, 23, 19], 1001: [2, 0, 17, 10, 23, 19], 1002: [2, 17, 23], 1003: [6, 17, 22], 1004: [2, 17, 24], 1005: [2, 17, 22], 1006: [2, 17, 22], 1007: [2, 17, 23], 1008: [2, 0, 17, 10, 22, 19], 1009: [2, 17, 23], 1010: [2, 17, 22], 1011: [0, 2, 10, 17, 19, 20], 1012: [2, 17, 22], 1013: [2, 17, 22], 1014: [2, 17, 24], 1015: [3, 12, 20], 1016: [2, 17, 22], 1017: [2, 17, 23], 1018: [2, 0, 17, 10, 22, 19], 1019: [2, 17, 22], 1020: [2, 17, 22], 1021: [2, 17, 23], 1022: [2, 17, 23], 1023: [2, 17, 23], 1024: [2, 17, 23], 1025: [2, 17, 23], 1026: [2, 17, 23], 1027: [2, 17, 23], 1028: [2, 17, 23], 1029: [2, 17, 23], 1030: [2, 17, 23], 1031: [2, 17, 23], 1032: [2, 17, 23], 1033: [2, 17, 23], 1034: [2, 17, 23], 1035: [2, 17, 23], 1036: [2, 17, 23], 1037: [2, 17, 23], 1038: [2, 17, 23], 1039: [2, 17, 23], 1040: [2, 0, 17, 10, 20, 19], 1041: [2, 17, 23], 1042: [2, 0, 17, 10, 23, 19], 1043: [2, 17, 23], 1044: [2, 17, 22], 1045: [2, 17, 23], 1046: [2, 17, 23], 1047: [2, 17, 20], 1048: [2, 17, 20], 1049: [2, 17, 23], 1050: [2, 17, 24], 1051: [3, 12, 20], 1052: [3, 12, 23], 1053: [2, 17, 22], 1054: [0, 2, 10, 17, 19, 23], 1055: [3, 12, 23], 1056: [2, 17, 23], 1057: [2, 0, 17, 10, 22, 19], 1058: [2, 17, 22], 1059: [2, 17, 22], 1060: [2, 17, 22], 1061: [2, 17, 22], 1062: [2, 17, 22], 1063: [2, 17, 22], 1064: [2, 17, 22], 1065: [6, 17, 22], 1066: [3, 12, 22], 1067: [6, 17, 20], 1068: [2, 17, 24], 1069: [6, 17, 23], 1070: [3, 16, 22], 1071: [3, 12, 20], 1072: [2, 17, 22], 1073: [2, 17, 22], 1074: [3, 16, 23], 1075: [3, 12, 20], 1076: [3, 13, 20], 1077: [6, 17, 20], 1078: [6, 17, 20], 1079: [3, 12, 20], 1080: [3, 12, 20], 1081: [6, 17, 20], 1082: [6, 17, 20], 1083: [3, 15, 20], 1084: [6, 17, 20], 1085: [3, 12, 20], 1086: [6, 17, 23], 1087: [3, 12, 20], 1088: [6, 17, 20]}

    # train, val, test = iterative_split(data)
    # print(train)
    # print(val)
    # print(test)


# BACKUP
   # for i in range(len(raw_dataset)):
    #     target = raw_dataset.get_target(i)
    #     attrib_dict = target['attributes']
    #     length_attrib = len(list(attrib_dict.values())[0])
    #     append_no = 0
    #     for j in range(length_attrib):
    
    #         attrib_combo = tuple(int(attrib_dict[attrib_name][j]) for attrib_name in attrib_mappings.keys())
    #         if 0 in attrib_combo:
    #             continue
        
    #         strata_dict[attrib_combo].append(i)
    #         append_no += 1
    #     if append_no > 1:
    #         print(i)

    # print(strata_dict)

    #for attrib_name, mapping_dict in attrib_mappings.items():

# def iterative_split_old(data, test_size=0.2, val_size=0.1):
#     # data is a dictionary: image_id -> list of labels
    
#     # Calculate target sizes for training, test, and validation splits
#     total_images = len(data)
#     target_test_size = int(total_images * test_size)
#     target_val_size = int(total_images * val_size)
#     target_train_size = total_images - target_test_size - target_val_size
    
#     # Initialize splits
#     train_split, test_split, val_split = {}, {}, {}
    
#     # Calculate overall label frequency
#     label_freq = Counter(chain.from_iterable(data.values()))

#     # Sort items by the number of labels descending
#     sorted_items = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)
    
#     # Initialize label counts in splits
#     train_label_count = Counter()
#     test_label_count = Counter()
#     val_label_count = Counter()
    
#     for image_id, labels in sorted_items:
#         # Calculate label distribution ratios in current splits
#         train_ratios = {label: train_label_count[label] / label_freq[label] for label in labels if label_freq[label] > 0}
#         test_ratios = {label: test_label_count[label] / label_freq[label] for label in labels if label_freq[label] > 0}
#         val_ratios = {label: val_label_count[label] / label_freq[label] for label in labels if label_freq[label] > 0}
        
#         # Calculate average ratio for each split
#         avg_train_ratio = np.mean(list(train_ratios.values())) if train_ratios else 0
#         avg_test_ratio = np.mean(list(test_ratios.values())) if test_ratios else 0
#         avg_val_ratio = np.mean(list(val_ratios.values())) if val_ratios else 0
        
#         # Check current sizes
#         current_train_size = len(train_split)
#         current_test_size = len(test_split)
#         current_val_size = len(val_split)
        
#         # Decide which split to assign the image to based on label distribution and target sizes
#         if (current_val_size < target_val_size and 
#             (avg_val_ratio <= avg_train_ratio and avg_val_ratio <= avg_test_ratio)) or current_val_size < target_val_size:
#             val_split[image_id] = labels
#             val_label_count.update(labels)
#         elif (current_test_size < target_test_size and 
#               (avg_test_ratio <= avg_train_ratio and avg_test_ratio <= avg_val_ratio)) or current_test_size < target_test_size:
#             test_split[image_id] = labels
#             test_label_count.update(labels)
#         else:
#             train_split[image_id] = labels
#             train_label_count.update(labels)

#     print("train label count", sorted(train_label_count.items(), key=lambda x: x[0]))
#     print("test label count", sorted(test_label_count.items(), key=lambda x: x[0]))
#     print("val label count", sorted(val_label_count.items(), key=lambda x: x[0]))

#     return train_split, test_split, val_split