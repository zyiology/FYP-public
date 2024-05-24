### SCRIPT TO SPLIT DATASET INTO TRAIN, VALIDATION, AND TEST SETS ###
# uses the iterative stratification algorithm to ensure that each split has a similar distribution of labels


import torch
import json
from custom_frcnn_dataset import CustomFRCNNAttentionDataset
from torchvision.transforms import v2 as T
from collections import defaultdict, Counter
import random
from itertools import combinations 

def main():
    '''
    Function to split the dataset into train, validation, and test sets.
    Adjust n_folds and proportions to change the number of folds and the desired proportions of the splits.
    Each fold is a subset of the dataset with a similar distribution of labels. Folds generally have different sizes,
    and combinations of folds are iteratively tested to find the best combination that approximates the desired proportions.
    '''
 
    annotations_filepath = 'data/mapped_combined_annotations.json'
    root = 'data/combined'

    with open("attrib_mappings.json", "r") as f:
        attrib_mappings = json.load(f)

    seed = 420
    generator = torch.Generator().manual_seed(seed)
    print(f"setting torch random seed to {seed}")

    # number of folds to split the dataset into - more folds generally gives results closer
    # to the desired proportions, but takes longer to run 
    n_folds = 16

    # desired proportions for training, validation, and test sets
    proportions = [0.7, 0.1, 0.2]

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

    # map each (attribute name, attribute class) combo to an integer,
    # and keep track of the "Unknown" labels to be ignored when splitting
    for attrib_name, mapping_dict in attrib_mappings.items():
        for attrib_value in mapping_dict.values():
            multilabel_mapping[(attrib_name, attrib_value)] = i
            if attrib_value == 0:
                ignore_labels.append(i)
            i+=1

    image_multilabel_dict = defaultdict(list)

    # count the number of each combination in the dataset
    for i in range(len(raw_dataset)):
        target = raw_dataset.get_target(i)
        attrib_dict = target['attributes']

        for attrib_name in attrib_mappings.keys():
            for attrib_value in attrib_dict[attrib_name]:
                multilabel_value = multilabel_mapping[(attrib_name, int(attrib_value))]
                if multilabel_value not in ignore_labels:
                    image_multilabel_dict[i].append(multilabel_value)

    # use iterative stratification to split the dataset
    # produces 16 "folds" of the dataset, each with a similar distribution of labels
    # then finds the best combination of folds to approximate the desired proportions
    folds, fold_distributions = iterative_stratification(image_multilabel_dict, n_folds)

    fold_lens = [len(f) for f in folds]
    print('folds lengths', fold_lens)

    train_ids, val_ids, test_ids = find_best_fold_combination(folds, proportions, fold_distribution=fold_distributions)

    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    print(f"Test samples: {len(test_ids)}")
    print(f"total samples: {len(raw_dataset)}")
    
    # save the split IDs to files
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
