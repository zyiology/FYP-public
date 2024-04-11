images should be stored in 'data/combined/images/'
annotations should be stored in 'data/'

split_train_val.py implements the iterative stratification algorithm to stratify the dataset by attribute labels
outputs train_ids.txt, val_ids.txt, and test_ids.txt to data/

test_images.py loads a trained model and evaluates it on the test set to print evaluation metrics
eval_images.py does the same, but plots the predictions on the images instead of printing evaluation metrics

custom_frcnn_dataset.py implements a custom pytorch dataset to load the images

custom_faster_rcnn_attention.py contains the implementation of the modified model

train_custom_fasterrcnn_attention is used to train the model
modifications can be toggled on/off
attrib_mappings.json contains the attributes that will be processed
(need to map the attribute strings to numbers for ease of processing)
(refer to attrib_mappings_backup.json for the full list)

attrib_eval.py contains the code to evaluate attribute prediction performance and overall performance

baseline_custom_train.py is used to train the baseline model
baseline_custom.py is used to test the trained baseline model on the test set

ALL SCRIPTS WERE RUN ON A SLURM CLUSTER, using sbatch to submit scripts
so sys.argv was used in most of the .py files to keep track of the job ID in the cluster
if not running on a slurm cluster, have to edit those files