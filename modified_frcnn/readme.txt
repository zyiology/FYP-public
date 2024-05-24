# INFO

reading the Faster R-CNN paper beforehand would be good to better understand the modifications that are made
conda environment can be recreated from frcnn_environment.yml
images should be stored in 'data/combined/images/'
annotations should be stored in 'data/'

# SCRIPTS THAT ARE MEANT TO BE RUN

test_images.py loads a trained model and evaluates it on the test set to print evaluation metrics
eval_images.py does the same, but plots the predictions on the images instead of printing evaluation metrics

train_custom_fasterrcnn_attention.py is used to train the model
modifications can be toggled on/off

baseline_custom_train.py is used to train the baseline model
baseline_custom.py is used to test the trained baseline model on the test set

split_train_val.py implements the iterative stratification algorithm to stratify the dataset by attribute labels
outputs train_ids.txt, val_ids.txt, and test_ids.txt to data/

ALL SCRIPTS WERE RUN ON A SLURM CLUSTER, using sbatch to submit scripts
SCRIPTS SHOULD BE CALLED with the job ID as the first argument, e.g. python test_images.py 12345
so sys.argv was used in most of the .py files to keep track of the job ID in the cluster
if not running on a slurm cluster, have to edit those files (search for sys.argv)

# FILES CONTAINING CLASSES/FUNCTIONS/DATA TO BE USED ABOVE

custom_frcnn_dataset.py implements a custom pytorch dataset to load the images

custom_faster_rcnn_attention.py contains the implementation of the modified model

attrib_eval.py contains the code to evaluate attribute prediction performance and overall performance

attrib_mappings.json contains the attributes that will be processed
(need to map the attribute strings to numbers for ease of processing)
(refer to attrib_mappings_backup.json for the full list)

# EXTRA NOTES

regarding postprocess_detections_with_attributes() in custom_faster_rcnn_attention.py
the original Faster R-CNN model generates one box per class per proposal when used in inference mode
but the modified model only generates one attribute prediction per proposal
so to ensure each box has a corresponding attribute, have to repeat by number of classes

# why is there one box per class per proposal? so that the model can learn how to refine the bounding box
# to improve refinement of bounding box, could perhaps generate one box per class per attribute per proposal
# but I judged that that would be counter productive

(in training mode, cross entropy is done using a given proposal vs a ground truth box, so this issue isn't encountered)
