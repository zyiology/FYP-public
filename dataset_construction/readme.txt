# DESCRIPTION OF FILES

1. id_mapping_ui.py provides a UI to map building attribute data from "Building data sampling - Data.csv" to
COCO annotation data (e.g. mapillary_annotations.json)
-if files are stored in appropriate directories, the UI will automatically load the data
-can select the bounding box for the building that should have the attribute data, to
 produce a map of the attribute data (i.e. .csv row to COCO bounding box id)

-once the map is produced, the data can be joined using process_anno.ipynb (i.e. pull the attribute data from the .csv to the COCO .json)
-process_anno.ipynb also re-adjusts the image_ids in the COCO annotation to match the file names
    -e.g. image_12.jpg will be given image_id=12

2. count_anno.py is used to count the overall numbers of each attribute label in the dataset

3. mapillary.py is used to download mapillary street-view images from a mapillary URL
-queries the API using the image key to get a download link for the image, requires a mapillary account

4. pano_extract_ui.py provides a UI to select which fov to use for a given panorama image
-iterates through all images in a directory, and produces a rectilinear image for each based on the selected fov
-nfov.py is taken from https://github.com/NitishMutha/equirectangular-toolbox/tree/master
    -processes panorama images to extract 'normal' rectilinear images

5. download_gsv.ahk is an autohotkey script to automatically download GSV images from the Building Data google sheets
-it "manually" copies and pastes the images from the google sheets into an open excel file, then saves the image
-quite tedious, but google sheets does not have image download functionality as far as I'm aware, and this was the
most convenient way to automate the downloading

6. annotations for mapillary dataset and GSV dataset are stored in respective .json files
-mapillary_annotations.json and gmaps_part1_annotations.json contain just the bounding boxes for buildings, with no attribute data
-id_mapping_ui and process_annoy.ipynb was used to map the attribute data to the bounding boxes
-mapped_mapillary_annotations.json and mapped_gmaps_part1_annotations.json are the result

the annotations were created using CVAT
-can be recreated by uploading images to CVAT and uploading corresponding annotation file
e.g. upload the mapillary images (use natural sorting to ensure correct order) and mapillary_annotations.json to CVAT - will recreate that portion of the dataset

7. combine_annos.py was used to merge the annotation data from mapillary and GSV
-result is mapped_combined_annotations.json