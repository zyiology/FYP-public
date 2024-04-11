id_mapping_ui.py provides a UI to map building attribute data from "Building data sampling - Data.csv" to
COCO annotation data (e.g. mapillary_annotations.json)

once the map is produced, the data can be joined using process_anno.ipynb
process_anno.ipynb also re-adjusts the image_ids in the COCO annotation to match the file names
e.g. image_12.jpg will be given image_id=12

count_anno.py is used to count the overall numbers of each attribute label in the dataset

mapillary.py is used to download mapillary street-view images from a mapillary URL
queries the API using the image key to get a download link for the image, requires a mapillary account

nfov.py is taken from https://github.com/NitishMutha/equirectangular-toolbox/tree/master
processes panorama images to extract 'normal' rectilinear images
used by pano_extract_ui.py to provide a UI to select which fov to use for a given panorama image

download_gsv.ahk is an autohotkey script to automatically download GSV images from the Building Data google sheets
it "manually" copies and pastes the images from the google sheets into an open excel file, then saves the image
(quite tedious, but google sheets does not have image download functionality as far as I'm aware, and this was the
most convenient way to automate the downloading)