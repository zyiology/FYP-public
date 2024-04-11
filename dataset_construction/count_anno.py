import pandas as pd
import json
from collections import defaultdict


if __name__ == "__main__":

    annotations_json = 'mapped_combined_annotations.json'
    with open(annotations_json, 'r') as f:
        anno_coco = json.load(f)

    attrib_count_dict = {
        'category':defaultdict(int),
        'occupancy_group':defaultdict(int),
        'occupancy_type':defaultdict(int),
        'no_floors':defaultdict(int),
        'basement':defaultdict(int),
        'material':defaultdict(int),
        'roof_shape':defaultdict(int),
        'roof_covers':defaultdict(int),
        'shutters':defaultdict(int),
        'window_area':defaultdict(int)
    }

    for anno in anno_coco['annotations']:
        attribs = anno['attributes']
        for attrib_name in attrib_count_dict.keys():
            val = attribs[attrib_name]
            attrib_count_dict[attrib_name][val] += 1

    with open('counts.txt', 'w') as f:
        for attrib_name, count_dict in attrib_count_dict.items():
            f.write(f"Attribute: {attrib_name}\n")
            for category, count in count_dict.items():
                f.write(f"{category}: {count}\n")
            f.write('\n')

    print('count completed.')
