import json

annotations1 = 'mapped_mapillary_1_annotations.json'
with open(annotations1, 'r') as f:
    anno1 = json.load(f)

annotations2 = 'mapped_gmaps_1_annotations.json'
with open(annotations2, 'r') as f:
    anno2 = json.load(f)

assert anno1['categories'] == anno2['categories'], "categories of merged annotations files don't match"

anno1['images'].extend(anno2['images'])

# find largest anno id in anno1
ids = [anno['id'] for anno in anno1['annotations']]
max_id = max(ids)

# increment ids in anno2 by max_id
for anno in anno2['annotations']:
    anno['id'] += max_id

anno1['annotations'].extend(anno2['annotations'])

output_file = 'mapped_combined_annotations.json'
with open(output_file, 'w') as f:
    json.dump(anno1, f)

