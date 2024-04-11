import tkinter as tk
from PIL import Image, ImageTk
import json
from os.path import join
import csv
import re


def draw_image_and_boxes(canvas, image_data, image_folder):
    # Clear previous items
    canvas.delete("all")

    file_path = join(image_folder, image_data['file_name'])

    # Load and display the image
    img = Image.open(file_path)
    orig_width, orig_height = img.size

    # Calculate scaling factor to fit the image within the canvas size
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    if canvas_width == canvas_height == 1:
        canvas_width, canvas_height = (1600,800)

    scale_width = canvas_width / orig_width
    scale_height = canvas_height / orig_height
    scale_factor = min(scale_width, scale_height)

    # Resize image and bounding boxes
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)
    canvas.image = img_tk  # Keep a reference!
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

    img_filename = image_data['file_name']
    pattern = r'image_(\d+)\.([a-zA-Z]+)'
    img_id = int(re.search(pattern, img_filename).group(1))

    update_attrib_info(img_id)  # image_data['annotations'][0]['image_id'])

    # Draw resized bounding boxes
    for anno in image_data['annotations']:
        bbox = anno['bbox']  # COCO bbox format: [x, y, width, height]
        scaled_bbox = [coord * scale_factor for coord in bbox]  # Scale bbox
        rectID = canvas.create_rectangle(scaled_bbox[0], scaled_bbox[1], scaled_bbox[0] + scaled_bbox[2],
                                scaled_bbox[1] + scaled_bbox[3], outline='red', tags=("bbox",), fill="gray",
                                         stipple="@transparent.xbm")
        canvas.create_text(scaled_bbox[0]+10,scaled_bbox[1]+10, text=f"{anno['id']}",fill='white', font=('Helvetica 15 bold'))
        bbox_id = anno['id']
        # Note: Update the lambda function as needed for your logging
        # Bind clicking on the bounding box
        # Note: This simplistic approach may not work perfectly for overlaps or precise clicking
        canvas.tag_bind(rectID, "<Button-1>",
                        lambda event, bbox_id_=bbox_id, img_id_=img_id: handle_click(bbox_id_, img_id_, canvas))


def handle_click(bbox_id, img_id, canvas):
    log_bbox(bbox_id, img_id)
    next_image(canvas)
    return

def log_bbox(bbox_id, image_id):
    global number_of_actions
    number_of_actions += 1

    with open('log.txt', 'a') as log_file:
        log_file.write(f'{image_id}, {bbox_id}\n')
    feedback_message = f'{number_of_actions}. Logged Image ID: {image_id}, BBox ID: {bbox_id}'  # Prepare the feedback message
    feedback_label.config(text=feedback_message)  # Update the label text
    print(feedback_message)  # For confirmation


def next_image(canvas):
    global current_image_index
    current_image_index = (current_image_index + 1) % len(images_data)

    # if only one bounding box, don't draw the image
    while len(images_data[current_image_index]['annotations']) <= 1:
        if len(images_data[current_image_index]['annotations']) == 1:
            bbox_id = images_data[current_image_index]['annotations'][0]['id']

            img_filename = images_data[current_image_index]['file_name']
            pattern = r'image_(\d+)\.([a-zA-Z]+)'
            img_id = int(re.search(pattern, img_filename).group(1))

            log_bbox(bbox_id, img_id)
        current_image_index = (current_image_index + 1) % len(images_data)

    draw_image_and_boxes(canvas, images_data[current_image_index], img_folder)


def update_attrib_info(id):
    row = csv_rows[id]
    text = (f"current id:{id},\n"
            f"building cat:{row[8]},\n"
            f"occupancy grp:{row[9]},\n"
            f"occupancy type:{row[10]},\n"
            f"no floors:{row[11]},\n"
            f"basement:{row[12]},\n"
            f" material:{row[13]},\n"
            f"roof shape:{row[14]},\n"
            f"roof covers:{row[15]},\n"
            f"shutters:{row[16]},\n"
            f"window:{row[17]}")
    attrib_info_label.config(text=text)
    return


if __name__ == "__main__":
    annotation_file = 'mapillary_annotations.json'
    img_folder = r"C:\Users\zhiyi\PyCharmProjects\FYP\curling\images\new_dataset"  # r"C:\Users\zyiol\OneDrive - Nanyang Technological University\school stuffs\mapillary"
    csv_filename = 'Building data sampling - Data.csv'

    exclude = [f'image_{i}.jpg' for i in range(0)]


    # Load COCO annotations
    with open(annotation_file) as f:
        data = json.load(f)

    # Initialize a list to hold processed data for easy access
    images_data = []
    for img in data['images']:
        if img['file_name'] in exclude:
            continue

        img_dict = {
            'id': img['id'],
            'file_name': img['file_name'],
            'annotations': []
        }
        for ann in data['annotations']:
            if ann['image_id'] == img['id']:
                img_dict['annotations'].append(ann)
        images_data.append(img_dict)

    for d in images_data[:10]:
        print(d)

    current_image_index = 0

    # Set up GUI
    root = tk.Tk()

    # Create a main frame to use the pack method effectively
    #main_frame = tk.Frame(root)
    root.geometry('1800x1000')
    #main_frame.pack(fill=tk.BOTH, expand=True)

    root.grid_columnconfigure(0, weight=4)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=4)  # Row for canvas
    root.grid_rowconfigure(1, weight=1)  # Row for next_img_btn
    root.grid_rowconfigure(2, weight=1)  # Row for feedback_label

    # # Create a frame for the left side (for canvas and buttons)
    # left_frame = tk.Frame(main_frame)
    # left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    #
    # # Create a frame for the right side (for attribute info)
    # right_frame = tk.Frame(main_frame)
    # right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    canvas1 = tk.Canvas(root, width=1200, height=700)
    canvas1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    # canvas1.pack()

    attrib_info_label = tk.Label(root, text="attribute data", anchor='w', justify='left')
    attrib_info_label.grid(row=0, column=1, rowspan=3, sticky="nw", padx=5, pady=5)
    # attrib_info_label.pack(side=tk.LEFT)
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_rows = list(csv_reader)
        #remove header
        csv_rows.pop(0)

    next_img_btn = tk.Button(root, text="Next Image", command=lambda: next_image(canvas1))
    next_img_btn.grid(row=1, column=0, sticky="ew", padx=5)
    # next_img_btn.pack()

    # At the end of your GUI setup, after creating the canvas and buttons
    feedback_label = tk.Label(root, text="Click on a bounding box to see info here")
    feedback_label.grid(row=2,column=0,sticky='ew',padx=5)
    # feedback_label.pack()


    #img_folder = r"C:\Users\zyiol\PycharmProjects\FYP\curling\images"
    number_of_actions = 0

    draw_image_and_boxes(canvas1, images_data[current_image_index], img_folder)

    root.mainloop()

# modifications:
# if image has only 1 anno, auto log it and skip
# after log, auto next