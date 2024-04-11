import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from os.path import join, exists
from os import rename
import matplotlib.pyplot as plt
import matplotlib.image as im
from nfov import NFOV
import queue


def update_display(task, images_folder, output_folder, root, img_labels, buttons, task_queue):
    x, y = task['x'], task['y']
    image_name = f"image_{task['index']}.jpg"
    image_path = join(images_folder, image_name)

    # # Archive the old image
    # archive_name = f"image_{task['index']}.jpg"
    # archive_path = join(images_folder, "archive", archive_name)
    output_path = join(output_folder, image_name)

    # rename(image_path, archive_path)
    img = im.imread(image_path)
    center_point = np.array([x, y])  # camera center point (valid range [0,1])
    fov_options = [0.4, 0.6, 0.9, 1.2]  # predefined FOV values

    pre_rendered_images = {}
    max_image_size = (root.winfo_screenwidth() // 2 - 50, root.winfo_screenheight() // 2 - 50)

    for i, fov in enumerate(fov_options):
        nfov = NFOV(height=1000, width=2000, fov=(fov, fov))
        output_img = nfov.toNFOV(img, center_point)
        pre_rendered_images[fov] = output_img

        display_img = Image.fromarray(output_img)
        display_img.thumbnail(max_image_size, Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(display_img)

        img_labels[i].config(image=img_tk)
        img_labels[i].image = img_tk
        buttons[i].config(command=lambda f=fov: on_fov_selected(f, output_path, pre_rendered_images, task_queue, images_folder, output_folder, root, img_labels, buttons))

    print('display updated')


def on_fov_selected(fov, output_path, pre_rendered_images, task_queue, images_folder, output_folder, root, img_labels, buttons):
    selected_img = pre_rendered_images[fov]
    plt.imsave(output_path, selected_img)
    print('Completed:', output_path.split('/')[-1])

    # Proceed to the next task
    if not task_queue.empty():
        next_task = task_queue.get()
        update_display(next_task, images_folder, output_folder, root, img_labels, buttons, task_queue)
    else:
        root.destroy()


def pano_extract(images_folder, image_info_file, output_folder):
    image_info_df = pd.read_csv(image_info_file, sep='\t', header=0)
    task_queue = queue.Queue()

    for index, row in image_info_df.iterrows():
        if row['is_pano'] == 1:
            output_name = f"image_{row['index']}.jpg"
            output_path = join(images_folder, "archive", output_name)
            if not exists(output_path):
                task_queue.put(row)

    # Layout positions
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    img_labels = [Label(root) for _ in positions]
    buttons = [Button(root, text=f"FOV: {fov}") for fov in [0.4, 0.6, 0.9, 1.2]]

    for i, (label, button) in enumerate(zip(img_labels, buttons)):
        label.grid(row=positions[i][0], column=positions[i][1])
        button.grid(row=positions[i][0], column=positions[i][1], sticky="s")

    if not task_queue.empty():
        first_task = task_queue.get()
        update_display(first_task, images_folder, output_folder, root, img_labels, buttons, task_queue)

    root.mainloop()

if __name__ == '__main__':
    img_folder = 'images/panos/'
    output_folder = 'images/panos/output'
    image_info = 'image_links_v3.txt'

    # Setup the main window
    root = tk.Tk()
    root.title("Image Processing")
    pano_extract(img_folder, image_info, output_folder)
