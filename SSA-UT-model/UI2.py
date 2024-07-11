import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os
import threading
import queue
from PIL import Image, ImageTk, ImageDraw

def run_main():
    data_dir = data_dir_entry.get()
    out_dir = out_dir_entry.get()
    save_img = save_img_var.get()
    save_json = save_json_var.get()
    ckpt_path = ckpt_path_entry.get()

    command = [
        "python", "main.py",
        "--data_dir", data_dir,
        "--out_dir", out_dir,
        "--ckpt_path", ckpt_path
    ]

    if save_img:
        command.append('--save_img')
    if save_json:
        command.append('--save_json')
    if light_mode:
        command.append('--light_mode')

    run_subprocess(command)

def run_infer_crack():
    img_dir = img_dir_entry.get()
    model_path = model_path_entry.get()
    model_type = model_type_combobox.get()
    out_viz_dir = out_viz_dir_entry.get()
    out_pred_dir = out_pred_dir_entry.get()
    threshold = threshold_entry.get()

    command = [
        "python", "infer_crack.py",
        "-img_dir", img_dir,
        "-model_path", model_path,
        "-model_type", model_type,
        "-out_viz_dir", out_viz_dir,
        "-out_pred_dir", out_pred_dir,
        "-threshold", str(threshold)
    ]

    run_subprocess(command)

def run_subprocess(command):
    process_queue = queue.Queue()

    def enqueue_output(proc, queue):
        for line in iter(proc.stdout.readline, b''):
            queue.put(line)
        proc.stdout.close()

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
    thread = threading.Thread(target=enqueue_output, args=(proc, process_queue))
    thread.daemon = True
    thread.start()

    def update_log():
        while True:
            try:
                line = process_queue.get_nowait()
            except queue.Empty:
                break
            else:
                log_text.insert(tk.END, line)
                log_text.see(tk.END)
        if proc.poll() is None:
            app.after(100, update_log)
        else:
            messagebox.showinfo("Info", "Execution completed.")

    update_log()

def select_data_dir():
    dir_name = filedialog.askdirectory()
    if dir_name:
        data_dir_entry.delete(0, tk.END)
        data_dir_entry.insert(0, dir_name)

def select_out_dir():
    dir_name = filedialog.askdirectory()
    if dir_name:
        out_dir_entry.delete(0, tk.END)
        out_dir_entry.insert(0, dir_name)

def select_ckpt_path():
    file_name = filedialog.askopenfilename()
    if file_name:
        ckpt_path_entry.delete(0, tk.END)
        ckpt_path_entry.insert(0, file_name)

def select_img_dir():
    dir_name = filedialog.askdirectory()
    if dir_name:
        img_dir_entry.delete(0, tk.END)
        img_dir_entry.insert(0, dir_name)

def select_out_viz_dir():
    dir_name = filedialog.askdirectory()
    if dir_name:
        out_viz_dir_entry.delete(0, tk.END)
        out_viz_dir_entry.insert(0, dir_name)

def select_out_pred_dir():
    dir_name = filedialog.askdirectory()
    if dir_name:
        out_pred_dir_entry.delete(0, tk.END)
        out_pred_dir_entry.insert(0, dir_name)

def load_image(index):
    if len(image_files) == 0:
        canvas.create_text(200, 150, text="No processed images available, please wait.")
    else:
        image_path = image_files[index]
        image = Image.open(image_path)

        if mask_var.get():
            mask_path = os.path.join(image_dir, os.path.basename(image_path).replace('.jpg', '_patch.jpg'))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).resize(image.size, Image.LANCZOS)
                image = Image.blend(image, mask, alpha=0.5)

        if box_var.get():
            mask_path = os.path.join(image_dir, os.path.basename(image_path).replace('.jpg', '_patch.jpg'))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).resize(image.size, Image.LANCZOS)
                draw = ImageDraw.Draw(image)
                threshold = box_threshold_var.get() / 100
                for x in range(mask.size[0]):
                    for y in range(mask.size[1]):
                        if mask.getpixel((x, y)) > int(threshold * 255):
                            draw.rectangle([x - 5, y - 5, x + 5, y + 5], outline="red")

        update_image_on_canvas(image)
        filename_label.config(text=os.path.basename(image_path))
        description_label.config(text=description_text.get())

def update_image_on_canvas(image):
    global canvas_image
    canvas.delete("all")
    canvas_width=600
    canvas_height=400
    # image_ratio = image.width / image.height
    # canvas_width = canvas.winfo_width()
    # canvas_height = int(canvas_width / image_ratio)
    canvas.config(height=canvas_height,width=canvas_width)
    # while canvas_height > 400:
    #     canvas_height = int(canvas_height*0.8)
    #     canvas_width = int(canvas_width*0.8)
    canvas_image = ImageTk.PhotoImage(image.resize((canvas_width, canvas_height), Image.LANCZOS))
    canvas.create_image(0, 0, anchor=tk.NW, image=canvas_image)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

def show_next_image():
    global current_image_index
    if len(image_files) > 0:
        current_image_index = (current_image_index + 1) % len(image_files)
        load_image(current_image_index)

def show_previous_image():
    global current_image_index
    if len(image_files) > 0:
        current_image_index = (current_image_index - 1) % len(image_files)
        load_image(current_image_index)

def refresh_images():
    global image_files
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if
                   file.endswith(('png', 'jpg', 'jpeg'))]
    load_image(current_image_index)

def switch_category():
    global image_dir, current_image_index
    if category_var.get() == "with_crack":
        image_dir = 'output/test_pred/with_crack'
        description_text.set("Currently viewing images with cracks. Click << or >> to browse.")
    else:
        image_dir = 'output/test_pred/without_crack'
        description_text.set("Currently viewing images without cracks. Click << or >> to browse.")
    current_image_index = 0
    refresh_images()

def reset_zoom():
    load_image(current_image_index)

def zoom(event):
    scale = 1.0
    if event.delta > 0:
        scale *= 1.1
    elif event.delta < 0:
        scale /= 1.1

    canvas.scale("all", event.x, event.y, scale, scale)
    canvas.configure(scrollregion=canvas.bbox("all"))

def zoom_in():
    canvas.scale("all", canvas.winfo_width() // 2, canvas.winfo_height() // 2, 1.1, 1.1)
    canvas.configure(scrollregion=canvas.bbox("all"))

def zoom_out():
    canvas.scale("all", canvas.winfo_width() // 2, canvas.winfo_height() // 2, 0.9, 0.9)
    canvas.configure(scrollregion=canvas.bbox("all"))

def start_zoom(event):
    global zoom_rect
    zoom_rect = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red')

def update_zoom_rect(event):
    global zoom_rect
    canvas.coords(zoom_rect, canvas.bbox(zoom_rect)[0], canvas.bbox(zoom_rect)[1], event.x, event.y)

def apply_zoom(event):
    global zoom_rect
    x1, y1, x2, y2 = canvas.coords(zoom_rect)
    canvas.delete(zoom_rect)
    if x2 - x1 > 0 and y2 - y1 > 0:
        canvas.scale("all", (x1 + x2) / 2, (y1 + y2) / 2, canvas.winfo_width() / (x2 - x1),
                     canvas.winfo_height() / (y2 - y1))
        canvas.configure(scrollregion=canvas.bbox("all"))

app = tk.Tk()
app.title("Run Wall Extraction and Crack Detection")
app.geometry("620x500")

main_canvas = tk.Canvas(app)
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar_y = ttk.Scrollbar(main_canvas, orient="vertical", command=main_canvas.yview)
scrollbar_y.pack(side=tk.RIGHT, fill="y")

# scrollbar_x = ttk.Scrollbar(main_canvas, orient="horizontal", command=main_canvas.xview)
# scrollbar_x.pack(side=tk.BOTTOM, fill="x")

scrollable_frame = ttk.Frame(main_canvas)
scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw",width=600, height=1000)
main_canvas.configure(yscrollcommand=scrollbar_y.set)
# main_canvas.configure(xscrollcommand=scrollbar_x.set)

tab_control = ttk.Notebook(scrollable_frame)
wall_extract_frame = ttk.Frame(tab_control)
infer_crack_frame = ttk.Frame(tab_control)
tab_control.add(wall_extract_frame, text="Step 1: Wall Extraction")
tab_control.add(infer_crack_frame, text="Step 2: Crack Detection")
tab_control.pack(expand=0, fill="both")

# main.py args
main_frame = ttk.LabelFrame(wall_extract_frame, text="main.py Arguments")
main_frame.pack(fill="x", padx=10, pady=10)

data_dir_label = ttk.Label(main_frame, text="Data Directory:")
data_dir_label.grid(row=0, column=0, padx=5, pady=5)
data_dir_entry = ttk.Entry(main_frame, width=40)
data_dir_entry.grid(row=0, column=1, padx=5, pady=5)
data_dir_button = ttk.Button(main_frame, text="Browse", command=select_data_dir)
data_dir_button.grid(row=0, column=2, padx=5, pady=5)

out_dir_label = ttk.Label(main_frame, text="Output Directory:")
out_dir_label.grid(row=1, column=0, padx=5, pady=5)
out_dir_entry = ttk.Entry(main_frame, width=40)
out_dir_entry.grid(row=1, column=1, padx=5, pady=5)
out_dir_button = ttk.Button(main_frame, text="Browse", command=select_out_dir)
out_dir_button.grid(row=1, column=2, padx=5, pady=5)

ckpt_path_label = ttk.Label(main_frame, text="Checkpoint Path:")
ckpt_path_label.grid(row=2, column=0, padx=5, pady=5)
ckpt_path_entry = ttk.Entry(main_frame, width=40)
ckpt_path_entry.grid(row=2, column=1, padx=5, pady=5)
ckpt_path_button = ttk.Button(main_frame, text="Browse", command=select_ckpt_path)
ckpt_path_button.grid(row=2, column=2, padx=5, pady=5)

save_img_var = tk.BooleanVar()
save_img_check = ttk.Checkbutton(main_frame, text="Save Images", variable=save_img_var)
save_img_check.grid(row=3, column=1, columnspan=2, padx=5, pady=5,sticky="w")

save_json_var = tk.BooleanVar()
save_json_check = ttk.Checkbutton(main_frame, text="Save JSON", variable=save_json_var)
save_json_check.grid(row=4, column=1, columnspan=2, padx=5, pady=5,sticky="w")

light_mode = tk.BooleanVar(value=True)
light_mode_check = ttk.Checkbutton(main_frame, text="Light Mode", variable=light_mode)
light_mode_check.grid(row=5, column=1, columnspan=2, padx=5, pady=5,sticky="w")

run_main_button = ttk.Button(main_frame, text="Run Step 1!", command=run_main)
run_main_button.grid(row=6, column=1, columnspan=1, padx=10, pady=10,sticky="ew")

# infer_crack.py args
infer_frame = ttk.LabelFrame(infer_crack_frame, text="infer_crack.py Arguments")
infer_frame.pack(fill="x", padx=10, pady=10)

img_dir_label = ttk.Label(infer_frame, text="Image Directory:")
img_dir_label.grid(row=0, column=0, padx=5, pady=5)
img_dir_entry = ttk.Entry(infer_frame, width=40)
img_dir_entry.grid(row=0, column=1, padx=5, pady=5)
img_dir_button = ttk.Button(infer_frame, text="Browse", command=select_img_dir)
img_dir_button.grid(row=0, column=2, padx=5, pady=5)

model_path_label = ttk.Label(infer_frame, text="Model Path:")
model_path_label.grid(row=1, column=0, padx=5, pady=5)
model_path_entry = ttk.Entry(infer_frame, width=40)
model_path_entry.grid(row=1, column=1, padx=5, pady=5)
model_path_button = ttk.Button(infer_frame, text="Browse", command=select_ckpt_path)
model_path_button.grid(row=1, column=2, padx=5, pady=5)

model_type_label = ttk.Label(infer_frame, text="Model Type:")
model_type_label.grid(row=2, column=0, padx=5, pady=5)
model_type_combobox = ttk.Combobox(infer_frame, values=["vgg16"],state="readonly")
model_type_combobox.grid(row=2, column=1, padx=5, pady=5)
model_type_combobox.set("vgg16")

out_viz_dir_label = ttk.Label(infer_frame, text="Visualization Output:")
out_viz_dir_label.grid(row=3, column=0, padx=5, pady=5)
out_viz_dir_entry = ttk.Entry(infer_frame, width=40)
out_viz_dir_entry.grid(row=3, column=1, padx=5, pady=5)
out_viz_dir_button = ttk.Button(infer_frame, text="Browse", command=select_out_viz_dir)
out_viz_dir_button.grid(row=3, column=2, padx=5, pady=5)

out_pred_dir_label = ttk.Label(infer_frame, text="Classification Output:")
out_pred_dir_label.grid(row=4, column=0, padx=5, pady=5)
out_pred_dir_entry = ttk.Entry(infer_frame, width=40)
out_pred_dir_entry.grid(row=4, column=1, padx=5, pady=5)
out_pred_dir_button = ttk.Button(infer_frame, text="Browse", command=select_out_pred_dir)
out_pred_dir_button.grid(row=4, column=2, padx=5, pady=5)

threshold_label = ttk.Label(infer_frame, text="Threshold:")
threshold_label.grid(row=5, column=0, padx=5, pady=5)
threshold_entry = ttk.Entry(infer_frame, width=40)
threshold_entry.grid(row=5, column=1, padx=5, pady=5)

run_infer_button = ttk.Button(infer_frame, text="Run Step 2!", command=run_infer_crack)
run_infer_button.grid(row=6, column=1, columnspan=1, padx=10, pady=10,sticky="ew")

main_frame.columnconfigure(0, minsize=150)
main_frame.columnconfigure(1, minsize=180)
main_frame.columnconfigure(2, minsize=50)
infer_frame.columnconfigure(0, minsize=150)
infer_frame.columnconfigure(1, minsize=180)
infer_frame.columnconfigure(2, minsize=50)


# Image Viewer
viewer_frame = ttk.LabelFrame(scrollable_frame, text="Image Viewer")
viewer_frame.pack(fill="both", padx=10, pady=10)

controls_frame_1 = ttk.Frame(viewer_frame)
controls_frame_1.pack(fill="x", padx=10, pady=10)

prev_button = ttk.Button(controls_frame_1, text="<<", command=show_previous_image)
prev_button.pack(side="left")

next_button = ttk.Button(controls_frame_1, text=">>", command=show_next_image)
next_button.pack(side="left")

refresh_button = ttk.Button(controls_frame_1, text="Refresh", command=refresh_images)
refresh_button.pack(side="left")

category_var = tk.StringVar(value="with_crack")
category_combobox = ttk.Combobox(controls_frame_1, textvariable=category_var, values=["with_crack", "without_crack"])
category_combobox.pack(side="right", padx=5)
category_combobox.bind("<<ComboboxSelected>>", lambda event: switch_category())

# Frame for the second row of controls (Zoom In, Zoom Out, Reset Zoom)
controls_frame_2 = ttk.Frame(viewer_frame)
controls_frame_2.pack(fill="x", padx=10, pady=10)

zoom_in_button = ttk.Button(controls_frame_2, text="Zoom In", command=zoom_in)
zoom_in_button.pack(side="left")

zoom_out_button = ttk.Button(controls_frame_2, text="Zoom Out", command=zoom_out)
zoom_out_button.pack(side="left")

reset_zoom_button = ttk.Button(controls_frame_2, text="Reset Zoom", command=reset_zoom)
reset_zoom_button.pack(side="left")

# Frame for the third row of controls (Category, Mask, Box, Threshold Slider)
controls_frame_3 = ttk.Frame(viewer_frame)
controls_frame_3.pack(fill="x", padx=10, pady=10)

mask_var = tk.BooleanVar()
mask_check = ttk.Checkbutton(controls_frame_3, text="Overlay Mask", variable=mask_var)
mask_check.pack(side="left", padx=5)

box_var = tk.BooleanVar()
box_check = ttk.Checkbutton(controls_frame_3, text="Overlay Bounding Box", variable=box_var)
box_check.pack(side="left", padx=5)

box_threshold_var = tk.DoubleVar(value=50)
box_threshold_slider = ttk.Scale(controls_frame_3, from_=0, to=100, variable=box_threshold_var, orient="horizontal")
box_threshold_slider.pack(side="left", padx=5)


scrollbar = ttk.Scrollbar(viewer_frame)
scrollbar.pack(side="right", fill="y")

canvas = tk.Canvas(viewer_frame, yscrollcommand=scrollbar.set)
canvas.pack(fill="both", expand=True)
scrollbar.config(command=canvas.yview)

canvas.bind("<MouseWheel>", zoom)
canvas.bind("<ButtonPress-1>", start_zoom)
canvas.bind("<B1-Motion>", update_zoom_rect)
canvas.bind("<ButtonRelease-1>", apply_zoom)

filename_label = ttk.Label(viewer_frame, text="")
filename_label.pack(fill="x")

description_text = tk.StringVar(value="Currently viewing images with cracks. Click << or >> to browse.")
description_label = ttk.Label(viewer_frame, textvariable=description_text)
description_label.pack(fill="x")



# Log output
log_frame = ttk.LabelFrame(scrollable_frame, text="Log Output")
log_frame.pack(fill="both", expand=True, padx=10, pady=10)

log_text = tk.Text(log_frame, wrap="word", height=10)
log_text.pack(fill="both", expand=True, padx=5, pady=5)

# Set default values for entries
data_dir_entry.insert(0, 'data/')
out_dir_entry.insert(0, 'output')
ckpt_path_entry.insert(0, 'ckp/sam_vit_h_4b8939.pth')
img_dir_entry.insert(0, './data')
model_path_entry.insert(0, './ckp/model_unet_vgg_16_best.pt')
out_viz_dir_entry.insert(0, './output/test_results')
out_pred_dir_entry.insert(0, './output/test_pred')
threshold_entry.insert(0, '0.1')

# Set the default image directory and refresh images
current_image_index = 0
image_files = []
image_dir = 'output/test_classification/with_crack'
refresh_images()

app.mainloop()
