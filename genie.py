from transformers import BlipProcessor, BlipForConditionalGeneration
from tkinter import Tk, filedialog
from tkinter import ttk  # Using ttk for modern widget styling
from PIL import Image, ImageTk

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# generate caption
def generate_image_caption(image_file_path):
    image = Image.open(image_file_path).convert('RGB')
    input_tensors = processor(image, return_tensors="pt")
    output_ids = model.generate(**input_tensors)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.capitalize()

# image upload process
def upload_and_process_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        generated_caption = generate_image_caption(file_path)
        uploaded_image = Image.open(file_path).resize((300, 300))
        image_to_display = ImageTk.PhotoImage(uploaded_image)
        image_display_label.configure(image=image_to_display)
        image_display_label.image = image_to_display
        caption_display_label.config(text=f"Caption: {generated_caption}")

# main appwindow
application_window = Tk()
application_window.title("Image Caption Generator")
application_window.geometry("450x600")
application_window.configure(bg="#f0f8ff") 

# heading label
heading_label = ttk.Label(application_window, text="Upload an Image to Generate a Caption", font=("Helvetica", 16, "bold"))
heading_label.pack(pady=20)

# frame to organize the image, caption
display_frame = ttk.Frame(application_window)
display_frame.pack(pady=20)

# label to display the uploaded image
image_display_label = ttk.Label(display_frame, relief="solid")
image_display_label.pack(pady=10)

# button with style
upload_button = ttk.Button(application_window, text="Upload Image", command=upload_and_process_image)
upload_button.pack(pady=10)

# label to display the generated caption
caption_display_label = ttk.Label(application_window, text="", wraplength=400, font=("Arial", 12))
caption_display_label.pack(pady=20)

# footer with subtle
footer_label = ttk.Label(application_window, text="Powered by BLIP | Hugging Face Transformers", font=("Arial", 10, "italic"))
footer_label.pack(side="bottom", pady=10)

# main app loop
application_window.mainloop()
