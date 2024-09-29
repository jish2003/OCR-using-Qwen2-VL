import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from PIL import Image
import io
from qwen_vl_utils import process_vision_info
import re

# ... other imports

# Load pre-trained model and tokenizer
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)

def extract_text(image):
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

# Load the image using the file path
  image = Image.open(image)

#   print(image)


  # Existing Colab code for image processing, text extraction, and model inference
  # Adapt paths and references to work within Streamlit
  messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all the text in Sanskrit and English from the image."}],
        }
    ]
  

  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  image_inputs, _ = process_vision_info(messages)

# Process inputs
  inputs = processor(
      text=[text],
      images=image_inputs,
      padding=True,
      return_tensors="pt"
  )

  inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to GPU if available


  with torch.no_grad():
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=2000, no_repeat_ngram_size=3, temperature=0.7)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
  # ... rest of the code for processing and returning extracted text
  return output_text




# Streamlit App Interface
st.title("OCR with Keyword Search")

uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
keyword = st.text_input("Enter a keyword to search")

if st.button("Extract Text"):
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    extracted_text = extract_text(image)

    if keyword:
      highlighted_text = re.sub(rf"(?i){keyword}", f"<mark>{keyword}</mark>", extracted_text)
    else:
      highlighted_text = extracted_text

    st.write(highlighted_text, unsafe_allow_html=True)

    st.image(image, caption='Uploaded Image', use_column_width=True)