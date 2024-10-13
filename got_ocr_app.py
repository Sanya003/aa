from PIL import Image
import streamlit as st
import torch
import os

def load_model():
  from transformers import AutoModel, AutoTokenizer
  
  tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
  model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
  model.eval().cuda()
  return model, tokenizer

def save_temp_image(image):
  temp_path = "temp_image.png"
  image.save(temp_path)
  return temp_path

st.title('📝 Textify ')
st.divider()

uploaded_img = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')

if uploaded_img is not None:
  image = Image.open(uploaded_img)

  st.image(image, caption='Uploaded image', use_column_width=True)
  st.divider()

  model, tokenizer = load_model()
  image_path = save_temp_image(image)
  
  with st.spinner('Performing OCR, please wait...'):
    extracted_text = model.chat_crop(tokenizer, image_path, ocr_type='ocr')

  st.header('Extracted Text')
  st.text_area('OCR output', value=extracted_text, height=300)

  os.remove(image_path)

  st.download_button('📩 Download Text', data=extracted_text, file_name='ocr_output.txt', mime='text/plain')
