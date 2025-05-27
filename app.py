import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

@st.cache_resource

def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("./gpt2-food-fact-model_20250526_203152")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🍽️ Food Fact Generator")
user_input = st.text_input("Enter a food item:")

if st.button("Generate Fun Fact"):
    with st.spinner("Generating..."):
        input_text = f"<|startoftext|>{user_input} ->"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.95)
        fact = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        st.success(fact)
        
# Deploy using Streamlit Sharing, Hugging Face Spaces, or Render.com
# You can push model to HF Hub using model.push_to_hub("your-model-name")
# and deploy with Gradio or Streamlit.

# This setup provides an end-to-end robust implementation of the Food Fact Generator using GPT-2.
