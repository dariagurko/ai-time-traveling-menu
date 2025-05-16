# === Imports ===
import os
import time
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

import ollama
from stable_diffusion_pytorch import pipeline, model_loader

# === Configuration ===
device = 'cpu'
strength = 0.8
do_cfg = True
height = 256
width = 256
sampler = "k_euler"
use_seed = False
seed = 42 if use_seed else None
num_steps = 25
guidance_scale = 9

# === Load Stable Diffusion Model ===
@st.cache_resource
def load_model():
    return model_loader.preload_models(device)

model = load_model()

# === Image Generation ===
def generate_image(prompt, num_steps, guidance_scale, width, height):
    with torch.no_grad():
        image = pipeline.generate(
            prompts=prompt,
            input_images=[],
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=guidance_scale,
            height=height,
            width=width,
            sampler=sampler,
            n_inference_steps=num_steps,
            seed=seed,
            models=model,
            device=device,
            idle_device='cpu'
        )[0]
    return image

# === LLM Recipe Generation ===
def get_historical_recipe(era, region):
    prompt = (
        f"Generate an authentic {era} dish from {region}. "
        "Include ingredients, preparation steps, and a brief history of the dish."
    )
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# === Streamlit UI ===
st.title("AI Time-Traveling Menu 🍽️")
st.subheader("Discover historical dishes from different eras!")

# User Inputs
era = st.selectbox("Choose an Era", [
    "Ancient Rome", "Medieval Europe", "Victorian England",
    "Renaissance Italy", "Ancient Egypt", "Ancient Greece"
])
region = st.text_input("Enter a Region or City (e.g., Italy, France, Florence, Sparta)")

# Generate Button Logic
if st.button("Generate Recipe"):
    if not region:
        st.warning("Please enter a region before generating.")
    else:
        with st.spinner("Summoning the ancient kitchen..."):
            progress = st.progress(0)

            # Get recipe
            progress.progress(20)
            recipe = get_historical_recipe(era, region)
            progress.progress(50)

            st.success("Here’s your dish from the past!")
            st.markdown("### 📝 Recipe")
            st.write(recipe)

            # Store in session
            full_recipe_text = f"{era} Dish from {region}\n\n{recipe}"
            st.session_state["recipe"] = recipe
            st.session_state["recipe_text"] = full_recipe_text

            # Extract dish title (optional enhancement)
            if "Dish:" in recipe:
                dish_title = recipe.split("Dish:")[1].split("\n")[0].strip()
            else:
                dish_title = f"{era} dish from {region}"

            # Create image prompt
            image_prompt = (
                f"A high-resolution, highly detailed food photograph of {dish_title}, "
                f"a traditional historical dish from {era}, prepared in {region}, "
                f"plated in rustic ceramic tableware with historical presentation, "
                f"served on a wooden table, soft natural lighting, top-down view, realistic textures, styled composition"
            )

            # Generate image
            st.info("🎨 Preparing visual representation...")
            image = generate_image([image_prompt], num_steps, guidance_scale, width, height)

            image_path = Path("historical_dish.png")
            image.save(image_path)
            st.session_state["image_path"] = str(image_path)

            st.markdown("### 🖼️ Visual Representation")
            st.image(image, caption="AI-Generated Historical Dish", use_container_width=True)

            progress.progress(100)
            progress.empty()

# === Download Buttons ===
if "recipe_text" in st.session_state:
    st.markdown("### Downloads")
    st.download_button("Download Recipe", st.session_state["recipe_text"], file_name="historical_recipe.txt")

if "image_path" in st.session_state and os.path.exists(st.session_state["image_path"]):
    with open(st.session_state["image_path"], "rb") as img_file:
        st.download_button("Download Image", img_file, file_name="historical_dish.png", mime="image/png")
