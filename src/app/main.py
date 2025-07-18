### Code
## Here we will write the code for the NN

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

st.title("MedSegFlow web site")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Title
# st.markdown("<h1 style='text-align: center; color: navy;'>MedSegFlow 🧠</h1>", unsafe_allow_html=True)

# Sidebar
# st.sidebar.title("Μενού Πλοήγησης")

# Content
# if section == "Αρχική":
#     st.image("images/brain_scan.jpg", caption="MRI Image", use_column_width=True)
#     st.markdown("Καλωσήρθες στο MedSegFlow! Εδώ μπορείς να κάνεις segment ιατρικές εικόνες.")
# elif section == "Εκπαίδευση":
#     st.markdown("## 🎯 Εκπαίδευση Μοντέλου")
#     st.slider("Αριθμός εποχών", 1, 100, 10)
# elif section == "Αξιολόγηση":
#     st.markdown("## 📈 Αξιολόγηση Μοντέλου")
#     if uploaded_file is not None:
#         # Open image
#         image = Image.open(uploaded_file)

#         # Convert to numpy array
#         image_array = np.array(image)
#         shape = image_array.shape



if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Convert to numpy array
    image_array = np.array(image)
    shape = image_array.shape

    st.image(image)
    # st.write(f"Shape: {shape}")
    # st.image(image, caption=f"Shape: {shape}", use_column_width=True)
    st.write(f"Height: {shape[0]} pixels")
    st.write(f"Width: {shape[1]} pixels")
    if len(shape) == 3:
        st.write(f"Channels: {shape[2]}")
    else:
        st.write("Channels: 1 (Grayscale)")