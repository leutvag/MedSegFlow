### Code
## Here we will write the code for the NN
# cd MedSegFlow/src/app
# streamlit run main.py

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import time
import io

# st.title("MedSegFlow web site")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Open image
#     image = Image.open(uploaded_file)

#     # Convert to numpy array
#     image_array = np.array(image)
#     shape = image_array.shape

#     st.image(image)
#     # st.write(f"Shape: {shape}")
#     # st.image(image, caption=f"Shape: {shape}", use_column_width=True)
#     st.write(f"Height: {shape[0]} pixels")
#     st.write(f"Width: {shape[1]} pixels")
#     if len(shape) == 3:
#         st.write(f"Channels: {shape[2]}")
#     else:
#         st.write("Channels: 1 (Grayscale)")

st.set_page_config(page_title="MedSegFlow", layout="centered")
# st.title("🧠 MedSegFlow MRI Analyzer")
st.markdown("<h1 style='text-align: center; color: rgb(200,200,125);'>🧠 MedSegFlow MRI Analyzer</h1>", unsafe_allow_html=True)


st.sidebar.header("📂 Ανέβασε MRI εικόνα")
uploaded_file = st.sidebar.file_uploader("Επέλεξε εικόνα...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    st.image(image, caption="🖼️ Αρχική εικόνα", use_column_width=True)
    st.write("### ℹ️ Στοιχεία εικόνας:")
    st.write(f"Height: {image_array.shape[0]} px")
    st.write(f"Width: {image_array.shape[1]} px")
    if len(image_array.shape) == 3:
        st.write(f"Channels: {image_array.shape[2]}")
    else:
        st.write("Channels: 1 (Grayscale)")

    st.markdown("---")
    st.write("🔍 **Ανίχνευση βλάβης**")

    with st.spinner("Γίνεται ανάλυση..."):
        time.sleep(2)  # Εδώ θα μπει το μοντέλο σου
        prediction = "Εντοπίστηκε πιθανός όγκος στον αριστερό λοβό"
        dice_score = 0.87
        accuracy = 0.92

    st.success("✅ Ανάλυση ολοκληρώθηκε!")

    st.markdown(f"**🧾 Πρόβλεψη:** {prediction}")
    st.markdown(f"**🎯 Dice Score:** {dice_score:.2f}")
    st.markdown(f"**📊 Accuracy:** {accuracy:.2%}")

    # Παράδειγμα: Εμφάνιση segmented mask
    st.markdown("### 🧩 Segmented Image")
    # st.image(segmented_image, caption="Segmented Output")

    # Προσθήκη κουμπιού για αποθήκευση
#     st.download_button("💾 Κατέβασε αποτελέσματα", data="dummy", file_name="result.txt")
# else:
#     st.info("Παρακαλώ ανέβασε εικόνα MRI από τον εγκέφαλο.")
download_format = st.selectbox("📥 Επίλεξε μορφή αποθήκευσης αποτελεσμάτων:", ["CSV", "TXT"])
results = {
    "Prediction": prediction,
    "Dice Score": dice_score,
    "Accuracy": accuracy,
    "Image Height": image_array.shape[0],
    "Image Width": image_array.shape[1],
    "Channels": image_array.shape[2] if len(image_array.shape) == 3 else 1
}
if download_format == "CSV":
    df = pd.DataFrame([results])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="💾 Κατέβασε CSV",
        data=csv_buffer.getvalue(),
        file_name="medsegflow_results.csv",
        mime="text/csv"
    )
else:
    txt_output = "\n".join([f"{key}: {value}" for key, value in results.items()])
    st.download_button(
        label="💾 Κατέβασε TXT",
        data=txt_output,
        file_name="medsegflow_results.txt",
        mime="text/plain"
    )
