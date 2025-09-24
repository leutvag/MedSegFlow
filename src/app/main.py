# main.py
### MedSegFlow CNN + Heatmap + Contour + AutoAugmentation + EarlyStopping
# cd MedSegFlow/src/app
# streamlit run main.py

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from glob import glob
import os
import zipfile
import io
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import cv2
from skimage import measure

# ---------------------------
# Βασικές ρυθμίσεις
# ---------------------------
st.set_page_config(page_title="MedSegFlow CNN Analyzer", layout="centered")
st.markdown("<h1 style='text-align: center; color: rgb(200,200,125);'>🧠 MedSegFlow CNN Analyzer (YES/NO)</h1>", unsafe_allow_html=True)

# ---------------------------
# Helper functions
# ---------------------------
def extract_zip_to_folder(zip_bytes, target_folder="dataset_temp"):
    if os.path.exists(target_folder):
        import shutil
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    z.extractall(target_folder)
    return target_folder

def create_image_generators(folder, target_size=(224,224), batch_size=8):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        brightness_range=(0.8,1.2),
        zoom_range=0.1,
        shear_range=0.1
    )
    train_gen = datagen.flow_from_directory(
        folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    return train_gen, val_gen

def build_model(input_shape=(224,224,3)):
    base_model = applications.EfficientNetB0(
        include_top=False, input_shape=input_shape, weights='imagenet'
    )
    base_model.trainable = True
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Grad-CAM
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap)+1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[2], img_array.shape[1]))
    # heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    return heatmap

def overlay_heatmap_on_image(img_pil, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    img = np.array(img_pil.convert("RGB"))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), colormap)
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlayed)

def draw_contour_on_heatmap(img_pil, heatmap, threshold=0.5):
    mask = (heatmap >= threshold).astype(np.uint8)
    contours = measure.find_contours(mask, 0.5)
    img_draw = img_pil.convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    y_scale = img_pil.height / mask.shape[0]
    x_scale = img_pil.width / mask.shape[1]
    for contour in contours:
        contour_scaled = [(c[1]*x_scale, c[0]*y_scale) for c in contour]
        draw.line(contour_scaled + [contour_scaled[0]], fill=(0,255,0), width=3)
    return img_draw

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("🔧 Settings")
target_size = st.sidebar.selectbox("Image size", [(224,224),(128,128)], index=0)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=5)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=32, value=8)
heatmap_threshold = st.sidebar.slider("Threshold for contour", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# ---------------------------
# Upload dataset
# ---------------------------
st.header("🏋️ Training")
st.write("Dataset: φάκελος με δύο υποφακέλους `YES` και `NO`")

uploaded_zip = st.file_uploader("Upload dataset zip", type=["zip"])
train_button = st.button("▶ Train CNN")

if train_button:
    if uploaded_zip is None:
        st.warning("Πρέπει να ανεβάσεις dataset zip πρώτα!")
    else:
        folder = extract_zip_to_folder(uploaded_zip.getvalue())
        st.success("Dataset εξήχθη.")

        train_gen, val_gen = create_image_generators(folder, target_size=target_size, batch_size=batch_size)
        st.success(f"Training samples: {train_gen.samples}, Validation samples: {val_gen.samples}")

        model = build_model(input_shape=(target_size[0], target_size[1],3))
        st.success("Model έτοιμο.")

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        with st.spinner("Training σε εξέλιξη..."):
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=[early_stop]
            )
        st.success("Training ολοκληρώθηκε!")
        model.save("cnn_yesno_model.h5")
        st.download_button("💾 Κατέβασε weights (.h5)", data=open("cnn_yesno_model.h5","rb").read(), file_name="cnn_yesno_model.h5")

# ---------------------------
# Prediction panel
# ---------------------------
st.header("🔍 Πρόβλεψη & Heatmap + Contour")
uploaded_file = st.file_uploader("Upload MRI image", type=["jpg","png","jpeg"])
predict_button = st.button("▶ Predict")

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.subheader("MRI Image")
    st.image(pil_img, use_column_width=True)

    if predict_button:
        try:
            img_resized = pil_img.resize(target_size)
            img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

            model = tf.keras.models.load_model("cnn_yesno_model.h5", compile=False)

            pred = model.predict(img_array)[0,0]
            st.success(f"Prediction: {'YES (όγκος)' if pred>0.5 else 'NO (όχι όγκος)'} ({pred:.2f})")

            heatmap = get_gradcam_heatmap(img_array, model)
            overlayed_img = overlay_heatmap_on_image(pil_img, heatmap)
            st.subheader("Heatmap (πιθανή περιοχή όγκου)")
            st.image(overlayed_img, use_column_width=True)

            contour_img = draw_contour_on_heatmap(pil_img, heatmap, threshold=heatmap_threshold)
            st.subheader("Contour γύρω από πιθανή περιοχή όγκου")
            st.image(contour_img, use_column_width=True)

        except Exception as e:
            st.error(f"Σφάλμα: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
**Σχετικά με την εφαρμογή:**  
Αυτό το Streamlit app εκπαιδεύει ένα CNN για να ταξινομεί MRI εικόνες σε YES (όγκος) ή NO (χωρίς όγκο).  
Μετά την εκπαίδευση ή φόρτωση του μοντέλου, μπορεί να κάνει πρόβλεψη για νέες εικόνες και να εμφανίζει:  
- **Prediction**: πιθανότητα ύπαρξης όγκου.  
- **Grad-CAM Heatmap**: πιθανοί όγκοι με έγχρωμη επισήμανση.  
- **Contour**: ακριβή περίγραμμα γύρω από περιοχές υψηλής πιθανότητας όγκου.
""")
