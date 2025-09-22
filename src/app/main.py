# main.py
### MedSegFlow: Streamlit app με U-Net, training & overlay
# cd MedSegFlow/src/app
# streamlit run main.py

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from skimage.transform import resize
from skimage import measure
import io
import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
from glob import glob

# ---------------------------
# Βασικές ρυθμίσεις
# ---------------------------
st.set_page_config(page_title="MedSegFlow", layout="centered")
st.markdown("<h1 style='text-align: center; color: rgb(200,200,125);'>🧠 MedSegFlow MRI Analyzer</h1>", unsafe_allow_html=True)

# ---------------------------
# Utility functions
# ---------------------------
def normalize_mri_pil(pil_image, target_size=(256,256)):
    image_array = np.array(pil_image).astype(np.float32)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = np.mean(image_array, axis=2)
    mn, mx = np.min(image_array), np.max(image_array)
    if mx - mn < 1e-8:
        image_norm = np.zeros_like(image_array)
    else:
        image_norm = (image_array - mn) / (mx - mn)
    image_resized = resize(image_norm, target_size, anti_aliasing=True)
    return image_resized, np.expand_dims(image_resized, axis=0)

def prepare_image_for_model(pil_image, target_size=(256,256)):
    resized, _ = normalize_mri_pil(pil_image, target_size=target_size)
    return np.expand_dims(resized, axis=(0, -1))  # shape (1,H,W,1)

def overlay_mask_on_image(orig_image_pil, mask, alpha=0.4):
    orig = np.array(orig_image_pil).astype(np.uint8)
    if orig.ndim == 2:
        orig_rgb = np.stack([orig, orig, orig], axis=-1)
    elif orig.ndim == 3 and orig.shape[2] == 4:
        orig_rgb = orig[:, :, :3]
    else:
        orig_rgb = orig.copy()

    mask_resized = resize(mask, (orig_rgb.shape[0], orig_rgb.shape[1]), order=0, preserve_range=True, anti_aliasing=False)
    mask_bin = (mask_resized > 0.5).astype(np.uint8)

    overlay = orig_rgb.copy()
    overlay[mask_bin == 1] = [255, 0, 0]

    blended = ((1 - alpha) * orig_rgb + alpha * overlay).astype(np.uint8)
    return Image.fromarray(blended)

def draw_contour(orig_image_pil, mask, threshold=0.5):
    """
    Σχεδιάζει contour γύρω από τον όγκο στη μάσκα.
    """
    mask_bin = (mask > threshold).astype(np.uint8)
    contours = measure.find_contours(mask_bin, level=0.5)
    
    y_scale = orig_image_pil.height / mask.shape[0]
    x_scale = orig_image_pil.width / mask.shape[1]

    img_draw = orig_image_pil.convert("RGB")
    draw = ImageDraw.Draw(img_draw)

    for contour in contours:
        contour_scaled = [(c[1]*x_scale, c[0]*y_scale) for c in contour]
        draw.line(contour_scaled + [contour_scaled[0]], fill="red", width=2)

    return img_draw

def draw_precise_green_contour(orig_image_pil, mask=None, adaptive=True):
    """
    Σχεδιάζει ακριβές contour γύρω από τον όγκο στην εικόνα.
    mask: αν υπάρχει, χρησιμοποιείται η μάσκα του μοντέλου
    adaptive: αν True, χρησιμοποιεί adaptive threshold από τη φωτεινότητα της εικόνας
    """
    # Μετατρέπουμε σε grayscale array
    img_array = np.array(orig_image_pil).astype(np.float32)
    if img_array.ndim == 3:
        img_array = np.mean(img_array, axis=2)
    img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)

    if mask is None:
        # Adaptive threshold: pixels πιο λευκά από μέσο+0.5*std θεωρούνται όγκος
        if adaptive:
            thresh = img_norm.mean() + 0.5 * img_norm.std()
            mask_bin = (img_norm >= thresh).astype(np.uint8)
        else:
            mask_bin = (img_norm > 0.5).astype(np.uint8)
    else:
        mask_bin = (mask > 0.5).astype(np.uint8)

    contours = measure.find_contours(mask_bin, level=0.5)

    y_scale = orig_image_pil.height / mask_bin.shape[0]
    x_scale = orig_image_pil.width / mask_bin.shape[1]

    img_draw = orig_image_pil.convert("RGB")
    draw = ImageDraw.Draw(img_draw)

    for contour in contours:
        contour_scaled = [(c[1]*x_scale, c[0]*y_scale) for c in contour]
        # Σχεδίασε γραμμή πράσινη με width=2
        draw.line(contour_scaled + [contour_scaled[0]], fill=(0,255,0), width=2)

    return img_draw



# ---------------------------
# U-Net model (Keras)
# ---------------------------
def unet_model(input_size=(256,256,1)):
    inputs = tf.keras.Input(shape=input_size)

    def conv_block(x, n_filters):
        x = layers.Conv2D(n_filters, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(n_filters, 3, activation="relu", padding="same")(x)
        return x

    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D((2,2))(c3)
    b = conv_block(p3, 128)

    u3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, 64)
    u2 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 32)
    u1 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, 16)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c6)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("🔧 Settings")
model_size = st.sidebar.selectbox("Input size", [(256,256),(128,128)], index=0)
target_size = model_size
epochs = st.sidebar.number_input("Epochs (training)", min_value=1, max_value=200, value=10)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=32, value=4)

@st.cache_resource
def get_model(input_size):
    model = unet_model(input_size=input_size + (1,))
    model.compile(optimizer="adam", loss=bce_dice_loss, metrics=[dice_coef, "accuracy"])
    return model

model = get_model(target_size)

st.sidebar.markdown("---")
st.sidebar.header("📁 Data / Weights")
uploaded_weights = st.sidebar.file_uploader("Φόρτωσε weights (.h5) (προαιρετικό)", type=["h5"])
if uploaded_weights is not None:
    tmp_path = "tmp_weights.h5"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_weights.getbuffer())
    try:
        model.load_weights(tmp_path)
        st.sidebar.success("Weights φορτώθηκαν.")
    except Exception as e:
        st.sidebar.error(f"Σφάλμα φόρτωσης weights: {e}")

# ---------------------------
# Training panel
# ---------------------------
st.header("🏋️ Training (προαιρετικό)")
st.write("Αν έχεις dataset, στείλε zip με δύο folders: `images/` και `masks/` (όλα σε μορφή jpg/png). "
         "Τα ονόματα εικόνων και masks πρέπει να ταιριάζουν (π.χ. images/001.png <--> masks/001.png).")

uploaded_zip = st.file_uploader("Άνέβασε dataset zip (images + masks)", type=["zip"])
train_button = st.button("▶ Εκκίνηση Training")

def extract_zip_to_folder(zip_bytes, target_folder="dataset_temp"):
    if os.path.exists(target_folder):
        import shutil
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    z.extractall(target_folder)
    return target_folder

def load_dataset_from_folder(folder, image_sub="images", mask_sub="masks", target_size=(256,256)):
    img_paths = sorted(glob(os.path.join(folder, image_sub, "*")))
    mask_paths = sorted(glob(os.path.join(folder, mask_sub, "*")))
    X, Y = [], []
    for ip, mp in zip(img_paths, mask_paths):
        pil_img = Image.open(ip).convert("L")
        pil_mask = Image.open(mp).convert("L")
        arr_img, _ = normalize_mri_pil(pil_img, target_size=target_size)
        mask_arr = np.array(pil_mask).astype(np.float32)
        mask_arr = (mask_arr - mask_arr.min()) / (mask_arr.max() - mask_arr.min() + 1e-8)
        mask_resized = resize(mask_arr, target_size, anti_aliasing=False)
        mask_resized = (mask_resized > 0.5).astype(np.float32)
        X.append(np.expand_dims(arr_img, axis=-1))
        Y.append(np.expand_dims(mask_resized, axis=-1))
    return np.array(X), np.array(Y)

if train_button:
    if uploaded_zip is None:
        st.warning("Πρέπει να ανεβάσεις zip με dataset πριν ξεκινήσει το training.")
    else:
        st.info("Εξαγωγή zip και φόρτωση δεδομένων...")
        try:
            folder = extract_zip_to_folder(uploaded_zip.getvalue(), target_folder="dataset_temp")
            X, Y = load_dataset_from_folder(folder, target_size=target_size)
            st.success(f"Φορτώθηκαν {len(X)} δείγματα.")
            st.info("Ξεκινάει το training...")
            with st.spinner("Training σε εξέλιξη..."):
                history = model.fit(X, Y, epochs=int(epochs), batch_size=int(batch_size), validation_split=0.1)
            st.success("Training ολοκληρώθηκε.")
            model.save_weights("trained_medsegflow_weights.h5")
            st.download_button("Κατέβασε weights (.h5)", data=open("trained_medsegflow_weights.h5","rb").read(), file_name="medsegflow_weights.h5")
        except Exception as e:
            st.error(f"Σφάλμα κατά το training: {e}")

# ---------------------------
# Prediction panel
# ---------------------------
st.header("🔍 Πρόβλεψη & Overlay")
st.write("Ανέβασε μία MRI εικόνα (jpg/png). Το μοντέλο θα προβλέψει μάσκα και θα την εμφανίσει πάνω στην εικόνα.")

uploaded_file = st.file_uploader("Επέλεξε εικόνα για πρόβλεψη", type=["jpg","jpeg","png"])
predict_button = st.button("▶ Πρόβλεψη")

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("L")
    st.subheader("Αρχική εικόνα")
    st.image(pil_img, use_column_width=True)

    st.write("Κανονικοποίηση και προετοιμασία για μοντέλο...")
    input_for_model = prepare_image_for_model(pil_img, target_size=target_size)

    if predict_button:
        try:
            with st.spinner("Κάνει πρόβλεψη..."):
                pred = model.predict(input_for_model)
                pred_mask = pred[0, :, :, 0]

                # Overlay
                overlayed = overlay_mask_on_image(pil_img, pred_mask, alpha=0.45)
                st.success("Πρόβλεψη ολοκληρώθηκε!")

                st.subheader("Mask (πιθανότητα)")
                st.image((pred_mask*255).astype(np.uint8), caption="Προβλεπόμενη μάσκα (probabilities)", use_column_width=True)

                st.subheader("MRI με overlay όπου βρέθηκε όγκος")
                st.image(overlayed, use_column_width=True)

                # Contour γύρω από τον όγκο
                # contour_img = draw_contour(pil_img, pred_mask)
                contour_img = draw_precise_green_contour(pil_img, mask=pred_mask, adaptive=True)
                st.subheader("MRI με contour γύρω από τον όγκο")
                st.image(contour_img, use_column_width=True)

                # Αποθήκευση overlay
                buf = io.BytesIO()
                overlayed.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("💾 Κατέβασε εικόνα με overlay", data=buf, file_name="medsegflow_overlay.png", mime="image/png")

                # Αποθήκευση contour
                buf_contour = io.BytesIO()
                contour_img.save(buf_contour, format="PNG")
                buf_contour.seek(0)
                st.download_button("💾 Κατέβασε εικόνα με contour", data=buf_contour, file_name="medsegflow_contour.png", mime="image/png")

        except Exception as e:
            st.error(f"Σφάλμα στην πρόβλεψη: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("**Οδηγίες / Σημειώσεις:**")
