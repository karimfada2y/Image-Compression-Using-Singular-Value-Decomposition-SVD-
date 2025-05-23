import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="SVD Image Compressor", layout="centered")

st.title("📉 SVD Image Compression App")

# Upload image
uploaded_file = st.file_uploader("Upload a grayscale image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    A = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    # Slider for k
    k = st.slider("Select number of singular values (k)", min_value=10, max_value=min(A.shape), value=50, step=5)

    # SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    def compress_image(U, S, Vt, k):
        Uk = U[:, :k]
        Sk = np.diag(S[:k])
        Vk = Vt[:k, :]
        return Uk @ Sk @ Vk

    # Compress
    Ak = compress_image(U, S, Vt, k)
    Ak = np.clip(Ak, 0, 255)

    # Show result
    compressed_img = Image.fromarray(Ak.astype(np.uint8))
    st.image(compressed_img, caption=f"Compressed Image (k={k})", use_column_width=True)

    # Metrics
    original_size = np.prod(A.shape)
    compressed_size = k * (A.shape[0] + A.shape[1] + 1)
    ratio = compressed_size / original_size
    error = np.linalg.norm(A - Ak, ord='fro')

    st.markdown("### 📊 Compression Metrics")
    st.write(f"**Compression Ratio:** {ratio:.4f}")
    st.write(f"**Reconstruction Error (Frobenius norm):** {error:.4f}")
