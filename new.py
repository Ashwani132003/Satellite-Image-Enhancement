import streamlit as st
import cv2
import numpy as np
import pywt
from PIL import Image
import streamlit.components.v1 as components
import base64
from io import BytesIO
import os
import os.path as osp
import torch
import RRDBNet_arch as arch


st.set_page_config(layout="wide")
# Inject CSS to hide top padding and Streamlit's menu
hide_st_style = """
    <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 0rem;
        }
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("Satellite Image Enhancement Toolkit")
st.markdown("Upload a satellite image and apply multiple enhancement techniques in sequence.")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

# Enhancement functions
def apply_clahe(image_bgr, clip_limit, tile_grid_size):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced

def histogram_equalization(image_bgr):
    img_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return hist_eq

def gamma_correction(image_rgb, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image_rgb, table)

def unsharp_mask(image_rgb, kernel_size=(5, 5), amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image_rgb, kernel_size, 0)
    sharpened = cv2.addWeighted(image_rgb, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.abs(image_rgb - blurred) < threshold
        np.copyto(sharpened, image_rgb, where=low_contrast_mask)
    return sharpened

def bilateral_filter(image_rgb, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(image_rgb, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

def thresholding(image_rgb, thresh_val):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def laplacian_filter(image_rgb, ksize):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    laplacian = cv2.convertScaleAbs(laplacian)
    return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)

def wavelet_enhancement(image_rgb, wavelet_type='db1', level=1):
    channels = cv2.split(image_rgb)
    enhanced_channels = []
    for ch in channels:
        coeffs = pywt.wavedec2(ch, wavelet=wavelet_type, level=level)
        cA, detail_coeffs = coeffs[0], coeffs[1:]
        detail_coeffs = [(pywt.threshold(d[0], 10, 'soft'),
                          pywt.threshold(d[1], 10, 'soft'),
                          pywt.threshold(d[2], 10, 'soft')) for d in detail_coeffs]
        coeffs = [cA] + detail_coeffs
        reconstructed = pywt.waverec2(coeffs, wavelet=wavelet_type)
        enhanced = np.clip(reconstructed, 0, 255).astype(np.uint8)
        enhanced_channels.append(enhanced)
    return cv2.merge(enhanced_channels)

def convert_to_grayscale(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def gaussian_filter(image_rgb, ksize):
    return cv2.GaussianBlur(image_rgb, (ksize, ksize), 0)
# Function to create magnifier HTML for a given image
def display_magnifier(image_array, label):
    buffered = BytesIO()
    pil_image = Image.fromarray(image_array)
    pil_image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    html_code = f"""
    <style>
    .img-magnifier-container {{ position: relative; }}
    .img-magnifier-glass {{
        position: absolute;
        border: 3px solid #000;
        border-radius: 50%;
        cursor: none;
        width: 150px;
        height: 150px;
        display: none;
        z-index: 10;
    }}
    </style>
    <div class="img-magnifier-container">
        <img id="{label}" src="data:image/jpeg;base64,{img_base64}" width="600">
    </div>
    <script>
    function magnify(imgID, zoom) {{
        var img, glass, w, h, bw;
        img = document.getElementById(imgID);
        glass = document.createElement("DIV");
        glass.setAttribute("class", "img-magnifier-glass");
        img.parentElement.insertBefore(glass, img);
        glass.style.backgroundImage = "url('" + img.src + "')";
        glass.style.backgroundRepeat = "no-repeat";
        glass.style.backgroundSize = (img.width * zoom) + "px " + (img.height * zoom) + "px";
        bw = 3; w = glass.offsetWidth / 2; h = glass.offsetHeight / 2;
        img.addEventListener("mousemove", moveMagnifier);
        img.addEventListener("mouseenter", function() {{ glass.style.display = "block"; }});
        img.addEventListener("mouseleave", function() {{ glass.style.display = "none"; }});
        function moveMagnifier(e) {{
            var pos = getCursorPos(e), x = pos.x, y = pos.y;
            if (x > img.width - (w / zoom)) {{ x = img.width - (w / zoom); }}
            if (x < w / zoom) {{ x = w / zoom; }}
            if (y > img.height - (h / zoom)) {{ y = img.height - (h / zoom); }}
            if (y < h / zoom) {{ y = h / zoom; }}
            glass.style.left = (x - w) + "px";
            glass.style.top = (y - h) + "px";
            glass.style.backgroundPosition = "-" + ((x * zoom) - w + bw) + "px -" + ((y * zoom) - h + bw) + "px";
        }}
        function getCursorPos(e) {{
            var a = img.getBoundingClientRect(), x = e.pageX - a.left, y = e.pageY - a.top;
            x = x - window.pageXOffset; y = y - window.pageYOffset;
            return {{x : x, y : y}};
        }}
    }}
    magnify("{label}", 2);
    </script>
    """
    # components.html(html_code, height=1000)
    # components.html(html_code)
    components.html(html_code, height=current_image.shape[0])  # +100 for padding/UI margin





@st.cache_resource
def load_esrgan_model():
    model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)
    return model, device


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)



    st.sidebar.header("Sequential Enhancement Steps")

    enhancement_steps = st.sidebar.multiselect(
        "Select enhancement techniques in order",
        [
            "CLAHE",
            "Histogram Equalization",
            "Gamma Correction",
            "Unsharp Masking",
            "Bilateral Filtering",
            "Gaussian Filtering", 
            "Thresholding",
            "Laplacian Filter",
            "Wavelet Enhancement",
            "Grayscale Conversion",
            "Super Resolution"
        ]
    )

    with st.spinner("Applying enhancement steps..."):
        current_image = image_np.copy()

        for i, step in enumerate(enhancement_steps):
            st.sidebar.markdown(f"### Step {i + 1}: {step}")
            if step == "CLAHE":
                clip_limit = st.sidebar.slider(f"Clip Limit (Step {i+1})", 1.0, 10.0, 2.0, step=0.1, key=f"clip_{i}")
                tile_grid_size = st.sidebar.slider(f"Tile Grid Size (Step {i+1})", 2, 16, 8, key=f"tile_{i}")
                img_bgr = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                current_image = apply_clahe(img_bgr, clip_limit, tile_grid_size)

            elif step == "Histogram Equalization":
                img_bgr = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                current_image = histogram_equalization(img_bgr)

            elif step == "Gamma Correction":
                gamma = st.sidebar.slider(f"Gamma (Step {i+1})", 0.1, 3.0, 1.0, step=0.1, key=f"gamma_{i}")
                current_image = gamma_correction(current_image, gamma)

            elif step == "Unsharp Masking":
                amount = st.sidebar.slider(f"Amount (Step {i+1})", 0.5, 3.0, 1.5, step=0.1, key=f"sharp_{i}")
                current_image = unsharp_mask(current_image, amount=amount)

            elif step == "Bilateral Filtering":
                d = st.sidebar.slider(f"Diameter (Step {i+1})", 1, 20, 9, key=f"d_{i}")
                sigma_color = st.sidebar.slider(f"Sigma Color (Step {i+1})", 1, 250, 75, key=f"color_{i}")
                sigma_space = st.sidebar.slider(f"Sigma Space (Step {i+1})", 1, 250, 75, key=f"space_{i}")
                current_image = bilateral_filter(current_image, d, sigma_color, sigma_space)

            elif step == "Gaussian Filtering":
                ksize = st.sidebar.selectbox(f"Kernel Size (odd only) (Step {i+1})", [3, 5, 7, 9, 11], index=1, key=f"gauss_{i}")
                current_image = gaussian_filter(current_image, ksize)

            # elif step == "Thresholding":
            #     thresh_val = st.sidebar.slider(f"Threshold Value (Step {i+1})", 0, 255, 128, key=f"thresh_{i}")
            #     current_image = thresholding(current_image, thresh_val)
            elif step == "Thresholding":
                # Convert to grayscale if needed
                gray_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY) if current_image.ndim == 3 else current_image

                # Get Otsu's threshold value
                otsu_thresh_val, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Slider with Otsu's value as default
                thresh_val = st.sidebar.slider(
                    f"Threshold Value (Step {i+1})",
                    0, 255,
                    int(otsu_thresh_val),  # default value = Otsu's threshold
                    key=f"thresh_{i}"
                )

                current_image = thresholding(current_image, thresh_val)


            elif step == "Laplacian Filter":
                ksize = st.sidebar.selectbox(f"Kernel Size (Step {i+1})", [1, 3, 5, 7], index=1, key=f"lap_{i}")
                current_image = laplacian_filter(current_image, ksize)

            elif step == "Wavelet Enhancement":
                wavelet = st.sidebar.selectbox(f"Wavelet Type (Step {i+1})", ['db1', 'haar', 'sym2'], index=0, key=f"wav_{i}")
                level = st.sidebar.slider(f"Decomposition Level (Step {i+1})", 1, 3, 1, key=f"lvl_{i}")
                current_image = wavelet_enhancement(current_image, wavelet, level)

            elif step == "Grayscale Conversion":
                current_image = convert_to_grayscale(current_image)

            # elif step == 'Super Resolution':
            #     # model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'
            #     # # Auto-select device (use CPU if CUDA is not available or you face OOM)
            #     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #     # print(f"Using device: {device}")
            #     # model = arch.RRDBNet(3, 3, 64, 23, gc=32)
            #     # model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
            #     # model.eval()
            #     # model = model.to(device)
            #     model, device = load_esrgan_model()
            #     print(f"Using device: {device}")



            #     MAX_PIXELS = 600 * 600  # Adjust as needed
            #     # MAX_PIXELS = 1280 * 720  # Adjust as needed
            #     # if image.shape[0] * image.shape[1] > MAX_PIXELS:
            #     #     scale = np.sqrt(MAX_PIXELS / (image.shape[0] * image.shape[1]))
            #     #     new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            #     #     print(f"Resizing image to {new_size} for memory efficiency.")
            #     #     img = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            #     if current_image.shape[0] * current_image.shape[1] > MAX_PIXELS:
            #         scale = np.sqrt(MAX_PIXELS / (current_image.shape[0] * current_image.shape[1]))
            #         new_size = (int(current_image.shape[1] * scale), int(current_image.shape[0] * scale))
            #         print(f"Resizing image to {new_size} for memory efficiency.")
            #         img = cv2.resize(current_image, new_size, interpolation=cv2.INTER_CUBIC)
            #     else:
            #         img = current_image


            #     img = img.astype(np.float32) / 255.0

            #     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            #     img_LR = img.unsqueeze(0).to(device)

            #     # Inference
            #     with torch.no_grad():
            #         output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

            #     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            #     output = (output * 255.0).round().astype(np.uint8)
            #     current_image = output
            elif step == 'Super Resolution':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


                model, device = load_esrgan_model()
                print(f"Using device: {device}")

                MAX_PIXELS = 600 * 600  # Adjust if needed to avoid OOM
                # MAX_PIXELS = 700 * 700  # Adjust if needed to avoid OOM


                # Resize if too large
                if current_image.shape[0] * current_image.shape[1] > MAX_PIXELS:
                    scale = np.sqrt(MAX_PIXELS / (current_image.shape[0] * current_image.shape[1]))
                    new_size = (int(current_image.shape[1] * scale), int(current_image.shape[0] * scale))
                    print(f"Resizing image to {new_size} for memory efficiency.")
                    img = cv2.resize(current_image, new_size, interpolation=cv2.INTER_CUBIC)
                else:
                    img = current_image

                def apply_esrgan_once(img_np, model, device):
                    img = img_np.astype(np.float32) / 255.0
                    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
                    img_LR = img.unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                    return (output * 255.0).round().astype(np.uint8)

                # First 4× pass
                output = apply_esrgan_once(img, model, device)

                # Second 4× pass = 16× total
                # output = apply_esrgan_once(output, model, device)

                current_image = output






        # Show original and enhanced images
        c1, c2 = st.columns(2)

        # Display download original imagebutton
        buffered = BytesIO()
        pil_image = Image.fromarray(image_np)
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        # Download button
        c1.download_button(
            label="Download Original Image",
            data=buffered.getvalue(),
            file_name="enhanced_image.jpg",
            mime="image/jpeg",key="download-button-1"
        )

        # Display download enhanced imagebutton
        buffered = BytesIO()
        pil_image = Image.fromarray(current_image)
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        # Download button
        c2.download_button(
            label="Download Enhanced Image",
            data=buffered.getvalue(),
            file_name="enhanced_image.jpg",
            mime="image/jpeg",key="download-button-2"
        )

        c1.subheader("Original Image")
        c2.subheader("Final Enhanced Image")
        with c1:
            display_magnifier(image_np, "Original_Image")
            # display_magnifier(image_np)

        with c2:
            display_magnifier(current_image, "Enhanced_Image")





