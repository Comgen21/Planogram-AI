import torch
import numpy as np
import cv2
import lancedb
import open_clip
import pyarrow as pa
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load FastSAM Model
fastsam_model = YOLO("C:/fast/FastSAM-s.pt")

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
clip_model.to(device)

# Connect to LanceDB
db = lancedb.connect("lancedb_dir")
table_name = "image_embeddings"

if table_name not in db.table_names():
    table = db.create_table(
        table_name,
        schema=[
            ("image", pa.string()),
            ("segment_id", pa.string()),
            ("embedding", pa.list_(pa.float32()))
        ],
        mode="overwrite"
    )
else:
    table = db.open_table(table_name)

MAX_SEGMENTS = 50  

def segment_image(image):
    img = cv2.imread(image)
    if img is None:
        raise ValueError(f"Error loading image: {image}")

    img = cv2.resize(img, (1024, 1024))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = fastsam_model(img_rgb, retina_masks=True, device=device)
    masks = results[0].masks

    if masks is None or not hasattr(masks, "data"):
        raise ValueError("FastSAM output does not contain valid masks")

    masks_np = masks.data.cpu().numpy()
    num_masks = min(masks_np.shape[0], MAX_SEGMENTS)

    return [(f"segment{idx:03d}", masks_np[idx].astype(np.uint8) * 255) for idx in range(num_masks)]
import cv2
import os

def save_colored_segments(image_path):
    """Save colored masks for visualization."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error loading image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_resized = cv2.resize(img, (1024, 1024))  # Resize to match FastSAM mask

        segments = segment_image(image_path)

        os.makedirs("colored_segments", exist_ok=True)

        for segment_id, mask in segments:
            color_mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
            color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8)  # Random color
            color_mask[mask > 0] = color  # Apply color only to mask regions
            
            # Overlay mask on resized image
            overlay = cv2.addWeighted(img_resized, 0.7, color_mask, 0.3, 0)

            save_path = os.path.join("colored_segments", f"{os.path.basename(image_path)}_{segment_id}.png")
            Image.fromarray(overlay).save(save_path)
        print(f"Running save_colored_segments for {image_path}")
    except Exception as e:
        print(f"Error in save_colored_segments: {e}")

def compute_clip_embedding(image_np):
    image_pil = Image.fromarray(image_np)
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy()

    if np.isnan(embedding).any():
        raise ValueError("NaN detected in CLIP embedding")

    return embedding.flatten()

def store_embeddings(image_path, embeddings, table):
    if not embeddings:
        print(f"No embeddings found for {image_path}. Skipping...")
        return

    try:
        formatted_data = {
            "image": [str(image_path)] * len(embeddings),
            "segment_id": [str(seg_id) for seg_id, _ in embeddings],
            "embedding": [embedding if isinstance(embedding, list) else embedding.flatten().tolist() for _, embedding in embeddings]
        }

        table_data = pa.table(formatted_data)
        table.add(table_data)
    
    except MemoryError:
        print(f"Not enough memory for {image_path}. Reducing batch size...")
    except Exception as e:
        print(f"Unexpected error in store_embeddings: {e}")

# 7️⃣ Process Image (Segment + Embed + Store)
def process_and_store(image, table):
    """Process image: segment, save colored masks, compute embeddings, and store in DB."""
    embeddings = []
    
    # Save colored segments
    save_colored_segments(image)

    for segment_id, mask in segment_image(image):  
        embedding = compute_clip_embedding(mask)
        if embedding is None:
            continue
        
        # Convert list<float> to NumPy array
        embedding_np = np.array(embedding, dtype=np.float32).flatten()

        embeddings.append((segment_id, embedding_np.tolist()))

    if not embeddings:
        print(f"No valid embeddings for {image}. Skipping...")
        return

    # Store embeddings in LanceDB
    store_embeddings(image, embeddings, table)


import pandas as pd

def compare_images(image1, image2, threshold=0.8, save_path="comparison_results.csv", save_format="csv"):
    try:
        df = table.to_pandas()

        df1 = df[df["image"] == image1]
        df2 = df[df["image"] == image2]

        if df1.empty or df2.empty:
            raise ValueError("One of the images has no stored embeddings")

        embeddings1 = np.stack(df1["embedding"].values)
        embeddings2 = np.stack(df2["embedding"].values)

        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        similar_pairs = [
            (image1, df1["segment_id"].values[i], image2, df2["segment_id"].values[j], similarity_matrix[i, j])
            for i in range(len(df1))
            for j in range(len(df2))
            if similarity_matrix[i, j] >= threshold
        ]

        if not similar_pairs:
            print("No similar segments found above the threshold.")
            return []

        # Convert to DataFrame
        results_df = pd.DataFrame(similar_pairs, columns=["Image1", "Segment1", "Image2", "Segment2", "Similarity"])

        # Save to CSV or Excel
        if save_format.lower() == "csv":
            results_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        elif save_format.lower() == "excel":
            excel_path = save_path.replace(".csv", ".xlsx")
            results_df.to_excel(excel_path, index=False)
            print(f"Results saved to {excel_path}")
        else:
            print("Unsupported format. Choose 'csv' or 'excel'.")

        return similar_pairs

    except Exception as e:
        print(f"Error in compare_images: {e}")


if __name__ == "__main__":
    image1 = "IMG_0292.jpeg"
    image2 = "IMG_0293.jpeg"

    process_and_store(image1, table)
    process_and_store(image2, table)

    compare_images(image1, image2, threshold=0.8)
