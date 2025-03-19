import torch
import numpy as np
import cv2
import duckdb
import open_clip
import pyarrow as pa
import pandas as pd
import time
import os
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def log_time(func):
    """Decorator to log execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

# Load FastSAM Model
fastsam_model = YOLO("C:/fast/FastSAM-s.pt")

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
clip_model.to(device)

# Connect to DuckDB
db = duckdb.connect("image_embeddings.duckdb")
table_name = "image_embeddings"

# Create table if not exists
db.execute(f"""
    CREATE TABLE image_embeddings_new AS 
    SELECT image, segment_id, embedding, NULL::INTEGER AS x_min, NULL::INTEGER AS y_min, 
           NULL::INTEGER AS x_max, NULL::INTEGER AS y_max 
    FROM image_embeddings
""")
db.execute("DROP TABLE image_embeddings")
db.execute("ALTER TABLE image_embeddings_new RENAME TO image_embeddings")

MAX_SEGMENTS = 50  
EMBEDDING_DIM = 512  

@log_time
def segment_image(image_path):
    """Segment an image using FastSAM and return segment paths along with bounding box coordinates."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")

    img = cv2.resize(img, (1024, 1024))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = fastsam_model(img_rgb, retina_masks=True, device=device)
    masks = results[0].masks

    if masks is None or not hasattr(masks, "data"):
        raise ValueError("FastSAM output does not contain valid masks")

    masks_np = masks.data.cpu().numpy()
    num_masks = min(masks_np.shape[0], MAX_SEGMENTS)

    # Create a directory to store segments
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    segment_dir = f"segments-imaged/{image_name}/"
    os.makedirs(segment_dir, exist_ok=True)

    segmented_data = []
    for idx in range(num_masks):
        mask = (masks_np[idx] * 255).astype(np.uint8)

        # Get non-zero pixel coordinates (bounding box)
        y_coords, x_coords = np.nonzero(mask)
        if len(x_coords) == 0 or len(y_coords) == 0:
            continue  # Skip empty masks

        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        # Crop the masked region
        cropped_img = img[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # Apply mask to the cropped region
        masked_cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)

        segment_path = os.path.join(segment_dir, f"{image_name}_segment{idx:03d}.png")
        cv2.imwrite(segment_path, masked_cropped_img)

        # Store segment info including bounding box coordinates
        segmented_data.append((f"segment{idx:03d}", segment_path, x_min, y_min, x_max, y_max))

    return segmented_data

@log_time
def compute_clip_embedding(image_path):
    """Compute CLIP embedding for a given image segment."""
    image_pil = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy().flatten()

    if np.isnan(embedding).any():
        raise ValueError("NaN detected in CLIP embedding")

    if embedding.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}, expected {EMBEDDING_DIM}")
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

@log_time
def store_embeddings(image_path, embeddings, db):
    """Store computed embeddings and bounding box coordinates in DuckDB."""
    if not embeddings:
        print(f"No embeddings found for {image_path}. Skipping...")
        return

    try:
        data_to_insert = [
            (
                image_path, 
                seg_id, 
                np.array(embedding, dtype=np.float32).tobytes(),
                int(x_min),  # Convert NumPy int to Python int
                int(y_min),  
                int(x_max),  
                int(y_max)  
            )
            for seg_id, embedding, x_min, y_min, x_max, y_max in embeddings
        ]
        db.executemany(
            f"INSERT INTO {table_name} (image, segment_id, embedding, x_min, y_min, x_max, y_max) VALUES (?, ?, ?, ?, ?, ?, ?)",
            data_to_insert
        )
    except Exception as e:
        print(f"Unexpected error in store_embeddings: {e}")

@log_time
def process_and_store(image_path, db):
    """Segment an image, compute embeddings, and store them along with bounding box coordinates."""
    embeddings = []
    
    for segment_id, segment_path, x_min, y_min, x_max, y_max in segment_image(image_path):  
        try:
            embedding = compute_clip_embedding(segment_path)
            embeddings.append((segment_id, embedding.tolist(), x_min, y_min, x_max, y_max))
        except Exception as e:
            print(f"Skipping segment {segment_id} due to error: {e}")
            continue

    if not embeddings:
        print(f"No valid embeddings for {image_path}. Skipping...")
        return

    store_embeddings(image_path, embeddings, db)

@log_time
def clean_database():
    """Clean the database by removing invalid embeddings."""
    print("Cleaning database...")
    db.execute(f"DELETE FROM {table_name} WHERE OCTET_LENGTH(embedding) != {EMBEDDING_DIM * 4}")
    print("Database cleaned.")

@log_time
def compare_images(image1, image2, threshold=0.8, save_path="comparison_results.csv"):
    """Compare embeddings of two images and find similar segments."""
    try:
        df = db.query(f"SELECT * FROM {table_name}").df()
        df1 = df[df["image"] == image1]
        df2 = df[df["image"] == image2]

        if df1.empty or df2.empty:
            raise ValueError("One of the images has no stored embeddings")

        embeddings1 = [np.frombuffer(e, dtype=np.float32) for e in df1["embedding"]]
        embeddings2 = [np.frombuffer(e, dtype=np.float32) for e in df2["embedding"]]

        unique_shapes1 = {e.shape for e in embeddings1}
        unique_shapes2 = {e.shape for e in embeddings2}

        print(f"Unique embedding shapes in {image1}: {unique_shapes1}")
        print(f"Unique embedding shapes in {image2}: {unique_shapes2}")

        if len(unique_shapes1) > 1 or len(unique_shapes2) > 1:
            raise ValueError("Inconsistent embedding shapes found. Check data integrity.")

        embeddings1 = np.stack(embeddings1)
        embeddings2 = np.stack(embeddings2)

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

        results_df = pd.DataFrame(similar_pairs, columns=["Image1", "Segment1", "Image2", "Segment2", "Similarity"])
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
        return similar_pairs

    except Exception as e:
        print(f"Error in compare_images: {e}")

if __name__ == "__main__":
    clean_database()
    process_and_store("IMG_0292.jpeg", db)
    process_and_store("IMG_0293.jpeg", db)
    compare_images("IMG_0292.jpeg", "IMG_0293.jpeg", threshold=0.8)
