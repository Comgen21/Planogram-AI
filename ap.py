import torch
import numpy as np
import cv2
import duck
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

# Connect to DuckDB
db = duck.connect("image_embeddings.duckdb")
table_name = "image_embeddings"

# Create table if not exists
db.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        image TEXT,
        segment_id TEXT,
        embedding BLOB
    )
""")

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

def compute_clip_embedding(image_np):
    image_pil = Image.fromarray(image_np)
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy().flatten()

    if np.isnan(embedding).any():
        raise ValueError("NaN detected in CLIP embedding")

    expected_dim = 512  # Check if this is the correct embedding size for your model
    if embedding.shape[0] != expected_dim:
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}, expected {expected_dim}")

    return embedding


def store_embeddings(image_path, embeddings, db):
    if not embeddings:
        print(f"No embeddings found for {image_path}. Skipping...")
        return

    try:
        data_to_insert = [(image_path, seg_id, np.array(embedding, dtype=np.float32).tobytes()) for seg_id, embedding in embeddings]
        db.executemany(f"INSERT INTO {table_name} (image, segment_id, embedding) VALUES (?, ?, ?)", data_to_insert)

    except MemoryError:
        print(f"Not enough memory for {image_path}. Reducing batch size...")
    except Exception as e:
        print(f"Unexpected error in store_embeddings: {e}")

def process_and_store(image, db):
    embeddings = []
    
    for segment_id, mask in segment_image(image):  
        embedding = compute_clip_embedding(mask)
        if embedding is None:
            continue
        
        embedding_np = np.array(embedding, dtype=np.float32).flatten()
        embeddings.append((segment_id, embedding_np.tolist()))

    if not embeddings:
        print(f"No valid embeddings for {image}. Skipping...")
        return

    store_embeddings(image, embeddings, db)

def compare_images(image1, image2, threshold=0.8, save_path="comparison_results.csv", save_format="csv"):
    try:
        df = db.query(f"SELECT * FROM {table_name}").df()

        df1 = df[df["image"] == image1]
        df2 = df[df["image"] == image2]

        if df1.empty or df2.empty:
            raise ValueError("One of the images has no stored embeddings")

        # Convert stored byte embeddings back to numpy arrays
        embeddings1 = [np.frombuffer(e, dtype=np.float32) for e in df1["embedding"]]
        embeddings2 = [np.frombuffer(e, dtype=np.float32) for e in df2["embedding"]]

        # Print unique shapes
        print("Unique embedding shapes in image1:", {e.shape for e in embeddings1})
        print("Unique embedding shapes in image2:", {e.shape for e in embeddings2})

        # Ensure uniform shape before stacking
        if len(set(e.shape for e in embeddings1)) > 1 or len(set(e.shape for e in embeddings2)) > 1:
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

    process_and_store(image1, db)
    process_and_store(image2, db)

    compare_images(image1, image2, threshold=0.8)
