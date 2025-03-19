import torch
import numpy as np
import cv2
import duckdb
import open_clip
import pyarrow as pa
import pandas as pd
import time
import os
import faiss  # 🔥 New import
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim

# --------------------------- Logging Decorator ----------------------------
def log_time(func):
    """Decorator to log execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

# --------------------------- Initialization -------------------------------
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
    CREATE TABLE IF NOT EXISTS {table_name} (
        image TEXT,
        segment_id TEXT,
        embedding BLOB,
        x_min INTEGER,
        y_min INTEGER,
        x_max INTEGER,
        y_max INTEGER
    )
""")

MAX_SEGMENTS = 50  
EMBEDDING_DIM = 512  

# --------------------------- Segmentation ----------------------------
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

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    segment_dir = f"segments-imaged/{image_name}/"
    os.makedirs(segment_dir, exist_ok=True)

    segmented_data = []
    for idx in range(num_masks):
        mask = (masks_np[idx] * 255).astype(np.uint8)

        y_coords, x_coords = np.nonzero(mask)
        if len(x_coords) == 0 or len(y_coords) == 0:
            continue

        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        cropped_img = img[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        masked_cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)

        segment_path = os.path.join(segment_dir, f"{image_name}_segment{idx:03d}.png")
        cv2.imwrite(segment_path, masked_cropped_img)

        segmented_data.append((f"segment{idx:03d}", segment_path, x_min, y_min, x_max, y_max))

    return segmented_data

# --------------------------- CLIP Embedding ----------------------------
@log_time
def compute_clip_embedding(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy().flatten()

    if np.isnan(embedding).any():
        raise ValueError("NaN detected in CLIP embedding")

    embedding = embedding / np.linalg.norm(embedding)
    return embedding

# --------------------------- Store Embeddings ----------------------------
@log_time
def store_embeddings(image_path, embeddings, db):
    if not embeddings:
        print(f"No embeddings found for {image_path}. Skipping...")
        return

    try:
        db.executemany(
            f"INSERT INTO {table_name} (image, segment_id, embedding, x_min, y_min, x_max, y_max) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(image_path, seg_id, np.array(embedding, dtype=np.float32).tobytes(), int(x_min), int(y_min), int(x_max), int(y_max))
             for seg_id, embedding, x_min, y_min, x_max, y_max in embeddings]
        )
    except Exception as e:
        print(f"Unexpected error in store_embeddings: {e}")

@log_time
def process_and_store(image_path, db):
    embeddings = []
    for segment_id, segment_path, x_min, y_min, x_max, y_max in segment_image(image_path):
        try:
            embedding = compute_clip_embedding(segment_path)
            embeddings.append((segment_id, embedding.tolist(), x_min, y_min, x_max, y_max))
        except Exception as e:
            print(f"Skipping segment {segment_id} due to error: {e}")

    if embeddings:
        store_embeddings(image_path, embeddings, db)

# --------------------------- Clean Database ----------------------------
@log_time
def clean_database():
    print("Cleaning database...")
    db.execute(f"DELETE FROM {table_name} WHERE OCTET_LENGTH(embedding) != {EMBEDDING_DIM * 4}")
    print("Database cleaned.")

# --------------------------- FAISS Functions ----------------------------
@log_time
def build_faiss_index(db, index_path="faiss.index", metric="cosine"):
    """Build and save a FAISS index from DuckDB embeddings."""
    df = db.query(f"SELECT rowid, embedding FROM {table_name}").df()
    ids = df["rowid"].values.astype(np.int64)
    embeddings = np.stack([np.frombuffer(e, dtype=np.float32) for e in df["embedding"]])

    if metric == "cosine":
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        faiss.normalize_L2(embeddings)
    else:  # L2
        index = faiss.IndexFlatL2(EMBEDDING_DIM)

    index_with_ids = faiss.IndexIDMap(index)
    index_with_ids.add_with_ids(embeddings, ids)

    faiss.write_index(index_with_ids, index_path)
    print(f"FAISS index built and saved at {index_path}")
    return index_with_ids

@log_time
def load_faiss_index(index_path):
    """Load a FAISS index from disk."""
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from {index_path}")
    return index

@log_time
def search_faiss(index, query_embedding, db, top_k=5):
    """Search the FAISS index with a query embedding and fetch metadata."""
    query = query_embedding.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query)

    distances, ids = index.search(query, top_k)
    ids = ids.flatten()

    if len(ids) == 0:
        print("No matches found.")
        return []

    query_result = db.query(f"SELECT * FROM {table_name} WHERE rowid IN ({','.join(map(str, ids))})").df()
    query_result["faiss_score"] = distances.flatten()
    print(query_result)
    return query_result
@log_time
def compare_images(image1, image2, cosine_thresh=0.99, mahal_thresh=1.0, save_path="strictcomparison_results.csv", save_all=True):
    """
    Compares segments of two images using cosine similarity and Mahalanobis distance.
    
    Args:
        image1 (str): First image filename.
        image2 (str): Second image filename.
        cosine_thresh (float): Cosine similarity threshold. Only pairs >= this value will be included.
        mahal_thresh (float): Mahalanobis distance threshold. Only pairs <= this value will be included.
        save_path (str): Output CSV file path.
        save_all (bool): If True, saves all pairs regardless of threshold filters.
    """
    try:
        print(f"Running comparison between {image1} and {image2} with thresholds:")
        print(f" - Cosine Similarity >= {cosine_thresh}")
        print(f" - Mahalanobis Distance <= {mahal_thresh}")

        df = db.query(f"SELECT * FROM {table_name}").df()
        print(f"Fetched {len(df)} total records from the DB.")

        df1 = df[df["image"] == image1]
        df2 = df[df["image"] == image2]

        print(f"{image1}: {len(df1)} segments found.")
        print(f"{image2}: {len(df2)} segments found.")

        if df1.empty or df2.empty:
            raise ValueError("One of the images has no stored embeddings")

        embeddings1 = np.stack([np.frombuffer(e, dtype=np.float32) for e in df1["embedding"]])
        embeddings2 = np.stack([np.frombuffer(e, dtype=np.float32) for e in df2["embedding"]])

        # Cosine similarity matrix
        cosine_matrix = cosine_similarity(embeddings1, embeddings2)

        # Mahalanobis distance matrix (with regularization)
        cov_matrix = np.cov(np.vstack((embeddings1, embeddings2)).T)
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # regularization to avoid singularity
        inv_cov = np.linalg.inv(cov_matrix)

        mahal_matrix = np.array([
            [mahalanobis(e1, e2, inv_cov) for e2 in embeddings2]
            for e1 in embeddings1
        ])

        # Collect pairs according to thresholds
        selected_pairs = []
        for i in range(len(df1)):
            for j in range(len(df2)):
                cosine_score = cosine_matrix[i, j]
                mahal_score = mahal_matrix[i, j]

                # Apply thresholds
                if cosine_score >= cosine_thresh and mahal_score <= mahal_thresh:
                    selected_pairs.append((
                        image1, df1["segment_id"].values[i],
                        image2, df2["segment_id"].values[j],
                        float(cosine_score), float(mahal_score)
                    ))

        # If no pairs pass thresholds
        if not selected_pairs:
            print("No pairs met the threshold criteria!")

        # If saving all comparisons regardless of thresholds
        if save_all:
            all_pairs = []
            for i in range(len(df1)):
                for j in range(len(df2)):
                    cosine_score = cosine_matrix[i, j]
                    mahal_score = mahal_matrix[i, j]

                    all_pairs.append((
                        image1, df1["segment_id"].values[i],
                        image2, df2["segment_id"].values[j],
                        float(cosine_score), float(mahal_score)
                    ))

            results_df = pd.DataFrame(all_pairs, columns=["Image1", "Segment1", "Image2", "Segment2", "CosineSim", "MahalanobisDist"])
            print(f"Saved ALL comparisons ({len(all_pairs)} pairs) to {save_path}")

        else:
            results_df = pd.DataFrame(selected_pairs, columns=["Image1", "Segment1", "Image2", "Segment2", "CosineSim", "MahalanobisDist"])
            print(f"Saved {len(selected_pairs)} filtered comparisons to {save_path}")

        # Save to CSV
        results_df.to_csv(save_path, index=False)
        print(f"Comparison results saved to {save_path}")

        return results_df

    except Exception as e:
        print(f"Error in compare_images: {e}")

# --------------------------- Example Pipeline ----------------------------
if __name__ == "__main__":
    clean_database()

    # Step 1: Process & store embeddings
    process_and_store("IMG_0292.jpeg", db)
    process_and_store("IMG_0293.jpeg", db)

    # Step 2: Build FAISS index from stored embeddings
    index = build_faiss_index(db)

    # Step 3: Query the index with a new segment/image embedding
    query_embedding = compute_clip_embedding("segments-imaged/IMG_0292/IMG_0292_segment000.png")
    search_faiss(index, query_embedding, db, top_k=5)
    compare_images("IMG_0292.jpeg", "IMG_0293.jpeg", cosine_thresh=0.99, mahal_thresh=1.0, save_path="all_results.csv", save_all=True)

    # Optional: Save the index for reuse
    faiss.write_index(index, "saved_faiss.index")
