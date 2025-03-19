import os
import cv2
import numpy as np
import torch
import open_clip
import lancedb
import pyarrow as pa
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from PIL import Image
import faiss
import time

# Initialize models
fastsam_model = YOLO("C:/fast/FastSAM-s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
clip_model.to(device)

# Constants
EMBEDDING_DIM = 512  
MAX_SEGMENTS = 50  

# LanceDB setup
db = lancedb.connect("image_embeddings_lance")
table_name = "image_embeddings"

# Define Schema
schema = pa.schema([
    ("image", pa.string()),
    ("segment_id", pa.string()),
    ("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ("x_min", pa.int32()),
    ("y_min", pa.int32()),
    ("x_max", pa.int32()),
    ("y_max", pa.int32())
])

# Create table if not exists
if table_name not in db.table_names():
    db.create_table(table_name, schema=schema)

# Timer decorator
def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
        return result
    return wrapper

# 1. Segment image using FastSAM
@log_time
def segment_image(image_path):
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
    segment_dir = f"segments/{image_name}/"
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

# 2. Compute CLIP embedding
@log_time
def compute_clip_embedding(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = clip_model.encode_image(img_tensor).cpu().numpy().flatten()
    
    if np.isnan(embedding).any():
        raise ValueError("NaN detected in CLIP embedding")
    
    return embedding / np.linalg.norm(embedding)

# 3. Store embeddings in LanceDB
@log_time
def store_embeddings(image_path, embeddings):
    if not embeddings:
        return
    table = db.open_table(table_name)
    table.add([
        {"image": image_path, "segment_id": seg_id, "embedding": embedding.tolist(),
         "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
        for seg_id, embedding, x_min, y_min, x_max, y_max in embeddings
    ])

@log_time
def process_and_store(image_path):
    embeddings = [(seg_id, compute_clip_embedding(seg_path), x_min, y_min, x_max, y_max)
                  for seg_id, seg_path, x_min, y_min, x_max, y_max in segment_image(image_path)]
    store_embeddings(image_path, embeddings)

# 4. Compare images based on embeddings
@log_time
def compare_images(image1, image2, cosine_thresh=0.99, mahal_thresh=1.0, save_path="strictcomparison_results.csv"):

    try:
        print(f"Comparing {image1} and {image2} with thresholds:")
        print(f" - Cosine Similarity >= {cosine_thresh}")
        print(f" - Mahalanobis Distance <= {mahal_thresh}")

        # Load LanceDB table
        table = db.open_table(table_name)
        df = table.to_pandas()  # Convert to DataFrame

        df1 = df[df["image"] == image1]
        df2 = df[df["image"] == image2]

        print(f"{image1}: {len(df1)} segments found.")
        print(f"{image2}: {len(df2)} segments found.")

        if df1.empty or df2.empty:
            raise ValueError("One of the images has no stored embeddings!")

        # Extract embeddings
        embeddings1 = np.vstack(df1["embedding"].apply(lambda e: np.array(e, dtype=np.float32)))
        embeddings2 = np.vstack(df2["embedding"].apply(lambda e: np.array(e, dtype=np.float32)))

        # Compute Cosine Similarity
        cosine_matrix = cosine_similarity(embeddings1, embeddings2)
        print(f"Cosine Similarity Matrix:\n{cosine_matrix}")

        # Compute Mahalanobis Distance
        cov_matrix = np.cov(np.vstack((embeddings1, embeddings2)).T)
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # Regularization
        inv_cov = np.linalg.inv(cov_matrix)

        mahal_matrix = np.array([
            [mahalanobis(e1, e2, inv_cov) for e2 in embeddings2]
            for e1 in embeddings1
        ])
        print(f"Mahalanobis Distance Matrix:\n{mahal_matrix}")

        # Collect **ALL** similarity scores
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

        # Save results to CSV
        results_df = pd.DataFrame(all_pairs, columns=["Image1", "Segment1", "Image2", "Segment2", "CosineSim", "MahalanobisDist"])
        results_df.to_csv(save_path, index=False)
        print(f"Comparison results saved to {save_path} with {len(all_pairs)} total pairs.")

        return results_df

    except Exception as e:
        print(f"Error in compare_images: {e}")
# 5. Pipeline execution
if __name__ == "__main__":
    process_and_store("IMG_0292.jpeg")
    process_and_store("IMG_0293.jpeg")

    compare_images("IMG_0292.jpeg", "IMG_0293.jpeg", cosine_thresh=0.99, mahal_thresh=1.0, save_path="all_results.csv")

