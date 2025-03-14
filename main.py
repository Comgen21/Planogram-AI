import os
import sys
import torch
import numpy as np
import logging
import lancedb
import pyarrow as pa
import open_clip
from torch.serialization import add_safe_globals
from torch.nn import Sequential
from PIL import Image
import torchvision.transforms as transforms
from fastsam import FastSAM, FastSAMPrompt
from sklearn.metrics.pairwise import cosine_similarity

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Allow PyTorch to load specific classes safely
add_safe_globals([Sequential])

# Add FastSAM to system path
sys.path.append("C:/fast/FastSAM")  # Adjust this path

# Load FastSAM Model
try:
    model = FastSAM("C:/fast/FastSAM-s.pt")  # Ensure correct model path
    logging.info("FastSAM model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading FastSAM model: {e}")
    sys.exit(1)

# Load CLIP Model
try:
    clip_model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip_model.eval()
    logging.info("CLIP model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading CLIP model: {e}")
    sys.exit(1)

def load_image(image_path):
    if not os.path.exists(image_path):
        logging.error(f"File not found: {image_path}")
        raise FileNotFoundError(f"Image path not found: {image_path}")
    
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((1024, 1024))
        return img_resized
    except PermissionError:
        logging.error(f"Permission denied for image: {image_path}")
        raise PermissionError(f"Cannot access file. Try running as administrator or changing file permissions.")
    except Exception as e:
        logging.error(f"Unexpected error loading image: {e}")
        raise e

# Load Query Image
IMAGE_PATH = "C:/fast/IMG_0292.jpeg"
image = load_image(IMAGE_PATH)# Improve the code by adding error handling and logging for the similarity search function
def similarity_search(query_image, top_k=5):
    try:
        query_embedding = extract_clip_embedding(query_image)
        results = table.search(query_embedding).limit(top_k).to_pandas()
        logging.info(f"Similarity search completed successfully. Found {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        return None

# Improve the code by adding error handling and logging for the compare_with_folder function
def compare_with_folder(query_image_path, target_folder):
    try:
        query_image = load_image(query_image_path)
        query_embedding = extract_clip_embedding(query_image)

        results = []
        for file in os.listdir(target_folder):
            target_image_path = os.path.join(target_folder, file)
            if os.path.isfile(target_image_path):
                try:
                    target_image = load_image(target_image_path)
                    target_embedding = extract_clip_embedding(target_image)
                    
                    # Compute cosine similarity
                    similarity = cosine_similarity([query_embedding], [target_embedding])[0][0]
                    similarity_percentage = round(similarity * 100, 2)  # Convert to percentage
                    
                    results.append((file, similarity_percentage))
                except Exception as e:
                    logging.error(f"Error processing image {file}: {e}")
        
        # Sort results by similarity score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        logging.info(f"Multi-image comparison completed successfully. Found {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"Error during multi-image comparison: {e}")
        return None
DEVICE = "cpu"

# Run FastSAM inference
try:
    results = model(image, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    logging.info("Inference completed successfully.")
except Exception as e:
    logging.error(f"Error during inference: {e}")
    sys.exit(1)

# Process results
prompt_processor = FastSAMPrompt(image, results, device=DEVICE)
annotations = prompt_processor.everything_prompt()

# Save segmentation results
output_path = "C:/fast/output/result.jpg"
prompt_processor.plot(annotations=annotations, output_path=output_path)
logging.info(f"Segmentation results saved at {output_path}")

# Convert images to CLIP embeddings
def extract_clip_embedding(image):
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
    return embedding.numpy().flatten()

# Convert segmentation masks to embeddings
segmentation_embeddings = []
for mask in annotations:
    mask_numpy = mask.cpu().detach().numpy().astype(np.uint8)
    mask_image = Image.fromarray(mask_numpy)
    embedding = extract_clip_embedding(mask_image)
    segmentation_embeddings.append(embedding)

logging.info("Segmentation embeddings extracted.")

# Connect to LanceDB
DB_PATH = "C:/fast/segmentation_db"
db = lancedb.connect(DB_PATH)

# Check embedding details
print("Type of segmentation_embeddings:", type(segmentation_embeddings))
if isinstance(segmentation_embeddings, list):
    print("Number of embeddings:", len(segmentation_embeddings))
    if len(segmentation_embeddings) > 0:
        print("Type of first embedding:", type(segmentation_embeddings[0]))
        print("Shape of first embedding:", segmentation_embeddings[0].shape if hasattr(segmentation_embeddings[0], 'shape') else "No shape attribute")
    else:
        print("segmentation_embeddings is empty!")
elif isinstance(segmentation_embeddings, np.ndarray):
    print("Shape of segmentation_embeddings:", segmentation_embeddings.shape)
else:
    print("Unexpected type:", type(segmentation_embeddings))

embedding_dim = segmentation_embeddings[0].shape[0]  # Assuming same embedding size per segment
schema = pa.schema([
    ("image_id", pa.int64()),  # Unique image ID
    ("segment_id", pa.int64()),  # Unique segment ID within an image
    ("embedding", pa.list_(pa.float32(), embedding_dim))  # Fixed 512-d embeddings per segment
])

# Create a table (if not exists)
try:
    table = db.create_table(
        "embeddings",
        schema=schema,  
        mode="overwrite"
    )
    logging.info("LanceDB table created successfully.")
except Exception as e:
    logging.error(f"Error creating LanceDB table: {e}")
    sys.exit(1)

# Insert embeddings
try:
    data = [
        {"image_id": 1, "segment_id": i, "embedding": emb.tolist()}
        for i, emb in enumerate(segmentation_embeddings)
    ]
    table.add(data)
    logging.info(f"{len(data)} embeddings stored in LanceDB.")
except Exception as e:
    logging.error(f"Error storing embeddings: {e}")
    sys.exit(1)

# Similarity Search Function
def similarity_search(query_image, top_k=5):
    query_embedding = extract_clip_embedding(query_image)
    results = table.search(query_embedding).limit(top_k).to_pandas()
    return results

# Function to compare query image against all images in a folder
def compare_with_folder(query_image_path, target_folder):
    query_image = load_image(query_image_path)
    query_embedding = extract_clip_embedding(query_image)

    results = []
    for file in os.listdir(target_folder):
        target_image_path = os.path.join(target_folder, file)
        if os.path.isfile(target_image_path):
            target_image = load_image(target_image_path)
            target_embedding = extract_clip_embedding(target_image)
            
            # Compute cosine similarity
            similarity = cosine_similarity([query_embedding], [target_embedding])[0][0]
            similarity_percentage = round(similarity * 100, 2)  # Convert to percentage
            
            results.append((file, similarity_percentage))

    # Sort results by similarity score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    return results

# Example Similarity Search
query_image = load_image("C:/fast/IMG_0293.jpeg")
similar_results = similarity_search(query_image)

# Print results
logging.info("Similarity search results:")
logging.info(similar_results)

# Compare against multiple target images in a folder
TARGET_FOLDER = "C:/fast/Imagez"
similarity_results = compare_with_folder(IMAGE_PATH, TARGET_FOLDER)

# Print multi-image comparison results
logging.info("Multi-image similarity scores:")
for file, score in similarity_results:
    logging.info(f"{file}: {score}% similarity")
