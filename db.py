import lancedb
import pyarrow as pa
import numpy as np
import torch

# Example: Your extracted segmentation embeddings (Replace with actual data)
segmentation_embeddings = torch.rand(300, 512)  # Example: (300 objects, 512-dim embeddings)

# Convert PyTorch tensor to NumPy
embeddings_numpy = segmentation_embeddings.cpu().detach().numpy()

# Convert embeddings into a list of dictionaries
data = [
    {"id": i, "embedding": embeddings_numpy[i].tolist()} for i in range(len(embeddings_numpy))
]

# Define correct PyArrow schema
schema = pa.schema([
    ("id", pa.int64()),
    ("embedding", pa.list_(pa.float32(), len(embeddings_numpy[0])))
])

# Convert data into PyArrow Table
table_data = pa.Table.from_pylist(data, schema=schema)

# Initialize LanceDB
db = lancedb.connect("C:/fast/lancedb")  # Adjust path accordingly
table_name = "segmentation_db"

# Create or open table correctly
if table_name in db.table_names():
    table = db.open_table(table_name)
    table.add(table_data)  # ✅ Insert data correctly
else:
    table = db.create_table(table_name, schema=schema, data=table_data)  # ✅ Now passing data

print(f"✅ Successfully stored {len(embeddings_numpy)} segmentation embeddings in LanceDB.")
