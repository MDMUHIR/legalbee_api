from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Filter
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# সব points delete করুন
client.delete(
    collection_name="bangladesh_laws",
    points_selector=models.FilterSelector(filter=Filter())
)

# কতটা বাকি আছে দেখুন
info = client.get_collection("bangladesh_laws")
print(f"✅ Done! Points remaining: {info.points_count}")