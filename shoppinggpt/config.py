# config.py
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from current directory
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths - use relative paths from current directory
DATA_PRODUCT_PATH = "data/products.csv"
DATA_TEXT_PATH = "data/policy.txt"
STORE_DIRECTORY = "data/datastore"

# Embeddings - only initialize if API key is available
EMBEDDINGS = None
if GOOGLE_API_KEY:
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")