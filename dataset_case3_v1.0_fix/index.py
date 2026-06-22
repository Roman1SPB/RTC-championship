import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer 
import zipfile
import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

with zipfile.ZipFile("codebase_python.zip", "r") as z:
    z.extractall(".")

def extract_chunks(py_file: Path, repo_root: Path):
    """Extract chunk_ids from a Python file using AST."""
    rel = py_file.relative_to(repo_root).as_posix()
    src = py_file.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            chunk_id = f"{rel}:{node.name}:{node.lineno}"
            chunks.append(chunk_id)
            # Methods inside the class
            for item in ast.walk(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = f"{rel}:{node.name}.{item.name}:{item.lineno}"
                    chunks.append(method_id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level functions (check not inside a class)
            chunk_id = f"{rel}:{node.name}:{node.lineno}"
            chunks.append(chunk_id)
    
    return chunks


model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # пример — выберите модель самостоятельно

repo_root = Path("gymhero")
index = {}  # chunk_id -> embedding
code_chunks = []

for py_file in repo_root.rglob("*.py"):
    rel = py_file.relative_to(repo_root).as_posix()
    src = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    
    lines = src.splitlines()
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Определите имя chunk
            name = node.name
            start = node.lineno - 1
            end = node.end_lineno
            chunk_text = "\n".join(lines[start:end])
            chunk_id = f"{rel}:{name}:{node.lineno}"
            code_chunks.append(chunk_text)
            embedding = model.encode(chunk_text)
            index[chunk_id] = embedding

print(f"Indexed {len(index)} chunks")

questions = json.loads(Path("eval_questions.json").read_text(encoding="utf-8"))
results = []

for q in questions:
    query_embedding = model.encode(q["query"])
    
    # Найти топ-5 по косинусному сходству
    scores = {}
    for chunk_id, emb in index.items():
        similarity = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-9
        )
        scores[chunk_id] = similarity
    
    top5 = sorted(scores, key=scores.get, reverse=True)[:5]
    results.append({"question_id": q["question_id"], "top_5_chunks": top5})

# Сохранить результаты
Path("results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("results.json saved")

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "code_embs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
  
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    documents=code_chunks,
    embeddings=list(index.values()),
    metadatas=[{"info": key} for key in index.keys()],
    ids=[f"id{i}" for i in range(len(index))],
)
