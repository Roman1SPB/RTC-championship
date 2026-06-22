import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer 
import zipfile
import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node

with zipfile.ZipFile("codebase_python.zip", "r") as z:
    z.extractall(".")

with zipfile.ZipFile("codebase_java.zip", "r") as z:
    z.extractall(".")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # пример — выберите модель самостоятельно
repo_py_root = Path("gymhero")
repo_java_root = Path("qrcode-generator-master")
index = {}
code_chunks = []

for py_file in repo_py_root.rglob("*.py"):
    rel = py_file.relative_to(repo_py_root).as_posix()
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
for java_file in repo_java_root.rglob("*.java"):
    JAVA_LANGUAGE = Language(tsjava.language())
    parser = Parser(JAVA_LANGUAGE)
    rel = java_file.relative_to(repo_java_root).as_posix()
    src = java_file.read_text(encoding="utf-8", errors="replace")
    src_bytes = src.encode('utf-8')
    
    try:
        tree = parser.parse(src_bytes)
    except SyntaxError:
        continue
    
    cursor = tree.walk()
    reached_root = False

    TARGETS = {
        "class_declaration", 
        "interface_declaration", 
        "method_declaration", 
        "constructor_declaration", 
    }

    while not reached_root:
        current_node = cursor.node
        
        if current_node.grammar_name in TARGETS:
            name_node = current_node.child_by_field_name("name")
            if name_node:
                entity_name = src_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8', errors='replace')
                chunk_text = src_bytes[current_node.start_byte:current_node.end_byte].decode('utf-8', errors='replace')
                chunk_id = f"{rel}:{entity_name}:{current_node.start_byte}"
                
                code_chunks.append(chunk_text)
                embedding = model.encode(chunk_text)
                index[chunk_id] = embedding
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        while True:
            if not cursor.goto_parent():
                reached_root = True
                break
            if cursor.goto_next_sibling():
                break

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
