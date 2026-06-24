import ast
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer 
import chromadb
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
import zipfile

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # пример — выберите модель самостоятельно
index = {}
code_chunks = []

def process_python_file(py_file, repo_root):
    rel = py_file.relative_to(repo_root).as_posix()
    src = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            start = node.lineno - 1
            end = node.end_lineno
            chunk_text = "\n".join(lines[start:end])
            chunk_id = f"{rel}:{name}:{node.lineno}"
            code_chunks.append(chunk_text)
            embedding = model.encode(chunk_text)
            index[chunk_id] = embedding

def process_java_file(java_file, repo_root):
    rel = java_file.relative_to(repo_root).as_posix()
    src = java_file.read_text(encoding="utf-8", errors="replace")
    src_bytes = src.encode('utf-8')
    try:
        parser = Parser(Language(tsjava.language()))
        tree = parser.parse(src_bytes)
    except Exception:
        return
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

def index_folders(folders):
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            print(f"Папка {root} не найдена")
            continue
        print(f"Индексируем: {root}")
        for py_file in root.rglob("*.py"):
            process_python_file(py_file, root)
        for java_file in root.rglob("*.java"):
            process_java_file(java_file, root)

CHROMA_DATA_PATH = "chroma_data/"
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

with zipfile.ZipFile("codebase_java.zip", "r") as z:
    z.extractall(".")
with zipfile.ZipFile("codebase_python.zip", "r") as z:
    z.extractall(".")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Укажите хотя бы одну папку для индексации.")
        sys.exit(1)
    
    index_folders(sys.argv[1:])
    collection.add(
        documents=code_chunks,
        embeddings=list(index.values()),
        metadatas=[{"info": key} for key in index.keys()],
        ids=[f"id{i}" for i in range(len(index))],
    )
print(f"Индексация завершена, загружено {len(index)} чанков."), 