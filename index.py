import ast
import sys
import os
import chromadb
from chromadb.utils import embedding_functions

# Папка передаётся аргументом: python index.py <папка>
PATH_TO_PY_FILES = sys.argv[1] if len(sys.argv) > 1 else "dataset_case3_v1.0_fix/codebase_python/gymhero/gymhero"

files_path = []
chunks = []
chunks_names = []
chunks_paths = []
chunks_lines = []  # НОВОЕ: номера строк

current_py_file_path = None


def is_python_file(file_name):
    return file_name.endswith(".py")


def searching_py_files(path):
    for file_name in os.listdir("."):
        if is_python_file(file_name):
            files_path.append(f"{path}/{file_name}")
        elif os.path.isdir(file_name):  # ИСПРАВЛЕНО: раньше было "." not in file_name
            os.chdir(file_name)
            searching_py_files(f"{path}/{file_name}")
            os.chdir("..")


class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.current_class = None  # НОВОЕ: отслеживаем в каком классе мы сейчас

    def visit_ClassDef(self, node):
        chunks.append(ast.unparse(node))
        chunks_names.append(node.name)
        chunks_paths.append(current_py_file_path)
        chunks_lines.append(node.lineno)  # НОВОЕ: сохраняем номер строки

        # Заходим в класс — запоминаем его имя
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        chunks.append(ast.unparse(node))
        # ИСПРАВЛЕНО: если внутри класса — пишем ClassName.method_name
        if self.current_class:
            chunks_names.append(f"{self.current_class}.{node.name}")
        else:
            chunks_names.append(node.name)
        chunks_paths.append(current_py_file_path)
        chunks_lines.append(node.lineno)  # НОВОЕ

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)


def make_chunks():
    global current_py_file_path
    for file_path in files_path:
        # ИСПРАВЛЕНО: путь теперь gymhero/api/routes/auth.py вместо ./api/routes/auth.py
        current_py_file_path = "gymhero/" + file_path[2:]
        with open(file_path, "r") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        visitor = FunctionCallVisitor()
        visitor.visit(tree)


CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "code_embs"

client = None
collection = None


def create_db():
    global client, collection
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    # ИСПРАВЛЕНО: удаляем старую коллекцию если есть, чтобы можно было переиндексировать
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks_to_db():
    global client, collection
    collection.add(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))],
        metadatas=[{
            "name": chunks_names[i],
            "path": chunks_paths[i],
            "lineno": chunks_lines[i],  # НОВОЕ
        } for i in range(len(chunks))],
    )


def init():
    os.chdir(PATH_TO_PY_FILES)
    searching_py_files(".")
    make_chunks()
    os.chdir("../../../../")
    create_db()
    add_chunks_to_db()
    print(f"Готово: проиндексировано {len(chunks)} фрагментов")  # ИСПРАВЛЕНО: убрали весь debug-вывод


init()
