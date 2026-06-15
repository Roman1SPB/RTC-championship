import ast
import os 
import chromadb 
from chromadb.utils import embedding_functions

PATH_TO_PY_FILES = "dataset_case3_v1.0_fix/codebase_python/gymhero/gymhero"
files_path = []
chunks = []
chunks_names = []

def is_python_file(file_name):
    return ".py" in file_name 

def is_directory(file_name):
    return "." not in file_name

def searching_py_files(path):
    current_files = os.listdir(".")
    for file_name in current_files:
        if is_python_file(file_name):
            files_path.append(f'{path}/{file_name}')
        if is_directory(file_name):
            os.chdir(file_name)
            searching_py_files(f'{path}/{file_name}')
    if (path != '.'):
        os.chdir("..")

# https://earthly.dev/blog/python-ast/
class FunctionCallVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        chunks.append(ast.dump(node))
        chunks_names.append(node.name)
        self.generic_visit(node)
    def visit_ClassDef(self, node):
        chunks.append(ast.dump(node))
        chunks_names.append(node.name)
        self.generic_visit(node)
    def visit_AsyncFunctionDef(self, node):
        chunks.append(ast.dump(node))
        chunks_names.append(node.name)
        self.generic_visit(node)
    # def visit_Lambda(self, node):
    #     chunks.append((node.name, ast.dump(node)))
    #     self.generic_visit(node)

def make_chunks():
    for file_path in files_path:
        file = open(file_path, 'r')
        code = ""
        for line in file:
            code += line 
        tree = ast.parse(code)
        visitor = FunctionCallVisitor()
        visitor.visit(tree) 
        file.close()

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "demo_docs"

client = None
collection = None
def create_db():
    global client, collection
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(\
        model_name=EMBED_MODEL)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
        get_or_create=True,
    )

def add_chunks_to_db():
    global client, collection
    collection.add(
        documents=chunks,
        ids=[f"id{i}" for i in range(len(chunks))],
        metadatas=[{"name": name} for name in chunks_names]
    )

def print_query_example():
    global client, collection 
    query_results = collection.query(
        query_texts=["how does the project load configuration depending on the runtime environment?"],
        n_results=3,
    )

    for query in query_results["documents"][0]:
        print(query)
        print()
        print()

    print(query_results["distances"])

def init():     
    os.chdir(PATH_TO_PY_FILES)
    searching_py_files('.')
    make_chunks()
    os.chdir("../../../../")
    create_db()
    add_chunks_to_db()
    print_query_example()

init()