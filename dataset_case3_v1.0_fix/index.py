"""
CodeLens RAG — индексатор кодовой базы.

Запуск:
    python index.py gymhero                       # только Python
    python index.py gymhero qrcode-generator-master   # Python + Java

Стратегия чанкинга (обоснование — в README):
    1 чанк = функция / класс / метод (ast.iter_child_nodes).
    - метод сохраняется как ClassName.method (различаем одноимённые функции/методы);
    - класс индексируется "карточкой": сигнатура, docstring, поля и сигнатуры
      методов БЕЗ их тел — чтобы не дублировать код и не размывать эмбеддинг;
    - в текст эмбеддинга добавляются путь к файлу, имя сущности и ДЕКОРАТОРЫ
      (для FastAPI-маршрутов это HTTP-метод и URL — сильный сигнал для NL-запросов).
    chunk_id строго: {relative_path}:{name}:{start_line}  (start_line = строка def/class).
"""

import ast
import json
import sys
import zipfile
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

try:
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser
    _JAVA_OK = True
except Exception:
    _JAVA_OK = False

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "code_embs"
# если папки нет на диске — распакуем соответствующий архив
ZIP_MAP = {"gymhero": "codebase_python.zip", "qrcode-generator-master": "codebase_java.zip"}


def _src_lines(node, lines, with_decorators=True):
    start = node.lineno
    if with_decorators and getattr(node, "decorator_list", None):
        start = min(start, node.decorator_list[0].lineno)
    return "\n".join(lines[start - 1:node.end_lineno])


def _class_card(node, lines) -> str:
    """Класс без тел методов: заголовок + docstring + поля + сигнатуры методов."""
    bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
    parts = [f"class {node.name}({bases}):" if bases else f"class {node.name}:"]
    doc = ast.get_docstring(node)
    if doc:
        parts.append(f'"""{doc}"""')
    for item in node.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):          # поля (модели/схемы)
            seg = ast.get_source_segment("\n".join(lines), item)
            if seg:
                parts.append(seg)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):  # только сигнатура
            sig_start = item.decorator_list[0].lineno if item.decorator_list else item.lineno
            sig_end = item.body[0].lineno - 1 if item.body else item.lineno
            parts.append("\n".join(lines[sig_start - 1:sig_end]))
    return "\n".join(parts)


def extract_python_chunks(py_file: Path, repo_root: Path):
    rel = py_file.relative_to(repo_root).as_posix()
    src = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    out = []

    def emit(name, kind, code, lineno):
        header = f"# file: {rel}\n# {kind}: {name}\n"
        out.append({
            "chunk_id": f"{rel}:{name}:{lineno}",
            "document": header + code,
            "path": rel, "name": name, "kind": kind, "start_line": lineno,
        })

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            emit(node.name, "class", _class_card(node, lines), node.lineno)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    emit(f"{node.name}.{item.name}", "method",
                         _src_lines(item, lines), item.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node.name, "function", _src_lines(node, lines), node.lineno)
    return out


def extract_java_chunks(java_file: Path, repo_root: Path):
    if not _JAVA_OK:
        return []
    rel = java_file.relative_to(repo_root).as_posix()
    b = java_file.read_text(encoding="utf-8", errors="replace").encode("utf-8")
    parser = Parser(Language(tsjava.language()))
    tree = parser.parse(b)
    TARGETS = {"class_declaration", "interface_declaration",
               "method_declaration", "constructor_declaration"}
    out, cur, done = [], tree.walk(), False
    while not done:
        n = cur.node
        if n.grammar_name in TARGETS:
            nm = n.child_by_field_name("name")
            if nm:
                name = b[nm.start_byte:nm.end_byte].decode("utf-8", "replace")
                code = b[n.start_byte:n.end_byte].decode("utf-8", "replace")
                line = n.start_point[0] + 1
                out.append({
                    "chunk_id": f"{rel}:{name}:{line}",
                    "document": f"# file: {rel}\n# {n.grammar_name}: {name}\n{code}",
                    "path": rel, "name": name, "kind": n.grammar_name, "start_line": line,
                })
        if cur.goto_first_child():
            continue
        if cur.goto_next_sibling():
            continue
        while True:
            if not cur.goto_parent():
                done = True
                break
            if cur.goto_next_sibling():
                break
    return out


def collect_chunks(folders):
    chunks = []
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            zip_name = ZIP_MAP.get(root.name)
            if zip_name and Path(zip_name).exists():
                print(f"Распаковываю {zip_name} -> {root.name}/")
                with zipfile.ZipFile(zip_name) as z:
                    z.extractall(".")
        if not root.exists():
            print(f"Папка не найдена, пропуск: {root}")
            continue
        print(f"Индексирую: {root}")
        for py in sorted(root.rglob("*.py")):
            chunks.extend(extract_python_chunks(py, root))
        for jv in sorted(root.rglob("*.java")):
            chunks.extend(extract_java_chunks(jv, root))
    return chunks


def main():
    folders = sys.argv[1:]
    if not folders:
        if not Path("gymhero").exists() and Path("codebase_python.zip").exists():
            with zipfile.ZipFile("codebase_python.zip") as z:
                z.extractall(".")
        folders = ["gymhero"]

    chunks = collect_chunks(folders)
    if not chunks:
        sys.exit("Не найдено ни одного чанка.")
    print(f"Извлечено чанков: {len(chunks)}")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode([c["document"] for c in chunks],
                              show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME,
                                           metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{
            "chunk_id": c["chunk_id"], "info": c["chunk_id"],  
            "path": c["path"], "name": c["name"], "kind": c["kind"],
            "start_line": c["start_line"],
        } for c in chunks],
    )
    print(f"Сохранено в ChromaDB: {collection.count()} чанков -> {CHROMA_DATA_PATH}")

    if Path("eval_questions.json").exists():
        questions = json.loads(Path("eval_questions.json").read_text(encoding="utf-8"))
        results = []
        for q in questions:
            qv = model.encode(q["query"], normalize_embeddings=True).tolist()
            r = collection.query(query_embeddings=[qv], n_results=5)
            results.append({"question_id": q["question_id"],
                            "top_5_chunks": [m["chunk_id"] for m in r["metadatas"][0]]})
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        print("results.json сохранён. Проверка: "
              "python score.py --predictions results.json --questions eval_questions.json")


if __name__ == "__main__":
    main()
