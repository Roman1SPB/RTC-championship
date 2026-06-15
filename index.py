import ast
import os 

PATH_TO_PY_FILES = "dataset_case3_v1.0_fix/codebase_python/gymhero/gymhero"
files_path = []
chunks = []

def is_python_file(file_name):
    return ".py" in file_name 

def is_directory(file_name):
    return "." not in file_name

def searching_files(path):
    current_files = os.listdir(".")
    for file_name in current_files:
        if is_python_file(file_name):
            files_path.append(f'{path}/{file_name}')
        if is_directory(file_name):
            os.chdir(file_name)
            searching_files(f'{path}/{file_name}')
    if (path != '.'):
        os.chdir("..")

# https://earthly.dev/blog/python-ast/
class FunctionCallVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        chunks.append((node.name, ast.dump(node)))
        self.generic_visit(node)
    def visit_ClassDef(self, node):
        chunks.append((node.name, ast.dump(node)))
        self.generic_visit(node)
    def visit_AsyncFunctionDef(self, node):
        chunks.append((node.name, ast.dump(node)))
        self.generic_visit(node)


def chunking():
    for file_path in files_path:
        file = open(file_path, 'r')
        code = ""
        for line in file:
            code += line 
        tree = ast.parse(code)
        visitor = FunctionCallVisitor()
        visitor.visit(tree) 
        file.close()
        
def init():     
    os.chdir(PATH_TO_PY_FILES)
    searching_files('.')
    chunking()
    for el in chunks:
        print(el[0], el[1])
        print()
        print()

init()