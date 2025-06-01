import os
import ast
import networkx as nx
import matplotlib.pyplot as plt

# List only Python files in the main project folder (ignore subdirectories)
def find_python_files(root_dir):
    py_files = []
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if os.path.isfile(path) and entry.endswith('.py'):
            py_files.append(path)
    return py_files

# Parse imports from a single file
def get_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=file_path)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split('.')[0] + '.py')
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split('.')[0] + '.py')
    return imports

# Build a directed graph of file dependencies
def build_graph(py_files):
    g = nx.DiGraph()
    # Map basename to full path
    file_map = {os.path.basename(p): p for p in py_files}
    # Add nodes
    for p in py_files:
        g.add_node(os.path.basename(p))
    # Add edges for each import that matches a project file
    for p in py_files:
        src = os.path.basename(p)
        for imp in get_imports(p):
            if imp in file_map:
                g.add_edge(src, imp)
    return g

# Plot and save the graph
def plot_graph(g):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(g, k=0.5, seed=42)
    nx.draw_networkx_nodes(g, pos, node_size=1500)
    nx.draw_networkx_edges(g, pos, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_labels(g, pos, font_size=8)
    plt.title('Project File Dependency Graph')
    plt.axis('off')
    plt.tight_layout()
    # Save as PNG in project root
    plt.savefig('pipeline_graph.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    files = find_python_files(root)
    graph = build_graph(files)
    plot_graph(graph)
