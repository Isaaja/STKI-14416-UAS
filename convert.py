import nbformat
import os

def py_to_ipynb(py_file_path, ipynb_file_path=None):
    """
    Converts a .py file to a .ipynb file.

    Args:
        py_file_path (str): Path to the .py file to convert.
        ipynb_file_path (str): Path to save the .ipynb file. Defaults to None, saving as the same name as .py file.
    """
    if not os.path.exists(py_file_path):
        raise FileNotFoundError(f"The file {py_file_path} does not exist.")
    
    if not py_file_path.endswith(".py"):
        raise ValueError(f"{py_file_path} is not a valid .py file.")

    # Read the Python file content
    with open(py_file_path, 'r', encoding='utf-8') as f:
        py_content = f.readlines()

    # Create a new Jupyter notebook
    notebook = nbformat.v4.new_notebook()

    # Add each line of Python code as a cell in the notebook
    notebook.cells = [
        nbformat.v4.new_code_cell("".join(py_content))
    ]

    # Define the output .ipynb file path
    if ipynb_file_path is None:
        ipynb_file_path = py_file_path.replace('.py', '.ipynb')

    # Write the notebook to a file
    with open(ipynb_file_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)

    print(f"Successfully converted {py_file_path} to {ipynb_file_path}")


# Example usage
py_file = "./modelling/modelling.py"  # Replace with your .py file path
py_to_ipynb(py_file)
