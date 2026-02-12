import json
import sys

def extract_code_from_ipynb(ipynb_path, output_path):
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        code_cells = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                code_cells.append(source)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n# Cell separator\n\n'.join(code_cells))
        
        print(f"Successfully extracted code to {output_path}")
    except Exception as e:
        print(f"Error extracting code: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_ipynb.py <input_ipynb> <output_py>")
    else:
        extract_code_from_ipynb(sys.argv[1], sys.argv[2])
