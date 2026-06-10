# convert_to_txt.py
"""
Utility script to convert PDF or Word (.docx) files to Plain Text (.txt).

Dependencies:
  - PyMuPDF (fitz) is required for PDF conversion.
  - Standard libraries (zipfile, xml) are used for Word (.docx) to avoid external dependencies.

Usage:
  python convert_to_txt.py <input_file> [output_file]
"""

import sys
import os
import zipfile
import xml.etree.ElementTree as ET

def docx_to_text(docx_path):
    """Extracts text from a .docx file using standard zipfile and xml parsing."""
    try:
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        paragraph_tag = ns + 'p'
        text_tag = ns + 't'
        
        paragraphs = []
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            for paragraph in root.iter(paragraph_tag):
                texts = [node.text for node in paragraph.iter(text_tag) if node.text]
                paragraphs.append(''.join(texts))
        return '\n'.join(paragraphs)
    except Exception as e:
        raise RuntimeError(f"Error reading Word document: {e}")

def pdf_to_text(pdf_path):
    """Extracts text from a .pdf file using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is not installed. Please install it using: pip install pymupdf"
        )
        
    try:
        doc = fitz.open(pdf_path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return '\n'.join(text)
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_txt.py <input_file_path> [output_file_path]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: File not found at '{input_path}'")
        sys.exit(1)
        
    ext = os.path.splitext(input_path)[1].lower()
    
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = os.path.splitext(input_path)[0] + "_extracted.txt"
        
    print(f"Converting '{input_path}'...")
    
    try:
        if ext == '.docx':
            text = docx_to_text(input_path)
        elif ext == '.pdf':
            text = pdf_to_text(input_path)
        else:
            print("Error: Unsupported file format. Only .docx and .pdf are supported.")
            sys.exit(1)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        print(f"Successfully extracted text to '{output_path}'")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
