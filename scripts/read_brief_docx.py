from docx import Document
import os

def read_docx(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        print("\n".join(full_text))
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

if __name__ == "__main__":
    file_path = r"c:\Users\Tommaso\Documents\HOWEST\TeamProject\MTS3-MCTE-Team-Project-Energy-G1\docs\brief\Project Proposal CTAI - Energy.docx"
    print(f"--- Reading {file_path} ---")
    read_docx(file_path)
