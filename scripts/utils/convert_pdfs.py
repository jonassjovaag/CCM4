#!/usr/bin/env python3
"""
PDF to Text Converter and File Organizer
Converts PDFs to text and organizes them into folders
"""

import os
import sys
from pathlib import Path

def convert_pdf_to_text(pdf_path: Path, txt_path: Path) -> bool:
    """Convert a PDF file to text using available method"""
    try:
        # Try using pdftotext (command line tool, most reliable)
        import subprocess
        result = subprocess.run(
            ['pdftotext', str(pdf_path), str(txt_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    try:
        # Try PyPDF2
        import PyPDF2
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write('\n\n'.join(text))
            return True
    except (ImportError, Exception) as e:
        print(f"   PyPDF2 failed: {e}")
    
    try:
        # Try pdfplumber
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = []
            for page in pdf.pages:
                text.append(page.extract_text())
            
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write('\n\n'.join(text))
            return True
    except (ImportError, Exception) as e:
        print(f"   pdfplumber failed: {e}")
    
    print(f"   ❌ All conversion methods failed")
    return False


def main():
    # Set up paths
    refs_dir = Path(__file__).parent / "References"
    converted_dir = refs_dir / "converted PDFs"
    txt_dir = refs_dir / "txt"
    
    # Create directories if they don't exist
    converted_dir.mkdir(exist_ok=True)
    txt_dir.mkdir(exist_ok=True)
    
    # Find unconverted PDFs in References folder
    unconverted = []
    for pdf_file in refs_dir.glob("*.pdf"):
        if pdf_file.is_file():
            unconverted.append(pdf_file)
    
    if not unconverted:
        print("✅ No unconverted PDFs found in References folder")
        return
    
    print(f"Found {len(unconverted)} PDF(s) to convert:\n")
    
    for pdf_path in unconverted:
        print(f"📄 Converting: {pdf_path.name}")
        
        # Define output paths
        txt_path = txt_dir / f"{pdf_path.stem}.txt"
        moved_pdf_path = converted_dir / pdf_path.name
        
        # Convert
        if convert_pdf_to_text(pdf_path, txt_path):
            print(f"   ✅ Converted to: {txt_path.relative_to(refs_dir)}")
            
            # Move original PDF to converted folder
            pdf_path.rename(moved_pdf_path)
            print(f"   📦 Moved to: {moved_pdf_path.relative_to(refs_dir)}")
        else:
            print(f"   ❌ Conversion failed")
        
        print()
    
    print("✅ Done!")


if __name__ == "__main__":
    main()

