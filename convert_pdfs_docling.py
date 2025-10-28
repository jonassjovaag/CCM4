#!/usr/bin/env python3
"""
Docling PDF to Markdown Converter
High-quality PDF extraction for research papers using AI-powered docling
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

def convert_single_pdf(pdf_path: Path, output_dir: Path) -> Tuple[bool, str]:
    """
    Convert a single PDF to markdown using docling
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Directory for output markdown
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        from docling.document_converter import DocumentConverter
        
        # Initialize converter
        converter = DocumentConverter()
        
        # Convert PDF
        result = converter.convert(str(pdf_path))
        
        # Export as markdown
        output_path = output_dir / f"{pdf_path.stem}.md"
        markdown_content = result.document.export_to_markdown()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return True, f"✅ Saved to: {output_path.name}"
        
    except Exception as e:
        return False, f"❌ Error: {str(e)[:100]}"


def main():
    """Convert all PDFs in converted PDFs folder to markdown"""
    
    # Set up paths
    base_dir = Path(__file__).parent
    converted_pdfs_dir = base_dir / "References" / "converted PDFs"
    output_dir = base_dir / "References" / "docling_output"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Find all PDFs
    pdf_files = sorted(list(converted_pdfs_dir.glob("*.pdf")))
    
    if not pdf_files:
        print("❌ No PDF files found in References/converted PDFs/")
        return
    
    total = len(pdf_files)
    print(f"📚 Found {total} PDF files to convert\n")
    print("=" * 70)
    
    # Track results
    successful = 0
    failed = 0
    failed_files = []
    
    # Convert each PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{total}] Converting: {pdf_path.name}")
        
        success, message = convert_single_pdf(pdf_path, output_dir)
        
        if success:
            successful += 1
            print(f"    {message}")
        else:
            failed += 1
            failed_files.append(pdf_path.name)
            print(f"    {message}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"\n📊 Conversion Summary:")
    print(f"   ✅ Successful: {successful}/{total}")
    print(f"   ❌ Failed: {failed}/{total}")
    
    if failed_files:
        print(f"\n   Failed files:")
        for filename in failed_files:
            print(f"      - {filename}")
    
    print(f"\n💾 Output saved to: {output_dir}")
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()





