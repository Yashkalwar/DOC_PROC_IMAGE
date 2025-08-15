### **Quick notes to run:**

- Install dependencies:
    
    ```
    pip install pymupdf pdfplumber opencv-python numpy pillow pytesseract requests
    ```
    
    and install Tesseract OCR if you want OCR fallback.
    
- Example run (process the Bray PDF with all pages):
    
    ```
    python pdf_image_extractor.py --url "https://www.bray.com/docs/default-source/manuals-guides/iom-manuals/en_iom_trilok-standard0f1daacad37449d99d8661eb939b667e.pdf" --outdir ./bray_output --zoom 4
    ```
    
- Output: extracted images + rendered crops saved under `./bray_output`, with a `summary.json` describing page-wise results and captions (if detected).
