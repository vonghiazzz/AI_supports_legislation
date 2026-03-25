import pytesseract
from pdf2image import convert_from_path
import os
import json

# ================= CONFIGURATION =================
# Configure Tesseract path (from Miniconda)
TESSERACT_PATH = "/Users/tannghiavo/miniconda3/bin/tesseract"
# Configure Poppler path (from Miniconda bin)
POPPLER_PATH = "/Users/tannghiavo/miniconda3/bin"
# Configure PDF file path (Use absolute path for certainty)
PDF_PATH = "/Users/tannghiavo/Documents/AllProjects/AI/AI_supports_legislation/backend/dataLaw/91.signed.pdf"
# =================================================

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
print(f"Starting OCR for file: {PDF_PATH}")

try:
    # Convert pages of PDF to images (300 DPI)
    # Provide explicit poppler_path to avoid 'pdfinfo not in PATH' error
    images = convert_from_path(
        PDF_PATH, 
        first_page=1, 
        last_page=100, 
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    if images:
        print("Performing OCR on all pages...")

        result = []

        for i, img in enumerate(images):
            print(f"OCR page {i+1}/{len(images)}")
            text = pytesseract.image_to_string(img, lang='vie')

            result.append({
                "page": i+1,
                "text": text
            })

        # Save JSON file
        with open("scripts/output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("✅ Successfully saved results to output.json")

    else:
        print("Unable to convert PDF to images.")
except Exception as e:
    print(f"Error: {e}")