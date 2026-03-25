import pytesseract
from pdf2image import convert_from_path
import os
from app.config import settings

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH

class OCRService:
    @staticmethod
    def extract_text(file_path: str):
        try:
            from PIL import Image
            print(f"[*] Starting OCR for: {file_path}")
            
            # Check if file is an image instead of a PDF
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                print(f"[*] Processing image file...")
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img, lang='vie')
                return f"--- Image Content ---\n{text}"
                
            # Otherwise, treat as PDF
            # Convert PDF to images
            images = convert_from_path(
                file_path, 
                dpi=300,
                poppler_path=settings.POPPLER_PATH
            )
            
            full_text = ""
            for i, img in enumerate(images):
                print(f"[*] OCR page {i+1}/{len(images)}")
                text = pytesseract.image_to_string(img, lang='vie')
                full_text += f"\n--- Page {i+1} ---\n{text}"
            
            return full_text
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            return f"Error: {e}"

ocr_service = OCRService()
