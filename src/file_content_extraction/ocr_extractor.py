from file_content_extraction.base_extractor import BaseExtractor

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

class OCRExtractor(BaseExtractor):
    """
    Strategy for OCR Extraction using Tesseract.
    """
    
    def extract(self, file_path):
        if not OCR_AVAILABLE:
            return "OCR skipped: Libraries not found"
        
        try:
            images = convert_from_path(file_path)
            full_text = []
            
            for i, img in enumerate(images):
                try:
                    text = pytesseract.image_to_string(img)
                    full_text.append(f"--- Page {i + 1} ---\n{text}")
                except Exception as e:
                    full_text.append(f"--- Error Page {i+1}: {e} ---")
                
            return "\n".join(full_text)
        except Exception as e:
            return f"Error (OCR): {e}"