from file_content_extraction.base_extractor import BaseExtractor

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

class StandardPDFExtractor(BaseExtractor):
    """
    Strategy for standard text extraction using PyPDF2.
    """
    
    def extract(self, file_path):
        if not PdfReader:
            return "Error: PyPDF2 is not isntalled"
        
        try:
            reader = PdfReader(file_path)
            full_text = [page.extract_text() for page in reader.pages]
            return "\n".join(full_text)
        except Exception as e:
            return f"Error (PyPDF2): {e}"