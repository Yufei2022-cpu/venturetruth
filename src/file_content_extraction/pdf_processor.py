import os
import unicodedata

from pathlib import Path

from file_content_extraction.ocr_extractor import OCRExtractor
from file_content_extraction.standard_extractor import StandardPDFExtractor
from file_content_extraction.data_schemes import FileObject

class PDFProcessor:
    """
    Handles the processing of a single PDF file.
    Composes multiple extractors to get different types of content.
    """
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.std_extractor = StandardPDFExtractor()
        self.ocr_extractor = OCRExtractor()
        
    def _find_file_robust(self, filename: str) -> str:
        """
        Attempts to find a file even if Unicode normalization (NFC/NFD) differs.
        Returns the absolute path if found, or None.
        """
        # 1. Direct check (fastest)
        direct_path = os.path.join(self.folder_path, filename)
        if os.path.exists(direct_path):
            return direct_path

        # 2. Normalize target filename in both NFC and NFD forms
        target_nfc = unicodedata.normalize('NFC', filename)
        target_nfd = unicodedata.normalize('NFD', filename)

        # 3. Iterate through directory and compare normalized names
        try:
            for actual_name in os.listdir(self.folder_path):
                actual_nfc = unicodedata.normalize('NFC', actual_name)
                actual_nfd = unicodedata.normalize('NFD', actual_name)

                # Try matching with both normalization forms
                if (actual_nfc == target_nfc or
                    actual_nfd == target_nfd or
                    actual_nfc == target_nfd or
                    actual_nfd == target_nfc):
                    return os.path.join(self.folder_path, actual_name)
        except FileNotFoundError:
            # The folder itself doesn't exist
            print(f"File not found: {filename}")
            return None

        # 4. Fallback: Try ASCII-only fuzzy matching for corrupted Unicode
        # Remove all non-ASCII characters and compare
        target_ascii = ''.join(c for c in filename if ord(c) < 128)

        try:
            for actual_name in os.listdir(self.folder_path):
                actual_ascii = ''.join(c for c in actual_name if ord(c) < 128)

                # If ASCII parts match, it's likely the same file with encoding issues
                if target_ascii == actual_ascii and target_ascii:
                    print(f"Warning: Matched '{filename}' to '{actual_name}' using ASCII fallback")
                    return os.path.join(self.folder_path, actual_name)
        except FileNotFoundError:
            pass

        return None
        
    def process_file(self, filename: str) -> FileObject:
        """
        Creates a FileObject for the given filename by running both extractors.

        Args:
            filename (str): name of the PDF file to process

        Returns:
            FileObject: parsed file content as object
        """
        full_path = self._find_file_robust(filename)
        
        if not full_path:
            error_msg = f"Error: File not found at provided {full_path}"
            return FileObject(
                filename=filename,
                extension=Path(filename).suffix,
                type="document",
                content=error_msg,
                content_ocr=error_msg
            )
            
        print(f"Processing: {filename}...")
        std_content = self.std_extractor.extract(full_path)
        ocr_content = self.ocr_extractor.extract(full_path)
        
        return FileObject(
            filename=filename,
            extension=Path(filename).suffix,
            type="document",
            content=std_content,
            content_ocr=ocr_content
        )