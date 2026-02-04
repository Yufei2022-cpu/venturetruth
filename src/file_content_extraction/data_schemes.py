from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class FileObject:
    """
    Dataclass to represent a single file's JSON structure.
    """

    filename: str
    extension: str
    type: str  # As requested, e.g., "document"
    content: str  # From PyPDF2
    content_ocr: str  # From Tesseract (OCR)

@dataclass
class MetadataFile:
    """
    Dataclass for the main JSON entity, combining
    CSV metadata with the corresponding file object(s).
    """

    metadata: Dict[str, Any]
    files: List[FileObject]