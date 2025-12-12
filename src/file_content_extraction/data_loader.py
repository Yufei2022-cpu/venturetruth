import ijson

from pathlib import Path
from typing import Iterator, List

from file_content_extraction.data_schemes import MetadataFile, FileObject

class DataLoader:
    """
    A DataLoader-style class that streams data from a JSON file without loading the entire file into RAM.
    """
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        
    def __iter__(self) -> Iterator[MetadataFile]:
        """
        Generator that yields MetadataFile objects one by one
        """
        with open(self.json_path, 'rb') as f:
            for record_dict in ijson.items(f, 'item'):
                yield self._dict_to_object(record_dict)
    
    def _dict_to_object(self, data: dict):
        """
        Helper function to convert the raw dictionary back into the Dataclasses

        Args:
            data (dict): dictionary we want to convert
        """
        
        files_list = []
        raw_files = data.get('files', [])
        
        for f in raw_files:
            files_list.append(FileObject(**f))
            
        return MetadataFile(
            metadata=data.get('metadata', {}),
            files=files_list
        )
        
if __name__ == "__main__":
    
    CURRENT_FILE = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE.parent.parent.parent 

    INPUT_JSON = PROJECT_ROOT / "res/output.json"
    
    loader = DataLoader(INPUT_JSON)
    
    for i, item in enumerate(loader):
        print(f"  Item {i+1}: {item.metadata.get('Account Name', 'Unknown')}")