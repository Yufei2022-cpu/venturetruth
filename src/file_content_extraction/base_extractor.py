from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """
    Abstract Base Class for extraction strategies.
    """
    
    @abstractmethod
    def extract(self, file_path: str) -> str:
        pass