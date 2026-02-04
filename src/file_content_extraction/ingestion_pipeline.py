import json
import os
import csv

from typing import List, Dict
from dataclasses import asdict

from file_content_extraction.pdf_processor import PDFProcessor
from file_content_extraction.data_schemes import MetadataFile

class IngestionPipeline:
    """
    Orchestrates the loading of CSV metadata and processing of documents.
    """
    
    def __init__(self, csv_path: str, pdf_folder: str, output_path: str, limit: int = 10):
        self.csv_path = csv_path
        self.pdf_folder = pdf_folder
        self.output_path = output_path
        self.pdf_processor = PDFProcessor(pdf_folder)
        self.limit = limit
        self.results: List[Dict] = []
    
    def _parse_filenames(self, filenames_str: str) -> List[str]:
        """
        Helper function to safely parse the JSON list of filenames from the CSV.

        Args:
            filenames_str (str): _description_

        Returns:
            List[str]: _description_
        """
        if not filenames_str:
            return []
        
        try:
            names = json.loads(filenames_str)
            if isinstance(names, list):
                return [f"{name}.pdf" if not name.endswith(".pdf") else name for name in names]
        except json.JSONDecodeError:
            print(f"Warning: Could not process filename JSON: {filenames_str}")
        return []
    
    def run(self):
        """
        Executes the full extraction pipeline
        """
        
        print(f"---Starting Ingestion Pipeline ---")
        print(f"Source CSV: {self.csv_path}")
        
        if not os.path.exists(self.csv_path):
            print("Error: CSV file not found")
            return
        
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print("Error: Empty CSV")
                return
            
            filename_col = reader.fieldnames[0]
            
            for row_idx, row in enumerate(reader):
                if row_idx >= self.limit:
                    break
                
                print(f"Processing Row {row_idx+1}...")
                
                # 1. Parse file names
                pdf_filenames = self._parse_filenames(row.get(filename_col))
                
                processed_files = []
                
                # 2. Process each file
                for pdf_name in pdf_filenames:
                    file_obj = self.pdf_processor.process_file(pdf_name)
                    processed_files.append(file_obj)
                    
                # 3. Create Metadata object
                
                meta_obj = MetadataFile(metadata=row, files=processed_files)
                self.results.append(asdict(meta_obj))
                
        self.save_results()
        
    def save_results(self):
        """
        Writes the accumulated results into a JSON
        """
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4)
            print(f"Saved the processed records to: {self.output_path}")
        except IOError as e:
            print(f"Error saving JSON: {e}")