import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

# --- PDF Extraction Library (PyPDF2) ---
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("Error: PyPDF2 library not found.")
    print("Please install it by running: pip install PyPDF2")
    exit(1)

# --- OCR & PDF-to-Image Libraries ---
# These are new for the OCR field.
# You must install them:
# pip install pytesseract pillow pdf2image
#
# AND you must install the Tesseract & Poppler system packages.
# See README.md for instructions.
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path

    OCR_ENABLED = True
except ImportError as e:
    print(f"Warning: OCR libraries not found. OCR content will be disabled. Error: {e}")
    print("To enable OCR, please run: pip install pytesseract pillow pdf2image")
    print("AND install the Tesseract & Poppler system packages (see README.md).")
    OCR_ENABLED = False


# --- Python Data Classes for JSON Structure ---


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


# --- Core Functions ---


def extract_pdf_content(pdf_path: str) -> str:
    """
    Extracts all text content from a given PDF file.

    Args:
        pdf_path: The full file path to the PDF.

    Returns:
        A string containing all extracted text, or an error message.
    """
    if not os.path.exists(pdf_path):
        return f"Error: File not found at {pdf_path}"

    try:
        reader = PdfReader(pdf_path)
        full_text = []
        for page in reader.pages:
            full_text.append(page.extract_text())

        return "\n".join(full_text)
    except Exception as e:
        return f"Error processing PDF {pdf_path} with PyPDF2: {e}"


def extract_pdf_content_ocr(pdf_path: str) -> str:
    """
    Extracts all text content from a given PDF file using OCR
    by converting each page to an image and running Tesseract.

    Args:
        pdf_path: The full file path to the PDF.

    Returns:
        A string containing all extracted OCR text, or an error message.
    """
    if not OCR_ENABLED:
        return "OCR libraries not installed. Skipping."

    if not os.path.exists(pdf_path):
        return f"Error: File not found at {pdf_path}"

    full_text = []
    try:
        # Convert PDF pages to a list of PIL Images
        images = convert_from_path(pdf_path)

        # Run OCR on each image
        for i, img in enumerate(images):
            try:
                text = pytesseract.image_to_string(img)
                full_text.append(f"--- Page {i + 1} ---\n{text}")
            except pytesseract.TesseractNotFoundError:
                return "Error: Tesseract executable not found. Please install Tesseract (see README.md)."
            except Exception as ocr_e:
                full_text.append(f"--- Error on Page {i + 1}: {ocr_e} ---")

        return "\n".join(full_text)

    except Exception as e:
        # This broad exception can catch errors from pdf2image (e.g., Poppler not found)
        return f"Error processing PDF {pdf_path} with OCR: {e}"


def process_documents(csv_path: str, pdf_folder: str, output_json_path: str):
    """
    Main function to read CSV, process PDFs, and write a final JSON.

    Args:
        csv_path: Path to the metadata.csv file.
        pdf_folder: Path to the folder containing PDF files.
        output_json_path: Path to write the final output.json file.
    """
    final_json_output = []

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            # Use DictReader to automatically get metadata as a dictionary
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                print(f"Error: CSV file '{csv_path}' is empty or has no header.")
                return

            # Get the name of the first column, which holds the filename
            filename_column = reader.fieldnames[0]
            print(
                f"Processing CSV... Using '{filename_column}' as the filename column."
            )

            for row in reader:
                filenames_string = row.get(filename_column)

                if not filenames_string:
                    print(f"Warning: Skipping row with empty filename column: {row}")
                    continue

                # --- MODIFICATION START ---
                # This row's 'files' field will be a list of FileObjects
                file_objects_list: List[FileObject] = []

                try:
                    # Parse the string representation of a list (e.g., "[\"file1\", \"file2\"]")
                    pdf_basenames = json.loads(filenames_string)
                    if not isinstance(pdf_basenames, list):
                        print(
                            f"Warning: Skipping row. Parsed data is not a list: {filenames_string}"
                        )
                        continue

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping row. Could not parse filenames JSON: {e} - Data: {filenames_string}"
                    )
                    continue

                # Loop through each base filename found in the list
                for base_filename in pdf_basenames:
                    # Add the .pdf extension as requested
                    pdf_filename = f"{base_filename}.pdf"
                    pdf_full_path = os.path.join(pdf_folder, pdf_filename)

                    # 1. Extract PDF content (Standard)
                    print(f"  - Extracting (PyPDF2): {pdf_full_path}")
                    pdf_content = extract_pdf_content(pdf_full_path)

                    # 2. Extract PDF content (OCR)
                    print(f"  - Extracting (OCR):    {pdf_full_path}")
                    ocr_content = extract_pdf_content_ocr(pdf_full_path)

                    # 3. Create the FileObject
                    file_obj = FileObject(
                        filename=pdf_filename,
                        extension=Path(pdf_filename).suffix,
                        type="document",  # As requested
                        content=pdf_content,
                        content_ocr=ocr_content,  # Add the new OCR field
                    )
                    file_objects_list.append(file_obj)

                # 4. Create the MetadataFile object
                # 'row' is already the metadata dictionary
                meta_file_obj = MetadataFile(
                    metadata=row,
                    files=file_objects_list,  # Pass the list of file objects
                )
                # --- MODIFICATION END ---

                # 5. Append the dictionary version to our final list
                final_json_output.append(asdict(meta_file_obj))

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # 5. Write the final JSON file
    try:
        with open(output_json_path, "w", encoding="utf-8") as out_file:
            json.dump(final_json_output, out_file, indent=4)
        print(f"\nSuccessfully processed all files.")
        print(f"Output JSON saved to: {output_json_path}")
    except IOError as e:
        print(f"Error writing JSON file: {e}")


# --- Main execution block ---


if __name__ == "__main__":
    # --- Configuration ---
    # !!! PLEASE UPDATE THESE PATHS !!!
    CSV_FILE_PATH = "/home/kirdmiv/code/uvc_nlp/Salesforce_Data_with_Attachment_File_Names_-_SF_DATA.csv"
    PDF_FOLDER_PATH = "/home/kirdmiv/code/uvc_nlp/sf_data_with_attachments/"
    OUTPUT_JSON_FILE = "output.json"
    # ---------------------

    # Create dummy files if they don't exist for testing
    if not os.path.exists(PDF_FOLDER_PATH):
        os.makedirs(PDF_FOLDER_PATH)
        print(f"Created dummy folder: {PDF_FOLDER_PATH}")

    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "author", "date"])
            writer.writerow(["slides_deck_01.pdf", "Alice", "2023-01-10"])
            writer.writerow(["email_printout.pdf", "Bob", "2023-02-15"])
            writer.writerow(["one_pager_project.pdf", "Charlie", "2023-03-20"])
            writer.writerow(["missing_file.pdf", "Dana", "2023-04-01"])
        print(f"Created dummy CSV: {CSV_FILE_PATH}")

        # Create dummy PDFs for testing (PyPDF2 can't create, so we just touch them)
        # Note: extract_pdf_content will return an error for these empty files,
        # which is expected behavior.
        open(os.path.join(PDF_FOLDER_PATH, "slides_deck_01.pdf"), "a").close()
        open(os.path.join(PDF_FOLDER_PATH, "email_printout.pdf"), "a").close()
        open(os.path.join(PDF_FOLDER_PATH, "one_pager_project.pdf"), "a").close()
        print(
            "Created dummy PDF files (Note: extraction will fail on these empty files)."
        )

    print("--- Starting PDF Document Processor ---")
    process_documents(CSV_FILE_PATH, PDF_FOLDER_PATH, OUTPUT_JSON_FILE)
    print("--- Process Finished ---")
