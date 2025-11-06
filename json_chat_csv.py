import os
import time
import json
import csv  # Added for CSV reading
import difflib
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
# 1. Load environment variables from .env file (if it exists)
load_dotenv()

# 2. Check for OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with OPENAI_API_KEY='your-key-here'")
    print("or set it manually by running: export OPENAI_API_KEY='your-key-here'")
    exit()

# 3. Set your CSV input configuration here
INPUT_CSV_FILE = "/home/kirdmiv/code/uvc_nlp/Salesforce_Data_with_Attachment_File_Names_-_SF_DATA.csv"
CSV_ROW_INDEX = 10  # 0 for the first row, 1 for the second, etc.
FILE_PATH_PREFIX = (
    "/home/kirdmiv/code/uvc_nlp/sf_data_with_attachments/"  # e.g., "data/reports/"
)
FILENAME_SEPARATOR = ","  # How filenames are separated in the CSV cell
# ---------------------


client = OpenAI()


def get_file_paths_from_csv(csv_path, row_index, prefix):
    """
    Reads a specific row from a CSV, parses the first column
    (which is a JSON array string like '["file1.pdf", "file2.pdf"]'),
    finds the best matching file in the prefix directory for each,
    and returns a list of full file paths.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at path: {csv_path}")
        return None

    if not os.path.isdir(prefix):
        print(f"Error: FILE_PATH_PREFIX '{prefix}' is not a valid directory.")
        return None

    try:
        # Get all available files in the target directory
        all_files_in_dir = os.listdir(prefix)
        if not all_files_in_dir:
            print(f"Warning: The directory '{prefix}' is empty.")
    except Exception as e:
        print(f"Error reading directory '{prefix}': {e}")
        return None

    try:
        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            all_rows = list(reader)

            if row_index >= len(all_rows):
                print(
                    f"Error: CSV_ROW_INDEX {row_index} is out of bounds. File only has {len(all_rows)} rows."
                )
                return None

            # Get the first column of the specified row
            filenames_str = all_rows[row_index][0]

            try:
                # Parse the string as a JSON array
                filenames_list = json.loads(filenames_str)
                if not isinstance(filenames_list, list):
                    print(
                        f"Error: CSV cell content in row {row_index} is not a valid JSON list."
                    )
                    return None
            except json.JSONDecodeError:
                print(f"Error: Failed to parse filenames string in row {row_index}.")
                print(f'Expected format: ["file1.pdf", "file2.pdf"]')
                print(f"Received: {filenames_str}")
                return None

            full_paths = []
            print(
                f"Attempting to match {len(filenames_list)} files from CSV row {row_index}:"
            )

            # For each filename from the CSV, find the best match in the directory
            for fname_from_csv in filenames_list:
                if not isinstance(fname_from_csv, str) or not fname_from_csv.strip():
                    continue

                fname_from_csv = fname_from_csv.strip()

                # Use difflib to find the best match
                # n=1: get only the single best match
                # cutoff=0.7: require a similarity score of at least 70%
                matches = difflib.get_close_matches(
                    fname_from_csv, all_files_in_dir, n=1, cutoff=0.7
                )

                if matches:
                    best_match = matches[0]
                    full_path = os.path.join(prefix, best_match)
                    full_paths.append(full_path)
                    print(f"  - Matched: '{fname_from_csv}' -> '{best_match}'")
                else:
                    print(
                        f"  - WARNING: Could not find a close match for '{fname_from_csv}' in {prefix}"
                    )

            if not full_paths:
                print(f"Error: No files were successfully matched.")
                return None

            print(f"Successfully found and matched {len(full_paths)} files:")
            for p in full_paths:
                print(f"  - {p}")
            return full_paths

    except Exception as e:
        print(f"Error reading or parsing CSV file: {e}")
        return None


def setup_vector_store(file_paths):
    """
    Uploads a list of PDF files and creates a new vector store.
    Returns the vector_store_id and a list of file objects.
    """
    file_objects = []
    file_ids = []

    # 1. Validate and Upload all files
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Error: PDF file not found at path: {path}")
            return None, None

        print(f"Uploading file: {path}...")
        try:
            with open(path, "rb") as file_stream:
                file_object = client.files.create(
                    file=file_stream,
                    purpose="assistants",  # Purpose is still 'assistants' for file_search
                )
                file_objects.append(file_object)
                file_ids.append(file_object.id)
                print(f"File uploaded with ID: {file_object.id}")
        except Exception as e:
            print(f"Error uploading file {path}: {e}")
            return None, None

    # 2. Create a Vector Store and add the files
    print("Creating Vector Store...")
    try:
        vector_store = client.vector_stores.create(
            name="PDF Q&A Store (Interactive)", file_ids=file_ids
        )
        print(f"Vector Store created with ID: {vector_store.id}")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None, file_objects

    # 3. Wait for all files to be processed
    print("Waiting for files to be processed...")
    all_files_ready = False
    while not all_files_ready:
        all_files_ready = True
        for file_id in file_ids:
            try:
                file_status = client.vector_stores.files.retrieve(
                    vector_store_id=vector_store.id, file_id=file_id
                )
                if file_status.status != "completed":
                    all_files_ready = False
                    if file_status.status in ["failed", "cancelled"]:
                        print(
                            f"File processing failed for {file_id}: {file_status.status}"
                        )
                        return None, file_objects
                    print(f"...file {file_id} is still {file_status.status}...")
            except Exception as e:
                print(f"Error checking file status for {file_id}: {e}. Retrying...")
                all_files_ready = False

        if not all_files_ready:
            time.sleep(5)

    print("All files processed successfully.")
    return vector_store, file_objects


def ask_question_in_loop(vector_store_id, question):
    """
    Asks a single question against the vector store using the Responses API
    and requests a structured JSON response.
    """

    # New prompt template to request JSON output
    json_prompt_template = f"""
Based on the provided file(s), please answer the following question:
"{question}"

Provide your response in a valid JSON format with two keys: "reasoning" and "answer".
- "reasoning": A brief explanation of how you arrived at the answer, citing the document if possible.
- "answer": The concise, direct answer to the question.

Example for "Who is the author?":
{{
  "reasoning": "The document's cover page lists 'Dr. Jane Doe' as the primary author.",
  "answer": "Dr. Jane Doe"
}}

Example for "What is the topic?":
{{
  "reasoning": "The abstract and introduction repeatedly mention 'quantum computing' and its applications.",
  "answer": "Quantum Computing"
}}

You must only output the raw JSON, starting with {{ and ending with }}.
Your JSON response:
"""

    try:
        # 4. Query using the Responses API with the new prompt
        print("\nThinking...")
        response = client.responses.create(
            model="gpt-4o",  # Using gpt-4o for better JSON adherence
            input=json_prompt_template,  # Pass the new combined prompt
            tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
        )

        # 5. Print the response
        print("--- Assistant's Response (Raw JSON) ---")
        # Correctly check if output is a list and has content
        if (
            response.output
            and isinstance(response.output, list)
            and len(response.output) > 0
            # and response.output[0].text
            # and response.output[0].text.value
        ):
            # Access the first item in the output list
            raw_response = str(response.output)  # [0].text.value
            print(raw_response)

            # Try to parse and print the JSON nicely
            try:
                # The model might include ```json ... ``` tags, though we asked it not to
                if raw_response.startswith("```json"):
                    raw_response = raw_response[7:-3].strip()

                parsed_json = json.loads(raw_response)
                print("\n--- Parsed JSON ---")
                print(f"Reasoning: {parsed_json.get('reasoning')}")
                print(f"Answer:    {parsed_json.get('answer')}")
            except json.JSONDecodeError:
                print("\n--- Warning: Could not parse response as JSON. ---")
            except Exception as e:
                print(f"\n--- Warning: Error processing JSON: {e} ---")

        else:
            print("Could not find text in the response.")
            print("Full response object:")
            print(response)
        print("------------------------------\n")

    except Exception as e:
        print(f"An unexpected error occurred while asking question: {e}")


def cleanup_resources(vector_store, file_objects):
    """
    Deletes the vector store and all uploaded files.
    """
    print("Cleaning up resources...")
    # Clean up vector store first
    if vector_store:
        print(f"Deleting vector store: {vector_store.id}")
        try:
            client.vector_stores.delete(vector_store_id=vector_store.id)
        except Exception as e:
            print(f"Could not delete vector store: {e}")

    # Now delete the files
    if file_objects:
        for file_obj in file_objects:
            print(f"Deleting file: {file_obj.id}")
            try:
                client.files.delete(file_id=file_obj.id)
            except Exception as e:
                print(f"Could not delete file {file_obj.id}: {e}")

    print("Cleanup complete.")


def main():
    # --- IMPORTANT ---
    # 1. Make sure your CSV configuration is set correctly.
    # 2. Make sure your OPENAI_API_KEY is in your .env file or environment.
    # -----------------

    if INPUT_CSV_FILE == "path/to/your/file_list.csv":
        print("=" * 50)
        print("ERROR: Please update 'INPUT_CSV_FILE' and other CSV")
        print("       variables at the top of the script.")
        print("=" * 50)
        return

    vector_store = None
    file_objects = None

    try:
        # Get file paths from CSV first
        pdf_file_paths = get_file_paths_from_csv(
            INPUT_CSV_FILE,
            CSV_ROW_INDEX,
            FILE_PATH_PREFIX,
            # Removed FILENAME_SEPARATOR
        )

        if not pdf_file_paths:
            print("Could not retrieve file paths from CSV. Exiting.")
            return

        # Setup runs once with the file list from the CSV
        vector_store, file_objects = setup_vector_store(pdf_file_paths)

        if not vector_store or not file_objects:
            print("Failed to set up vector store. Exiting.")
            return

        print("\n" + "=" * 50)
        print("Chat with your PDF(s) is ready!")
        print("Type your question and press Enter.")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 50 + "\n")

        # Interactive chat loop
        while True:
            user_question = input("You: ")
            if user_question.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break

            if not user_question.strip():
                continue

            ask_question_in_loop(vector_store.id, user_question)

    except KeyboardInterrupt:
        print("\nCaught interrupt, shutting down...")
    finally:
        # Cleanup runs when loop exits or on error
        if vector_store or file_objects:
            cleanup_resources(vector_store, file_objects)


if __name__ == "__main__":
    main()
