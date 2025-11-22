import os
import pickle
import argparse
import pprint

def find_and_extract_model_info(search_path: str):
    """
    Searches for 'model_meta_info.pkl' in subdirectories of a given path.
    If the .pkl file contains data, it extracts the entire content, formats it,
    and writes it to a 'model_meta_info.md' file in the same directory.
    It will skip creating a .md file if the .pkl file is empty.

    Args:
        search_path (str): The root directory to start the search from.
    """
    print(f"Starting search in: {os.path.abspath(search_path)}")
    
    # Check if the search path exists
    if not os.path.isdir(search_path):
        print(f"Error: The specified search path does not exist: {search_path}")
        return

    pkl_files_found = 0
    md_files_generated = 0

    # Walk through the directory tree
    for root, dirs, files in os.walk(search_path):
        if "model_meta_info.pkl" in files:
            pkl_files_found += 1
            model_meta_path = os.path.join(root, "model_meta_info.pkl")
            output_md_path = os.path.join(root, "model_meta_info.md")
            
            print(f"Found meta info file: {model_meta_path}")
            
            try:
                # Open and load the pickle file
                with open(model_meta_path, "rb") as f:
                    # Use a try-except block in case the pickle file is empty
                    try:
                        model_meta_info = pickle.load(f)
                    except EOFError:
                        model_meta_info = None # Handle empty file case
                
                # If the pickle file was empty or contained no data (e.g., None), skip.
                if not model_meta_info:
                    print(f"  -> Skipping MD file creation: Pickle file is empty or contains no data.")
                    continue
                
                # Get the relative path of the model's folder for cleaner output
                model_folder = os.path.relpath(root, search_path)
                
                # Pretty-format the entire dictionary content
                full_metadata = pprint.pformat(model_meta_info, indent=2)

                # Prepare the content for the individual Markdown file
                md_content = (
                    f"# Model Meta Information: {model_folder}\n\n"
                    f"**Source PKL:** `{os.path.basename(model_meta_path)}`\n\n"
                    f"**Full Metadata:**\n"
                    f"```python\n{full_metadata}\n```\n"
                )

                # Write the content to the new .md file in the same directory
                with open(output_md_path, "w", encoding="utf-8") as f_out:
                    f_out.write(md_content)
                
                print(f"  -> Generated: {output_md_path}")
                md_files_generated += 1

            except pickle.UnpicklingError:
                print(f"  [!] Warning: Could not unpickle file: {model_meta_path}")
            except Exception as e:
                print(f"  [!] Error: An unexpected error occurred while processing {model_meta_path}: {e}")

    # Final summary message
    print("\n--- Search Complete ---")
    if pkl_files_found > 0:
        print(f"Found {pkl_files_found} 'model_meta_info.pkl' file(s).")
        print(f"Generated {md_files_generated} '.md' file(s) for models with content.")
    else:
        print("No 'model_meta_info.pkl' files were found in the specified directory.")

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Find all 'model_meta_info.pkl' files in a directory, "
                    "extract all content, and generate a corresponding 'model_meta_info.md' file for each."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./checkpoint/",
        help="The root directory to search for model checkpoints. Defaults to './checkpoint/'."
    )
    
    args = parser.parse_args()

    # Run the main function
    find_and_extract_model_info(args.path)

