import os
import shutil
from pathlib import Path


def search_and_copy_folder():
    # Get folder name from user
    folder_name = input("Enter the folder name to search for: ")

    # Define source and target directories
    source_dir = Path("./runs_all")
    target_dir = Path("./runs_saved")

    # Create target directory if it doesn't exist
    target_dir.mkdir(exist_ok=True)

    # Search for the folder
    found = False
    for root, dirs, _ in os.walk(source_dir):
        if folder_name in dirs:
            found = True
            source_path = Path(root) / folder_name
            target_path = target_dir / folder_name

            # Copy the folder
            try:
                shutil.copytree(source_path, target_path)
                print(f"Successfully copied {folder_name} to {target_dir}")
            except FileExistsError:
                print(
                    f"Folder {folder_name} already exists in target directory")
            except Exception as e:
                print(f"Error copying folder: {e}")
            break

    if not found:
        print(f"Folder {folder_name} not found in {source_dir}")


if __name__ == "__main__":
    search_and_copy_folder()
