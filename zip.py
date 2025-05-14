import os
import sys
import zipfile

def zip_dir(source_dir: str, zip_path: str) -> None:
    """
    Recursively zip up source_dir into zip_path, emulating `zip -r`.
    - Includes hidden files
    - Follows symlinks (adds the file pointed to)
    - Preserves empty directories
    """
    source_dir = os.path.normpath(source_dir)
    base_dir   = os.path.dirname(source_dir)

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            # add files
            for fn in files:
                abs_path = os.path.join(root, fn)
                arcname  = os.path.relpath(abs_path, base_dir)
                zf.write(abs_path, arcname)

            # if this dir is empty, write a directory entry so zip -r shows it
            if not files and not dirs:
                dir_arc = os.path.relpath(root, base_dir) + '/'
                zf.writestr(dir_arc, '')

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <directory_to_zip> <output_zip_file>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    if not os.path.isdir(src):
        print(f"Error: “{src}” is not a directory.")
        sys.exit(1)

    # ensure the parent of dst exists
    parent = os.path.dirname(os.path.abspath(dst))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    zip_dir(src, dst)
    print(f"✓ Created {dst}")

if __name__ == "__main__":
    main()
