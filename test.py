import os

EXCLUDED_DIRS = {'node_modules', '__pycache__', '.git'}

def list_directory_contents(path, indent=0):
    try:
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                if item in EXCLUDED_DIRS:
                    continue
                print('  ' * indent + '|-- ' + item + '/')
                list_directory_contents(full_path, indent + 1)
            else:
                print('  ' * indent + '|-- ' + item)
    except PermissionError:
        print('  ' * indent + '|-- [Permission Denied]')

# Replace with your target directory or use '.' for current
start_directory = '.'
list_directory_contents(start_directory)
