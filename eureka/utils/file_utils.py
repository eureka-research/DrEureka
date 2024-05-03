import os
import importlib.util

def find_files_with_substring(directory, substring):
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matches.append(os.path.join(root, file))
    return matches

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function