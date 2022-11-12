current_file_path=__file__
last_slash=current_file_path.rfind("\\")
directory=current_file_path[:last_slash+1]
print(directory,end="")
