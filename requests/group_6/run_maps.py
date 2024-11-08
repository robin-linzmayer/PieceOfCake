import subprocess
import glob
import os

json_files = glob.glob('requests/**/*.json', recursive=True)
log_file = "requests/group_6/log.txt"
for file in json_files:
    command = [
        'python', 'main.py',  
        '-d', '5',            
        '-rq', file,   
        '-p', '6'           
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout

    completed_line_index = -1
    for i, line in enumerate(output.splitlines()):
        if "Assignment completed!" in line:
            completed_line_index = i
            break

    if completed_line_index != -1:
        relevant_output = output.splitlines()[completed_line_index + 1:]
        
        if not os.path.exists(log_file):
            with open(log_file, 'w') as log:
                log.write("Execution Log\n")
                log.write("--------------\n\n")
        
        with open(log_file, 'a') as log:
            log.write(f"File: {file}\n")
            log.write(f"{"\n".join(relevant_output)}\n\n")


