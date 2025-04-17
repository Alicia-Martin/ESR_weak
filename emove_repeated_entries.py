import os
import sys
import ast
import numpy as np
import re
# def remove_repeated_entries(file1, file2, output_file):
#     # Read the contents of both files
#     with open(file1, 'r') as f1, open(file2, 'r') as f2:
#         set1 = set(f1.readlines())  # Store lines as a set
#         set2 = set(f2.readlines())
#         print(len(set1))
#         print(len(set2))

#     # Find non-repeated entries (unique to each file)
#     unique_entries = set1.symmetric_difference(set2)
#     print(len(unique_entries))

#     # Write unique entries to the output file
#     with open(output_file, 'w') as f_out:
        # f_out.writelines(unique_entries)

def process_files(file1, file2,  output_file):
    # Read entries from both files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        set1 = set(f1.readlines())
        set2 = set(f2.readlines())
        #take out \n form set 2
        # set2 = {line.strip() for line in set2}
        # set2 = {line.strip() for line in set2}
        print(len(set1))
        print(len(set2))

    # Read entries from other files
    # set3 = set()
    # for file in os.listdir(other_files):
    #     if file.endswith('.txt') and file.startswith('results'):
    #         with open(os.path.join(other_files, file), 'r') as f:
    #             # print(file)
            
    #             for line in f:
    #                 line = re.sub(r"Array\(([^,)]*), dtype=[^)]*\)", r"\1", line)
    #                 line = re.sub(r"array\(([^)]*)\)", r"\1", line)  # Remove array([...])
    #                 line = re.sub(r'\binf\b', "0", line)  
    #                 # print(line)

    #                 # match = re.match(r"\[.*?,\s*'([^']+)'", line)
    #                 # match = re.match(r"\[.*?,\s*'([^']+)'(?:,\s*(.*))?\]", line)

    #                 # data = ast.literal_eval(match.group(0))

    #                 data = ast.literal_eval(line)

    #                 function = data[1]  # Extract function (second element)
    #                 target_number = data[-2]  # Extract second-to-last number
    #                 target_array = data[-1]  # Extract last array
    #                 target_array = tuple(target_array) # Convert to numpy array
    #                 # print(target_array)

    #                 formatted_entry = f"{function}|{target_array}|{target_number}"
    #                 # print(formatted_entry)
    #                 set3.add(str(formatted_entry))

    # print(len(set3&set2))
    # print(len(set3))

    # functions_left_to_do = set2 - set3
    # print(len(functions_left_to_do))
    # print(functions_left_to_do)

    # with open(output_file, 'w') as out:
    #     for entry in sorted(functions_left_to_do):  
    #         print(entry)
    #         out.write(f"{entry}\n")
    # print(len(repeated_entries))
    # print(len(unique_entries))
    
    # Write the unique entries ensuring repeated ones appear only once
    unique_entries = set1 | set2
    print(len(unique_entries))
    with open(output_file, 'w') as out:
        for entry in sorted(unique_entries):  
            out.write(entry)

    # #divide it in five files now
    # num_files = 5
    # output_dir = 'esr/fitting/output_decreasing/output/combining_clusters/'
    # # Divide las funciones en partes iguales
    # chunks = [[] for _ in range(num_files)]
    # for i, entry in enumerate(unique_entries):
    #     # print(item)
    #     # chunks[i % num_files].append(item)
    #     chunks[i % num_files].append(entry.strip())

    # # Save each chunk into a separate file
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i + 1}: {len(chunk)} entries")
    #     filename = f'final_global_local_funcs_part_{i + 1}.txt'
    #     file_path = os.path.join(output_dir, filename)
    #     with open(file_path, 'w') as f:
    #         for entry in chunk:
    #             f.write(f"{entry}\n")  # Write each entry in the chunk

    # # Guarda cada chunk en un archivo separado
    # for i, chunk in enumerate(chunks):
    #     print('chunk', chunk, len(chunk))
    #     filename = f'final_global_local_funcs_part_{i + 1}.txt'
    #     file_path = os.path.join(output_dir, filename)
    #     with open(file_path, 'w') as f:
    #         for func, final_indices, complexity in chunk:
    #             f.write(f'{func}|{str(final_indices)}|{complexity}\n')
    

# Example usage
file1 = '/Users/aliciamartin/Downloads/global_local_funcs_katz.txt'
file2 = '/Users/aliciamartin/Downloads/global_local_funcs.txt'

# file1 = '/Users/aliciamartin/Downloads/global_local_funcs_katz.txt'
# file2 = '/Users/aliciamartin/Downloads/final_global_local_funcs_katz.txt'
# # other_files = 'esr/fitting/output_decreasing/output/combining_clusters/intermediate_results'

output_file = 'esr/fitting/output_decreasing/output/combining_clusters/global_funcs_total.txt'
# output_file = '/Users/aliciamartin/Downloads/functions_left.txt'

# remove_repeated_entries(file1, file2, output_file)
            

process_files(file1, file2, output_file)
