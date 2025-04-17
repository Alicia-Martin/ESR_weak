import sympy
import esr.generation.simplifier as simplifier
from esr.fitting.sympy_symbols import *
from mpi4py import MPI



def load_table(file_path):

    def try_convert(value):
        try:
            # Attempt to convert to a float
            return float(value)
        except ValueError:
            # Return the original value if conversion fails
            return value

    loaded_headers = []
    loaded_table_data = []

    # Read the file
    with open(file_path, "r") as file:
        # Read headers
        headers = file.readline().strip().split("\t")
        
        # Read the rest of the rows
        for line in file:
            row = line.strip().split("\t")
            row = [try_convert(element) for element in row]
            row.append("---")
            loaded_table_data.append(row)

    return headers, loaded_table_data

def save_table(file_path, headers, table_data):
    with open(file_path, "w") as file:
        # Write headers
        file.write("\t".join(headers) + "\n")
        # Write rows
        for row in table_data:
            file.write("\t".join(map(str, row)) + "\n")

def check_integrable(fcn):
    try:
        eq = sympy.sympify(fcn,
                        locals={"inv": inv,
                            "square": square,
                                "cube": cube,
                                "sqrt": sqrt,
                                "log": log,
                                "pow": pow,
                                "x": x,
                                "a0": a0,
                                "a1": a1,
                                "a2": a2})
        
    except Exception as e:
        print(fcn)
        print(e)
        return True

    tmax = 60*2
    try:
        with simplifier.time_limit(tmax):
            integral  = sympy.integrate(eq, x)

            if integral == None:
                return False
            elif integral.has(sympy.Integral):
                return False
            else:
                return integral
        
    except Exception as e:
        return "timeout"
    
def main():
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    input_file = "esr/fitting/output_decreasing/output/combining_clusters/pretty_all_clusters_comp7_katz.txt"
    output_file = "esr/fitting/output_decreasing/output/12_clusters/updated_table.dat"

    if rank == 0:
        headers, table_data = load_table(input_file)
        rows_per_process = len(table_data) // size
        # Divide work among processes
        split_data = [
            table_data[i * rows_per_process : (i + 1) * rows_per_process] for i in range(size)
        ]
        if len(table_data) % size != 0:  # Add leftovers to the last process
            split_data[-1].extend(table_data[size * rows_per_process:])
    else:
        split_data = None
        headers = None

    # Scatter data to all processes
    local_data = comm.scatter(split_data, root=0)

    # Compute integrals locally
    for row in local_data:
        function = row[1]  # Assuming the function is in the second column
        row[-1] = check_integrable(function)

    # Gather results
    gathered_data = comm.gather(local_data, root=0)

    if rank == 0:
        # Flatten the gathered data and save
        updated_table = [row for process_data in gathered_data for row in process_data]
        headers.append("Integral")  # Add header for the new column
        save_table(output_file, headers, updated_table)
        print(f"Updated table saved to {output_file}")


if __name__ == "__main__":
    main()
