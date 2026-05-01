import time
import os
import subprocess
from joblib import Parallel, delayed
import sys

from commons import Simulation, Helpers
from commons.datafile import DataFile

# ========== Start timing ==========
start_time = time.time()

# ========== Get filetag ==========
filetag = input("Enter filetag: ")

# ========== Simulation parameters ==========
number_of_cores = 8 # You can set the number of cores to use for parallel processing. Set to -1 to use all available cores.
number_of_simulations = 16 # This is set to 16 for testing purposes. The paper uses 1000 simulations.
number_of_generations = 25
number_of_cohorts = 10
write_agent_data = False # The default is to write community level datae. Set to true if you want to look deeper into agent data but it # will take longer to run and generate larger files. The agent data will be written to raw/data folder with the simulation name you provide.


# ========== Set up data folder ==========
data_folder = os.path.join("data", "raw", filetag)
if write_agent_data:
    if not os.path.exists(data_folder):
        print(f"Creating folder: {data_folder}")
        os.makedirs(data_folder)

# ========== Current timestamp for filenames ==========
current_time = Helpers.datetime_now()
file_name_base = f"{filetag}_{current_time}"

# ========== Define Simulation function ==========
def simulation_function(i):
    file_name = f"{file_name_base}_{i}.csv"
    file_path = os.path.join(data_folder, file_name)

    fileobj = DataFile(file_path, columns=Helpers.get_columns())

    if write_agent_data:
        fileobj.write_header()

    print("Simulation number:", i)
    s = Simulation(
        sim_no=i,
        num_coms=7,
        com_size=160,
        fileobj=fileobj,
    )
    s.run_sim(num_gen=number_of_generations, num_cohorts=number_of_cohorts, write_agent_data=write_agent_data)

# ========== Run simulations in parallel ==========
Parallel(n_jobs=-1)(
    delayed(simulation_function)(i) for i in range(number_of_simulations)
)

# ========== Final timing ==========
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
