import time
import os
import subprocess
from joblib import Parallel, delayed
import sys

from commons import Simulation, Helpers
from commons.datafile import DataFile

# ========== 1️⃣ Start timing ==========
start_time = time.time()

# ========== 2️⃣ Get filetag ==========
filetag = input("Enter filetag: ")

# ========== 5️⃣ Simulation parameters ==========
number_of_cores = 8
number_of_simulations = 8 # 1000 in paper
number_of_generations = 15
number_of_cohorts = 10
write_agent_data = False # agent level data, larger files, not needed for main results


# ========== 3️⃣ Set up data folder ==========
data_folder = os.path.join("data", "raw", filetag)
if write_agent_data:
    if not os.path.exists(data_folder):
        print(f"Creating folder: {data_folder}")
        os.makedirs(data_folder)

# ========== 4️⃣ Current timestamp for filenames ==========
current_time = Helpers.datetime_now()
file_name_base = f"{filetag}_{current_time}"

# ========== 6️⃣ Simulation function ==========
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


# ========== 7️⃣ Run simulations in parallel ==========
Parallel(n_jobs=-1)(
    delayed(simulation_function)(i) for i in range(number_of_simulations)
)

# ========== 9️⃣ Final timing ==========
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
