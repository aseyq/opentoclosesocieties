from commons import Simulation, Helpers
from commons.datafile import *

from joblib import Parallel, delayed

filetag = input("enter filetag: ")


data_folder = "data/simulated/" # don't forget the slash
current_time = Helpers.datetime_now()

file_name = f"{filetag}_{current_time}"


number_of_cores = 3  # number of cores to use. related to parallelization

#number_of_simulations = 1000
number_of_simulations = 3

#number_of_generations = 15
number_of_generations = 3

#number_of_cohorts = 10
number_of_cohorts = 10



def simulation_function(i):
    fileobj = DataFile(f"{data_folder}{file_name}_{i}.csv", columns = Helpers.get_columns())
    fileobj.write_header()

    print("sim number ", i)
    s = Simulation(sim_no=i,
                   num_coms=7,
                   com_size=160, 
                   init_egalitarian=False, # True or False
                   asset_inheritance_egalitarian=True, # True or False
                   fileobj=fileobj,

                   attr_min=0,
                   attr_max=1,

                   asset_min=0,
                   asset_max=1,

                   # we use same community bonus
                   # instead of cost but we keep it for 
                   # robustness checks
                   cost_min=0,
                   cost_max=0, 

                   endowment_min=0,
                   endowment_max=160,

                   utility="sqrt", # linear, sqrt
                   util_coef_attr=1,
                   util_coef_asset=1,
                   util_coef_endowment=1,
                   util_coef_cost=0,
                   same_community_bonus = 1,


                   switch_rule = "both", # to no_switch, to_patri, to_egal, both
                   switchable_communities = None, # in terms of their wealthrank. eg. (1,2,3) means that 1,2,3 are switchable 

                   switch_threshold_to_patri_certain=0.33, 
                   switch_threshold_to_patri_possible=0.93,

                   switch_threshold_to_egal_possible=0.98, 
                   switch_threshold_to_egal_certain=1.58, 

                   random_dist="beta", # beta, uniform
                   trait_inheritance = "average", # randomparent, average, 
                   parent_attribute_share=0.8,
                   membership_inheritance="single", # sticky, basic,none, single

                   proposer = "random", #men, women, random
                   topography_structure="complete", # complete, circular, manual
                   )

    s.run_sim(num_gen= number_of_generations, num_cohorts=number_of_cohorts)


Parallel(n_jobs=number_of_cores)(delayed(simulation_function)(i) for i in range(number_of_simulations))


