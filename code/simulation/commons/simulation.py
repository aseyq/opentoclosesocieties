from commons.community import Community
from commons.topography import Topography
from commons.marriagemarket import MarriageMarket
import numpy as np
import random
from commons.config import Config, global_config
import csv
import os


class Simulation(Config):
    def __init__(
        self,
        sim_no=1,
        num_coms=None,
        com_size=None,
        fileobj=None,
        communities=None,
        overrides=None,  # ⬅️ here!
    ):
        super().__init__(overrides=overrides)  # ⬅️ ensure this is here!
        init_egalitarian = self["init_egalitarian"]
        self.sim_no = sim_no
        self.communities = []

        global parameters
        parameters = self.params  # for legacy compatibility

        if communities:
            self.set_communities(communities)
        elif num_coms and com_size:
            self.create_communities(num_coms, com_size, init_egalitarian)

        if fileobj:
            self.fileobj = fileobj

    def set_communities(self, communities):
            self.communities = communities

    def create_communities(self, num_coms, com_size, is_egalitarian=True, is_switchable=True):
        # self.communities = [Community(size=com_size, is_egalitarian=is_egalitarian, is_switchable=is_switchable)
        #                      for _ in range(num_coms)]

        # if switchable communities are empty, create_egalitarian = is_egalitarian
        # if not, true

        # the idea is if there are some switchable communities, they should be created as egalitarian. later i switch non-switchable communities to patrilinear if necessary
        if self["switchable_communities"] is not None:
            create_egalitarian = True
        else:
            create_egalitarian = is_egalitarian

        self.communities = [Community(size=com_size, is_egalitarian=create_egalitarian, is_switchable=is_switchable)
                             for _ in range(num_coms)]


        sortedcoms = sorted(self.communities, key=lambda x: -x.endowment)
        for c in sortedcoms:
            c.wealthrank = sortedcoms.index(c) + 1

        if self["switchable_communities"] is not None:
            for c in self.communities:
                if c.wealthrank in self["switchable_communities"]:
                    c.is_switchable = True

                    # Switchable communities should start egalitarian
                    # for testing domino and lock in effects
                    # if not is_egalitarian:
                    #     c.switch_patrilinear()
                else:
                    c.is_switchable = False
                    if not is_egalitarian:
                        c.switch_patrilinear()

        self.set_topography(self["topography_structure"])

    def set_topography(self, structure):
        self.topography = Topography(self, structure)

    def get_people(self, order="random"):
        people = [a for c in self.communities for a in c.agents]
        if order == "random":
            random.shuffle(people)
        return people

    def get_men(self, order="random"):
        return [a for a in self.get_people(order) if a.gender == "male"]

    def get_women(self, order="random"):
        return [a for a in self.get_people(order) if a.gender == "female"]

    def create_cohorts(self, num_cohorts):
        cohorts_men = np.array_split(self.get_men(), num_cohorts)
        cohorts_women = np.array_split(self.get_women(), num_cohorts)
        return zip(cohorts_men, cohorts_women)

    def write_agent_data(self, gen_no, coh_no, round_no):
        # return
        parameter_dict = {"sim": self.sim_no, "generation": gen_no, "cohort": coh_no, "round": round_no}
        # Add all parameters with "par_" prefix
        for k, v in self.params.items():
            parameter_dict[f"par_{k}"] = v

        for a in self.get_people(order=None):
            agent_dict = a.get_info_dict()
            combined = {**parameter_dict, **agent_dict}
            # Filter out any keys not in the CSV's fieldnames
            filtered = {k: v for k, v in combined.items() if k in self.fileobj.fieldnames}
            self.fileobj.write_line(**filtered)

    def write_community_data(self, gen_no, cohort_no, round_no):
        # skip the rest
        #return
        #print(f"Writing community data for simulation {self.sim_no}, generation {gen_no}, cohort {cohort_no}, round {round_no}")

        for c in self.communities:

            community_dict = {
                "sim": self.sim_no,
                "generation": gen_no,
                "cohort": cohort_no,
                "round": round_no,
                "community_name": c.name,
                "community_wealth": c.endowment,
                "community_rank": c.wealthrank,
                "egalitarian": c.is_egalitarian,
                "endowment_pc": c.get_endowment_pc(),
                "n_residents": c.get_size(),
                "n_residents_male": c.get_male_size(),
                "n_residents_female": c.get_female_size(),
                "n_members": c.get_member_size(),
                "n_members_male": c.get_male_member_size(),
                "n_members_female": c.get_female_member_size(),
                "location_x": c.location[0],
                "location_y": c.location[1],
            }

            fieldnames = [
                "sim", "generation", "cohort", "round", "community_name", "community_wealth",
                "community_rank", "egalitarian", "endowment_pc", "n_residents",
                "n_residents_male", "n_residents_female", "n_members", "n_members_male",
                "n_members_female", "location_x", "location_y"
            ]
            # append in filename at the beginning simdata
            file_path = f"sim{self.fileobj.filename}"

            # # generate folder if it does not exist
            folder_path = os.path.dirname(file_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


            write_header = not os.path.exists(file_path)

            with open(file_path, 'a', newline='', buffering=1024*1024) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(community_dict)

    def run_sim(self, num_gen, num_cohorts, write_to_data=True, write_agent_data=False):
        if write_to_data:
            self.write_community_data(gen_no=0, cohort_no=0, round_no=0)

        if write_agent_data:
            self.write_agent_data(gen_no=0, coh_no=0, round_no=0)


        round_no = 1

        for g in range(num_gen):
            cohorts = self.create_cohorts(num_cohorts)

            for i, (men, women) in enumerate(cohorts):
                MarriageMarket(men, women)

                for c in self.communities:
                    if self["switch_rule"] in ["to_patri", "to_egal", "both"]:
                        c.decide_switch(restriction=self["switch_rule"])

                if write_agent_data:
                    self.write_agent_data(gen_no=g + 1, coh_no=i + 1, round_no=round_no)
                    # pass

                round_no += 1

                if write_to_data:
                    self.write_community_data(gen_no=g + 1, cohort_no=i + 1, round_no=round_no)

            for w in self.get_women():
                w.couple.reproduce_and_die()
