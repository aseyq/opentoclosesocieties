import random
import uuid
from operator import itemgetter

from .helpers import Helpers

from .config import global_config as parameters
from math import dist, sqrt
import random


class Agent:
    def __init__(
        self,
        gender="random",
        residence=None,
        name=None,
        attr=None,
        asset=None,
        membership=None,
        father_name=None,
        mother_name=None,
        first_gen=False,
        family=None,
    ):
        if gender not in ["male", "female", "random"]:
            raise ValueError("gender must be 'male', 'female', or 'random'.")

        self.gender = gender if gender != "random" else random.choice(["male", "female"])
        self.name = name or uuid.uuid4().hex[:10]
        self.father_name = father_name
        self.mother_name = mother_name

        self.attr = attr if attr is not None else Helpers.get_random_comp(
            parameters["attr_min"], parameters["attr_max"], parameters["random_dist"]
        )

        self.asset = asset if asset is not None else (
            Helpers.get_random_comp(
                parameters["asset_min"], parameters["asset_max"], parameters["random_dist"]
            ) if self.gender == "male" or parameters["asset_inheritance_egalitarian"] else 0
        )

        self.residence = residence
        self.birthplace = residence.name if residence else None
        if residence:
            residence.add_agent(self)

        self.family = family
        if family:
            family.add_member(self)

        self.membership = membership
        if first_gen and residence:
            self.member_init()

        self.spouse = None
        self.couple = None
        self.matcher = None

    def __str__(self):
        digits = 2
        return (
            f"({self.name} G: {self.gender} At: {Helpers.roundif(self.attr, digits)} "
            f"As: {Helpers.roundif(self.asset, digits)} C: {Helpers.roundif(self.cost, digits)} "
            f"M: {getattr(self.membership, 'name', None)} R: {getattr(self.residence, 'name', None)})\n"
        )

    def __repr__(self):
        return self.__str__()

    def is_member(self, community):
        return self.membership is community

    def leave_current_city(self):
        self.membership = None
        if self.residence:
            self.residence.remove_agent(self)

    def member_init(self):
        if self.residence.is_egalitarian or self.gender == "male":
            self.membership = self.residence
        else:
            self.membership = None

    def get_member_coef(self):
        return 1 if self.residence is self.membership else 0

    def set_residence(self, community, with_membership=None):
        if self.residence is not community:
            community.add_agent(self, with_membership)
        elif with_membership:
            print(
                f"{self.name} is asked to move to {community.name} with membership but was already there"
            )

    def get_endowment(self):
        return self.residence.get_endowment_pc() * self.get_member_coef()

    def get_relocation_cost_to(self, community):
        if parameters["topography_structure"] == "complete":
            #print("We are in complete topography structure")
            return 0

        if parameters["topography_structure"] == "metric":
            #print("We are in metric topography structure")
            distance = self.residence.topography.get_distance(self.residence, community)
            #print(f"Relocation cost from {self.residence.name} to {community.name}: {relocation_cost}")
            relocation_cost = distance ** 2
            
            return relocation_cost

    def potential_communities_with(self, other):
        return {
            c for c in [
                self.residence,
                self.membership,
                other.residence,
                other.membership,
            ] if c
        }

    def get_endowment_in(self, community):
        return community.get_endowment_pc() if self.membership == community else 0.0

    def partner_suitable(self, other):
        if (self.gender, other.gender) not in [("male", "female"), ("female", "male")]:
            raise ValueError("Partner not suitable (heterosexuality)")


    def eval_marriage_with_in(self, other, com):
        self.partner_suitable(other)

        endowment = max(self.get_endowment_in(com), other.get_endowment_in(com))
        relocation_cost_community = self.get_relocation_cost_to(com)
        same_community = self.residence == other.residence

        ## SET UTILITY
        util = parameters["util_coef_attr"] * other.attr + parameters["util_coef_asset"] * Helpers.util(other.asset) + parameters["util_coef_endowment"] * Helpers.util(2 * endowment) - parameters["util_coef_cost"] * relocation_cost_community + parameters["same_community_bonus"] * same_community
        

        return util

    def create_relocational_tuples_from(self, other):
        return [(other, c) for c in self.potential_communities_with(other)]

    def create_relocational_tuples_from_list(self, list_agents):
        tuples = set()
        for other in list_agents:
            tuples.update(self.create_relocational_tuples_from(other))
        return list(tuples)
    
    def get_prefs_tuples(self, list_agents, verbose=False):
        tuples = self.create_relocational_tuples_from_list(list_agents)
        pref_agents = [self.eval_marriage_with_in(other, com) for other, com in tuples]
        zipprefs = zip(pref_agents, tuples)

        # ✅ Fix: specify sorting key
        sorted_tuples = [x for _, x in sorted(zipprefs, reverse=True, key=lambda pair: pair[0])]

        if verbose:
            for (util, (agent, com)) in zip(pref_agents, tuples):
                print(f"{agent.name} in {com.name} → {util}")
                print("--")

        return sorted_tuples

    def get_info_dict(self):
        return dict(
            name=self.name,
            gender=self.gender,
            residence_name=getattr(self.residence, "name", None),
            residence_egalitarian=getattr(self.residence, "is_egalitarian", None),
            residence_endowment=getattr(self.residence, "endowment", None),
            residence_init_endow_pc=getattr(self.residence, "initial_endowment_pc", None),
            residence_size=getattr(self.residence, "get_size", lambda: None)(),
            residence_member_size=getattr(self.residence, "get_member_size", lambda: None)(),
            residence_switchable=getattr(self.residence, "is_switchable", None),
            residence_wealthrank=getattr(self.residence, "wealthrank", None),
            residence_location=getattr(self.residence, "location", None),
            membership_name=getattr(self.membership, "name", None),
            attr=self.attr,
            asset=self.asset,
            endowment_pc=self.get_endowment(),
            birthplace=self.birthplace,
            partner_name=getattr(self.spouse, "name", None),
            partner_birthplace=getattr(self.spouse, "birthplace", None),
            father_name=self.father_name,
            mother_name=self.mother_name,
        )
