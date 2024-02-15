import os
import random
import uuid
import itertools
from matching.games import StableMarriage
from matching import Player
from operator import itemgetter
import numpy as np
from datetime import datetime
import csv
import math
from itertools import combinations

from .datafile import DataFile  # My own data wrapper
from .matchers import match_couples

class Helpers:
    @staticmethod
    def datetime_now():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def roundif(number, ndigits=None):
        if number is not None:
            return round(number, ndigits)
        else:
            return None

    @staticmethod
    def get_classname(obj):
        return type(obj).__name__

    @staticmethod
    def key_with_maxval(d):
        # https://stackoverflow.com/a/12343826
        """a) create a list of the dict's keys and values;
        b) return the key with the max value
        """
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    @staticmethod
    def get_random_comp_uniform(min_val, max_val):
        random_comp = min_val + random.uniform(0, 1) * (max_val - min_val)
        return random_comp

    @staticmethod
    def get_random_comp_beta(min_val, max_val):
        beta_a = 5
        beta_b = 5
        random_comp = min_val + np.random.beta(beta_a, beta_b) * (max_val - min_val)
        return random_comp

    @staticmethod
    def get_random_comp(min_val, max_val, dist="uniform"):

        if dist == "uniform":
            return Helpers.get_random_comp_uniform(min_val, max_val)

        if dist == "beta":
            return Helpers.get_random_comp_beta(min_val, max_val)

    @staticmethod
    def log(x):
        return math.log(x)

    @staticmethod
    def util(x):
        if parameters['utility'] == "linear":
            return x

        if parameters['utility'] == "sqrt":
            return math.sqrt(x)

    @staticmethod
    def get_columns():
        columns = [
            "sim",
            "generation",
            "cohort",
            "round",
            "name",
            "gender",
            "residence_name",
            "residence_egalitarian",
            "residence_endowment",
            "residence_init_endow_pc",
            "residence_size",
            "residence_member_size",
            "residence_switchable",
            "residence_wealthrank",
            "membership_name",
            "cost",
            "attr",
            "asset",
            "endowment_pc",
            "birthplace",
            "partner_name",
            "partner_birthplace",
            "father_name",
            "mother_name",
            "par_attr_min",
            "par_attr_max",
            "par_asset_min",
            "par_asset_max",
            "par_cost_min",
            "par_cost_max",
            "par_endowment_min",
            "par_endowment_max",
            "par_switch_threshold_to_egal_certain",
            "par_switch_threshold_to_patri_certain",
            "par_switch_threshold_to_egal_possible",
            "par_switch_threshold_to_patri_possible",
            "par_switch_rule",
            "par_parent_attribute_share",
            "par_random_dist",
            "par_trait_inheritance",
            "par_asset_inheritance_egalitarian",
            "par_membership_inheritance",
            "par_utility",
            "par_util_coef_attr",
            "par_util_coef_asset",
            "par_util_coef_cost",
            "par_util_coef_endowment",
            "par_proposer",
            "par_same_community_bonus",
            "par_topography_structure",
        ]
        return columns


h = Helpers()


class Agent:
    def __init__(
        self,
        gender,
        residence=None,
        name=None,
        attr=None,
        asset=None,
        cost=None,
        membership=None,
        father_name=None,
        mother_name=None,
        first_gen=False,  # Should be explicitly defined
        family = None,
    ):

        if gender not in ["male", "female", "random"]:
            raise ValueError("gender should be 'male' or 'female' or 'random'.")

        if gender == "random":
            self.gender = random.choice(["male", "female"])
        else:
            self.gender = gender

        self.father_name = father_name
        self.mother_name = mother_name

        if name:
            self.name = name
        else:
            self.name = uuid.uuid4().hex[:10]

        if attr is not None:
            self.attr = attr
        else:
            if parameters["random_dist"] == "uniform":
                self.attr = Helpers.get_random_comp(
                    min_val=parameters["attr_min"],
                    max_val=parameters["attr_max"],
                    dist="uniform",
                )

            if parameters["random_dist"] == "beta":
                self.attr = Helpers.get_random_comp(
                    min_val=parameters["attr_min"],
                    max_val=parameters["attr_max"],
                    dist="beta",
                )

        if asset is not None:
            self.asset = asset
        else:
            if gender == "male" or parameters["asset_inheritance_egalitarian"]:
                if parameters["random_dist"] == "uniform":
                    self.asset = Helpers.get_random_comp(
                        min_val=parameters["asset_min"],
                        max_val=parameters["asset_max"],
                        dist="uniform",
                    )

                if parameters["random_dist"] == "beta":
                    self.asset = Helpers.get_random_comp(
                        min_val=parameters["asset_min"],
                        max_val=parameters["asset_max"],
                        dist="beta",
                    )
            else:
                self.asset = 0

        if cost is not None:
            self.cost = cost
        else:
            if parameters["random_dist"] == "uniform":
                self.cost = Helpers.get_random_comp(
                    min_val=parameters["cost_min"],
                    max_val=parameters["cost_max"],
                    dist="uniform",
                )

            if parameters["random_dist"] == "beta":
                self.cost = Helpers.get_random_comp(
                    min_val=parameters["cost_min"],
                    max_val=parameters["cost_max"],
                    dist="beta",
                )

        self.residence = residence

        self.family = family
        if self.family:
            family.add_member(self)

        if residence:
            self.birthplace = residence.name
            residence.add_agent(self)
        else:
            self.birthplace = None

        self.membership = membership

        if first_gen and residence:
            self.member_init()

        self.spouse = None
        self.couple = None
        self.matcher = None

    def __str__(self):
        # printing an agent like: (myname P: 0 T: cooperator)
        if self.membership:
            membership_name = self.membership.name
        else:
            membership_name = None

        if self.residence:
            residence_name = self.residence.name
        else:
            residence_name = None

        return (
            "("
            + self.name
            + " G: "
            + str(self.gender)
            + " At: "
            + str(Helpers.roundif(self.attr, parameters["digits_show"]))
            + " As: "
            + str(Helpers.roundif(self.asset, parameters["digits_show"]))
            + " C: "
            + str(Helpers.roundif(self.cost, parameters["digits_show"]))
            + " M: "
            + str(membership_name)
            + " R: "
            + str(residence_name)
            + ") \n"
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
        """
        Initiates membership when the agent is born without parents.
        Triggered explicitly by first_gen = True
        """

        if self.residence.is_egalitarian:
            self.membership = self.residence
        else:
            if self.gender == "male":
                self.membership = self.residence

            if self.gender == "female":
                self.membership = None

    def get_member_coef(self):
        """
        Membership coefficient for the community the agent lives. If I am a member, I get 1, otherwise 0. Used for endowment calculations.
        """

        if self.residence is self.membership:
            return 1
        else:
            return 0

    def partner_suitable(self, other):
        """
        Checks if the partner has the opposite gender.
        """
        if not (
            (self.gender, other.gender) in [("male", "female"), ("female", "male")]
        ):
            raise ValueError("Partner not suitable (heterosexuality)")

    def set_residence(self, community, with_membership=None):
        """
        This changes the residence (moves the agent) to another town.
        IF the person is not already living there. Its a substitute to
        community.add_agent but doesn't do nothing.
        """

        if self.residence is not community:
            community.add_agent(self, with_membership)

        if self.residence is community and with_membership:
            print(
                self.name
                + "is asked to move to "
                + self.community.name
                + "with membership but was already living there"
            )

    def get_endowment(self):
        """
        Returns endowment per capita if is a member of the community the agent lives in
        """
        return self.residence.get_endowment_pc() * self.get_member_coef()

    def get_relocation_cost_to(self, community):
        """
        Returns relocation cost to a community

        Args:
            community (community): Target community object to move

        Returns:
            integer: the amount of costs (positive)
        """
        
        multiplier = self.residence.topography.get_connection_multiplier(self.residence, community)
        if multiplier == 0:
            return 0
        elif multiplier == 1:
            return self.cost
        elif multiplier == -1:
            return 9999999
        else:
            raise ValueError("Multiplier not valid")
        
        

    def potential_communities_with(self, other):
        """
        Returns the set of all the communities that one can live with the given partner.
        That includes the communities that either of them live in or are members of.

        Returns:
            set: set of potential communities with the other.
        """

        potential_communities = [
            self.residence,
            self.membership,
            other.residence,
            other.membership,
        ]

        community_set = {c for c in potential_communities if c}

        return community_set

    def get_endowment_in(self, community):
        """
        Returns endowment per capita in case the agents lives in the given community.
        This is to evaluate the communities of which the agent have the membership
        outside the community he is a resident of.

        Args:
            community (community): Community to check the membership/endowment_pc.

        Returns:
            integer: the endowment_pc if moved to the community
        """

        if self.membership == community:
            return community.get_endowment_pc()
        else:
            return 0.0

    def potential_community_utils_with(self, other, verbose=False):
        """
        THIS IS RETIRED AFTER THE NEW MATCHING PROCEDURE
        Returns the couple utilities of the all the potential communities that one can live with the given partner.

        Returns:
            dictionary: keys are communities, values are utilities.
        """

        communities = self.potential_communities_with(other)

        c_with_util = dict.fromkeys(communities, 0)

        for c in c_with_util.keys():
            my_cost = self.get_relocation_cost_to(c)
            other_cost = other.get_relocation_cost_to(c)

            my_endowment = self.get_endowment_in(c)
            other_endowment = other.get_endowment_in(c)
            max_endowment = max(my_endowment, other_endowment)

            util = parameters['util_coef_endowment'] * h.util(2 * max_endowment) - parameters['util_coef_cost'] * my_cost - parameters['util_coef_cost'] * other_cost

            c_with_util[c] = util

            if verbose:
                print(c)

        return c_with_util

    def determine_community_to_move_with(self, other):
        c_with_util = self.potential_community_utils_with(other)
        best_community = Helpers.key_with_maxval(c_with_util)

        return best_community

    def eval_marriage_with(self, other, verbose=False):
        """
        THIS IS (PROBABLY) RETIRED AFTER THE NEW MATCHING PROCEDURE
        Evaluates marriage with another agents and returns the potential utility from the marriage

        Args:
            other (agent): Agent object to be evaluated

        Returns:
            float: Utility from marrying the agent
        """

        self.partner_suitable(other)

        com = self.determine_community_to_move_with(other)

        endowment = max(self.get_endowment_in(com), other.get_endowment_in(com))
        
        cost = self.get_relocation_cost_to(com)
        #        util = other.attr + other.asset + 2 * endowment - cost

        util =  parameters['util_coef_attr'] * other.attr + parameters['util_coef_asset'] * h.util(other.asset) + parameters['util_coef_endowment'] * h.util(2 * endowment) - cost

        if verbose:
            print("community:", com.name, "| endowment:", endowment, "| cost:", cost)

        return util

    def eval_marriage_with_in(self, other, com):
        """
        Evaluates marriage with another agents and returns the potential utility from the marriage

        Args:
            other (agent): Agent object to be evaluated

        Returns:
            float: Utility from marrying the agent
        """

        self.partner_suitable(other)

        endowment = max(self.get_endowment_in(com), other.get_endowment_in(com))
        
        cost = self.get_relocation_cost_to(com)

        # python coerses this to 0 and 1. but maybe better to be 
        # explicit
        same_community = self.residence == other.residence
        
        util = parameters['util_coef_attr'] * other.attr \
                + parameters['util_coef_asset'] * h.util(other.asset) \
                + parameters['util_coef_endowment'] * h.util(2 * endowment)  \
                - parameters['util_coef_cost'] * cost \
                + parameters['same_community_bonus'] * same_community
        
        return util

    def create_relocational_tuples_from(self, other):
        """
        Creates the preferences for a potential partner for each potential location.
        A relocational triple (agent, location)

        Args:
            other (agent): Partner to create the relocational tuples for

        Returns:
            list: list of partner-location tuple
        """

        communities = self.potential_communities_with(other)

        tuples = []

        for c in communities:
            tuples.append((other,c))

        return tuples

    def create_relocational_tuples_from_list(self,list_agents):
        """
        Creates the preferences for a potential partner for each potential location.
        A relocational triple (agent, location)

        Args:
            other (agent): Partner to create the relocational tuples for

        Returns:
            list: list of partner-location tuple
        """

        tuples = set()

        for other in list_agents:
            _t = self.create_relocational_tuples_from(other)
            for t in _t:
                tuples.add(t)

        return list(tuples)

    def get_prefs_tuples(self, list_agents, verbose=False):

        tuples = self.create_relocational_tuples_from_list(list_agents)

        pref_agents = [self.eval_marriage_with_in(other, com) for (other, com) in tuples]
        zipprefs = zip(pref_agents, tuples)

        sorted_tuples = [
            x for y, x in sorted(zipprefs, reverse=True, key=itemgetter(0))
        ]

        if verbose:
            for i, p in enumerate(pref_agents):
                print(tuples[i], p)
                print("--")

        return sorted_tuples

    def get_prefs(self, list_agents, verbose=False):
        """
        Evaluates marriage with a list of people and ranks them

        Args:
            list_agents (list): A list of agents object

        Returns:
            list: List of agents in the order of most preferred to least preferred
        """

        pref_agents = [self.eval_marriage_with(a) for a in list_agents]
        zipprefs = zip(pref_agents, list_agents)

        sorted_list_agents = [
            x for y, x in sorted(zipprefs, reverse=True, key=itemgetter(0))
        ]

        if verbose:
            for i, p in enumerate(pref_agents):
                print(list_agents[i], p)
                print("--")

        return sorted_list_agents

    def get_info_dict(self):
        """
        Returns a dictionary with agent information. To be used to write the data
        """
        info = dict(
            name=self.name,
            gender=self.gender,
            residence_name=self.residence.name,
            residence_egalitarian=self.residence.is_egalitarian,
            residence_endowment=self.residence.endowment,
            residence_init_endow_pc=self.residence.initial_endowment_pc,
            residence_size=self.residence.get_size(),
            residence_member_size=self.residence.get_member_size(),
            residence_switchable=self.residence.is_switchable,
            residence_wealthrank=self.residence.wealthrank,
            membership_name=None if self.membership is None else self.membership.name,
            cost=self.cost,
            attr=self.attr,
            asset=self.asset,
            endowment_pc=self.get_endowment(),
            birthplace=self.birthplace,
            partner_name=None if self.spouse is None else self.spouse.name,
            partner_birthplace=None if self.spouse is None else self.spouse.birthplace,
            father_name=self.father_name,
            mother_name=self.mother_name,
        )
        return info

class Family:
    def __init__(self, members=None):
        """
        Initializes a family with a list of agents

        Args:
            members (list): List of agents
        """
        if members:
            self.members = set(members)

            # TODO assign members to family
        else:
            self.members = set()


    def add_member(self, a):
        self.members.add(a)
        a.family = self
        
    def remove_member(self, a):
        self.members.remove(a)
        a.family = None

class Couple:
    def __init__(self, female, male):

        if female.gender == "female":
            self.female = female
        else:
            raise ValueError("Female is not a proper object")

        if male.gender == "male":
            self.male = male
        else:
            raise ValueError("Male is not proper object")

        if male.spouse or female.spouse or male.couple or female.couple:
            raise ValueError("Already married person in the couple")

        self.spouses = [female, male]
        self.host = None

        self._marry_couple()

        _family = Family()
        _family.add_member(male)
        _family.add_member(female)
        

        self.city_to_move = None

    def member_in_residence(self):
        if self.female.residence in self.get_couple_memberships():
            return True
        else:
            return False

    def __str__(self):
        return "(" + self.female.name + "," + self.male.name + ")"

    def __repr__(self):
        return self.__str__()

    def _marry_couple(self):
        """
        Marries the couple by linking couple and spouse attributes for both agents. This is triggered at the creation of the couple
        """
        self.male.spouse = self.female
        self.male.couple = self
        self.female.spouse = self.male
        self.female.couple = self

    
    def _relocate_couple_in(self, com):
        """
        Relocates couple to the most profitable town
        """

        city_to_move = com

        self.male.set_residence(city_to_move)
        self.female.set_residence(city_to_move)

        if self.female.membership == city_to_move:
            city_to_move.grant_membership(self.male)

        if self.male.membership == city_to_move:
            if city_to_move.is_egalitarian:
                city_to_move.grant_membership(self.female)


    def get_child_attr(self):
        """
        Picks attrativeness for a child of the couple. It is a mix between the average attractiveness of the couple and a random compenent.

        Returns:
           float: Attractiveness

        """
        parent_attr_avg = (self.male.attr + self.female.attr) / 2

        if parameters["trait_inheritance"] == "average":
            parent_attr_avg = (self.male.attr + self.female.attr) / 2
            inherited_component = parent_attr_avg

        if parameters["trait_inheritance"] == "randomparent":
            random_parent = random.choice([self.male, self.female])
            inherited_component = random_parent.attr

        if parameters["random_dist"] == "uniform":
            random_component = Helpers.get_random_comp(
                min_val=parameters["attr_min"],
                max_val=parameters["attr_max"],
                dist="uniform",
            )

        if parameters["random_dist"] == "beta":
            random_component = Helpers.get_random_comp(
                min_val=parameters["attr_min"],
                max_val=parameters["attr_max"],
                dist="beta",
            )

        parent_share = parameters["parent_attribute_share"]

        return round(
            parent_attr_avg * parent_share + random_component * (1 - parent_share),
            parameters["digits_after"],
        )

    def get_child_asset(self, gender):
        if parameters["asset_inheritance_egalitarian"]:
            return (self.male.asset + self.female.asset) / 2

        else:
            if gender == "male":
                return self.male.asset + self.female.asset
            else:
                return 0

    def get_child_cost(self, random_dist="uniform", beta_a=5, beta_b=5):
        """
        Picks a cost for a child of the couple. It is a mix between the average cost of the couple and a random compenent.

        Returns:
           float: Cost

        """
        if parameters["trait_inheritance"] == "average":
            parent_cost_avg = (self.male.cost + self.female.cost) / 2
            inherited_component = parent_cost_avg

        if parameters["trait_inheritance"] == "randomparent":
            random_parent = random.choice([self.male, self.female])
            inherited_component = random_parent.cost

        if parameters["random_dist"] == "uniform":
            random_component = Helpers.get_random_comp(
                min_val=parameters["cost_min"],
                max_val=parameters["cost_max"],
                dist="uniform",
            )

        if parameters["random_dist"] == "beta":
            random_component = Helpers.get_random_comp(
                min_val=parameters["cost_min"],
                max_val=parameters["cost_max"],
                dist="beta",
            )

        parent_share = parameters["parent_attribute_share"]

        return round(
            inherited_component * parent_share + random_component * (1 - parent_share),
            parameters["digits_after"],
        )

    def get_couple_memberships(self):
        memberships_or_none = {self.female.membership, self.male.membership}
        memberships = list(filter(None, memberships_or_none))
        return memberships

    def get_child_membership(self, gender):
        """
        Determines the membership of the child. At this point the couple

        """

        if parameters["membership_inheritance"] == "sticky":
            residence = self.female.residence

            couple_memberships = self.get_couple_memberships()
            child_membership = None

            # Residence egalitraian
            if residence.is_egalitarian:
                # Egalitarian and couple is member in residence
                if residence in couple_memberships:
                    child_membership = residence

                # Egalitarian and couple is not a member
                else:
                    # Father's other membership
                    if self.male.membership:
                        if self.male.membership.is_egalitarian:
                            child_membership = self.male.membership
                        else:
                            if gender == "male":
                                child_membership = self.male.membership
                            else:
                                child_membership = None  # Unnecessary but explecit
            # Residence patrilinear
            else:
                # Father is a member
                if self.male.membership == residence:

                    if gender == "male":
                        child_membership = residence

                    else:
                        child_membership = None  # Unnecessary but explicit

                elif self.male.membership:
                    if self.male.membership.is_egalitarian:
                        child_membership = self.male.membership
                    else:
                        if gender == "male":
                            child_membership = self.male.membership
                        else:
                            child_membership = None  # Unnecessary but explicit

            return child_membership

        if parameters["membership_inheritance"] == "basic":

            couple_memberships = self.get_couple_memberships()
            child_membership = None
            # Egalitraian
            residence = (
                self.female.residence
            )  # doens't matter male of female. They have the same res.

            if residence in couple_memberships and gender == "male":
                child_membership = residence

            elif residence in couple_memberships and residence.is_egalitarian:
                child_membership = residence

            elif len(couple_memberships) > 0:
                random.shuffle(couple_memberships)

                for c in couple_memberships:
                    if gender == "male":
                        child_membership = c
                        break

                    if gender == "female":
                        if c.is_egalitarian:
                            child_membership = c
                            break
                        else:
                            pass

            return child_membership

        if parameters["membership_inheritance"] == "none":
            return None


        if parameters["membership_inheritance"] == "single":

            couple_memberships = self.get_couple_memberships()
            child_membership = None
            # Egalitraian
            residence = (
                self.female.residence
            )  # doens't matter male of female. They have the same res.

            if residence in couple_memberships and gender == "male":
                child_membership = residence

            elif residence in couple_memberships and residence.is_egalitarian:
                child_membership = residence

            return child_membership
         
    def give_birth(self, gender, attr=None, asset=None, cost=None):
        if gender not in ["male", "female"]:
            raise ValueError("wrong gender")

        if not attr:
            attr = self.get_child_attr()

        if not asset:
            asset = self.get_child_asset(gender)

        if not cost:
            cost = self.get_child_cost()

        child = Agent(
            gender=gender,
            attr=attr,
            asset=asset,
            cost=cost,
            father_name=self.male.name,
            mother_name=self.female.name,
            residence=self.female.residence,
            membership=self.get_child_membership(gender),
            family = self.female.family,
        )

        return child

    def reproduce(self):
        if parameters["trait_inheritance"] == "comb":
            # If combinatorcs, agent will give their features in combination to children
            attr_set = [self.male.attr, self.female.attr]
            random.shuffle(attr_set)

            cost_set = [self.male.cost, self.female.cost]
            random.shuffle(cost_set)

            self.give_birth("male", attr=attr_set[0], cost=cost_set[0])
            self.give_birth("female", attr=attr_set[1], cost=cost_set[1])

        else:
            self.give_birth("male")
            self.give_birth("female")

    def die(self):
        self.male.leave_current_city()
        self.male.family.remove_member(self.male)

        self.female.leave_current_city()
        self.female.family.remove_member(self.female)


    def reproduce_and_die(self):
        self.reproduce()
        self.die()


class Community:
    def __init__(
        self,
        name=None,
        endowment=None,
        agents=[],
        create_agents=True,
        size=None,
        is_egalitarian=True,
        is_switchable = True,
    ):

        self.agents = set(agents)
        self.is_egalitarian = is_egalitarian
        self.topography = None
        self.is_switchable = is_switchable
        self.wealthrank = None

        if name:
            self.name = name
        else:
            self.name = uuid.uuid4().hex[:5].upper()

        if endowment:
            self.endowment = endowment
        else:
            if parameters["random_dist"] == "uniform":
                self.endowment = Helpers.get_random_comp(
                    min_val=parameters["endowment_min"],
                    max_val=parameters["endowment_max"],
                    dist="uniform",
                )

            if parameters["random_dist"] == "beta":
                beta_a = 5
                beta_b = 5

                self.endowment = Helpers.get_random_comp(
                    min_val=parameters["endowment_min"],
                    max_val=parameters["endowment_max"],
                    dist="beta",
                )

        for a in agents:
            a.residence = self

        if create_agents and size and not agents:
            self.create_community(size)

        if size and agents:
            print(
                "Warning: Looks like you set custom agents and size at the same time. Size is ignored. You can add as many as agents"
            )

        self.initial_endowment_pc = self.get_endowment_pc()

    def __str__(self):
        #        return "hi"
        return (
            "("
            + self.name
            + ", Eg:"
            + str(self.is_egalitarian)
            + ", W:"
            + str(self.endowment)
            + ", P: "
            + str(self.get_size())
            + ")"
        )

    def __repr__(self):
        return self.name

    def assign_location(self):
        pass

    def get_members(self):
        # to be deprecated after get_member_families
        return [a for a in self.agents if a.membership == self]

    def get_member_families(self):
        # Families is either a couple or a single person
        # It creates a set of mixed objects. So you better come up with a better design.
        families = set()

        for a in self.agents:
            if a.membership == self:
                families.add(a.family)
            #### if married
            #if a.couple:
            #    if a.couple.member_in_residence():
            #        families.add(a.couple)

            #### if single
            #else:
            #    if a.membership == self:
            #        families.add(a)

        return families

    def get_num_families(self):
        return len(self.get_member_families())

    def get_member_size(self):
        return len(self.get_members())

    def get_male_member_size(self):
        men = [a for a in self.get_members() if a.gender == "male"]
        return len(men)

    def grant_membership(self, a):
        a.membership = self

    def add_agent(self, a, with_membership=None):
        if self.get_agent_by_name(a.name):
            raise ValueError("some agent with the same name is already in")

        if a.residence and (a.residence is not self):
            a.residence.remove_agent(a)

        a.residence = self

        self.agents.add(a)

        if with_membership:
            self.grant_membership(a)

    def remove_agent(self, a):
        #        print(a)
        #        print(self)
        if a.residence == self:
            a.residence = None
        self.agents.remove(a)

    def get_size(self):
        return len(self.agents)

    def get_agent_by_name(self, name):
        return next((a for a in self.agents if a.name == name), None)

    def get_agents_by_name(self, names):
        agents = [a for a in self.agents if a.name in names]
        if agents:
            return agents
        else:
            return None

    def get_endowment_pc(self, level="couple"):
        # In order to make patrilineal and egalitarian comparable
        # I am implementing a family based calculation

        ## The old way
        if level == "individual":
            member_size = self.get_member_size()
            if member_size == 0:
                return self.endowment
            return self.endowment / self.get_member_size()

        if level == "family":
            num_families = self.get_num_families()
            if num_families == 0:
                return self.endowment

            return self.endowment / (2 * num_families)

        if level == "couple":
            member_size = self.get_member_size()
    
            if self.is_egalitarian:
                if member_size == 0:
                    return self.endowment
                    
                return 2 * self.endowment / member_size
            
            if not self.is_egalitarian:
                if member_size == 0:
                    return self.endowment

                return self.endowment / member_size
                
        if level == "male":
            male_member_size = self.get_male_member_size()
            return 2 * self.endowment / male_member_size

    def repopulate(self):
        pass

    def create_community(self, size):
        #iter_gender = itertools.cycle(["male", "female"])

        for i in range(0, int(size/2)):
            _family = Family()
            #Agent(gender=next(iter_gender), residence=self, first_gen=True)
            #Agent(gender=next(iter_gender), residence=self, first_gen=True)
            _male = Agent(gender="male", residence=self, first_gen=True)
            _female = Agent(gender="female", residence=self, first_gen=True)

            _family.add_member(_male)
            _family.add_member(_female)
            
    def show_members(self):
        return [a for a in self.agents if a.is_member]

    def get_num_members(self):
        return len(self.show_members())

    def switch_patrilinear(self):
        self.is_egalitarian = False

        for a in self.agents:

            if a.gender == "female":
                a.membership = None

    def switch_egalitarian(self):
        self.is_egalitarian = True

        for a in self.agents:
            ## Here we give the membership to all women born in tne community. this might be wrong but otherwise the endowment_pc calculation
            ## will have a low shock (egalitarian endowment_pc calculation is endowment / num_members while patrilineal es 2 * endowment / num_members)
            if a.gender == "female" and a.birthplace == self:
                a.is_member = True


    # switch method
    def get_prob_egal(self):
        egalitarian_possible = parameters['switch_threshold_to_egal_possible']
        egalitarian_certain = parameters['switch_threshold_to_egal_certain']

        endowmentpc_ratio = h.util(self.get_endowment_pc()) / h.util(self.initial_endowment_pc)

        if endowmentpc_ratio <= egalitarian_possible:
            return 0
        
        elif endowmentpc_ratio >= egalitarian_certain:
            return 1
        
        else:
            return (endowmentpc_ratio - egalitarian_possible) / (egalitarian_certain - egalitarian_possible)
        
    def get_prob_patri(self):
        patrilineal_possible = parameters['switch_threshold_to_patri_possible']
        patrilineal_certain = parameters['switch_threshold_to_patri_certain']

        endowmentpc_ratio = h.util(self.get_endowment_pc()) / h.util(self.initial_endowment_pc)

        if endowmentpc_ratio <= patrilineal_certain:
            return 1
        
        elif endowmentpc_ratio >= patrilineal_possible:
            return 0
        
        else:
            return 1 - (endowmentpc_ratio - patrilineal_certain) / (patrilineal_possible - patrilineal_certain)
    

    
    def get_decision_probs(self, restriction=None):
        if restriction == "to_patri":
            p_egal = 0
        else:
            p_egal = self.get_prob_egal()
        
        if restriction == "to_egal":
            p_patri = 0
        else:
            p_patri = self.get_prob_patri()
            
        p_statusquo = 1 - p_egal - p_patri
        return dict(egal=p_egal, patri=p_patri, statusquo=p_statusquo)


    def get_result_from_probs(self, probs:dict):
        return random.choices(list(probs.keys()), weights=probs.values(), k=1)[0]

    def get_switch_decision(self, restriction=None):
        probs = self.get_decision_probs(restriction=restriction)
        result = self.get_result_from_probs(probs)
        return result

    def decide_switch(self, restriction=None):
        if self.is_switchable:
            decision = self.get_switch_decision(restriction=restriction)

            if decision == "egal":
                self.switch_egalitarian()

            if decision == "patri":
                self.switch_patrilinear()

            if decision == "statusquo":
                pass
        


class MarriageMarketOLD:
    """
    # Matching and relocatin happens at market generation
    """

    def __init__(self, men, women):
        self.men = set(men)
        self.women = set(women)
        self.men_matchers = None
        self.women_matchers = None
        self.couples = []
        self.solution = None

        self._create_matching_players()
        self._create_player_preferences()
        self._get_matching()
        self._pair_solution()

    def get_by_name(self, name):
        # This is a workaround. At some point (I guess at the game.solve(), agent object duplicates. So my previous implementation, based on linking Player object from the matching module and the agent object from our code didn't work
        return next((x for x in self.men.union(self.women) if x.name == name), None)

    def _create_matching_players(self):
        men_matchers = []
        for m in self.men:
            if m.spouse:
                raise ValueError("Already married:" + m.name)

            current_player = Player(name=m.name)
            current_player.agent = m
            m.player = current_player
            men_matchers.append(current_player)

        women_matchers = []
        for w in self.women:
            if w.spouse:
                raise ValueError("Already married:" + w.name)
            current_player = Player(name=w.name)
            current_player.agent = w
            w.player = current_player
            women_matchers.append(current_player)

        self.men_matchers = men_matchers
        self.women_matchers = women_matchers

    def _create_player_preferences(self):
        for m in self.men:
            prefs = m.get_prefs(self.women)
            #            print(prefs)
            prefs_players = [w.player for w in prefs]
            m.player.set_prefs(prefs_players)

        for w in self.women:
            prefs = w.get_prefs(self.men)
            prefs_players = [m.player for m in prefs]
            w.player.set_prefs(prefs_players)

    def _get_matching(self):
        game = StableMarriage(self.men_matchers, self.women_matchers)
        market_solution = game.solve()
        self.solution = market_solution

    def _pair_solution(self):
        for f, m in self.solution.items():
            current_couple = Couple(self.get_by_name(m.name), self.get_by_name(f.name))
            current_couple._relocate_couple()
            self.couples.append(current_couple)



class Simulation:
    def __init__(
        self,
        sim_no=1,
        num_coms=None,
        com_size=None,
        fileobj=None,
        init_egalitarian=True,
        communities=None,
        attr_min=1,
        attr_max=20,
        asset_min=1,
        asset_max=20,
        cost_min=0,
        cost_max=4,
        endowment_min=110,
        endowment_max=440,
        switch_threshold_to_patri_certain=0.70,
        switch_threshold_to_egal_certain=1.2,
        switch_threshold_to_patri_possible=1,
        switch_threshold_to_egal_possible=0.9,
        switch_rule="no_switch",
        switchable_communities=None,
        parent_attribute_share=0.75,
        random_dist="uniform",  # options: uniform, beta
        trait_inheritance="average",  # options: average, randomparent
        asset_inheritance_egalitarian=True,
        membership_inheritance="basic",  # options: basic, sticky
        utility="linear",  # options: linear, log, logoneplusx, logoneplusxdivten
        util_coef_attr=1,
        util_coef_asset=1,
        util_coef_cost=1,
        util_coef_endowment=1,
        proposer="men",
        same_community_bonus = 0,
        topography_structure= "complete",
    ):

        self.set_parameters(
            attr_min=attr_min,
            attr_max=attr_max,
            asset_min=asset_min,
            asset_max=asset_max,
            cost_min=cost_min,
            cost_max=cost_max,
            endowment_min=endowment_min,
            endowment_max=endowment_max,
            switch_threshold_to_patri_certain=switch_threshold_to_patri_certain,
            switch_threshold_to_egal_certain=switch_threshold_to_egal_certain,
            switch_threshold_to_patri_possible=switch_threshold_to_patri_possible,
            switch_threshold_to_egal_possible=switch_threshold_to_egal_possible,
            switch_rule=switch_rule,
            switchable_communities=switchable_communities,
            parent_attribute_share=parent_attribute_share,
            random_dist=random_dist,
            trait_inheritance=trait_inheritance,
            asset_inheritance_egalitarian=asset_inheritance_egalitarian,
            membership_inheritance=membership_inheritance,
            utility=utility,
            util_coef_attr=util_coef_attr,
            util_coef_asset=util_coef_asset,
            util_coef_cost=util_coef_cost,
            util_coef_endowment=util_coef_endowment,
            proposer=proposer,
            same_community_bonus=same_community_bonus,
            topography_structure=topography_structure,
        )

        self.sim_no = sim_no
        self.communities = []

        if communities:
            self.set_communities(communities)
            
        elif num_coms and com_size:
            self.create_communities(num_coms, com_size, init_egalitarian)

        if fileobj:
            self.fileobj = fileobj

    def set_parameters(
        # TODO I should fix the repetitions in the parameters
        self,
        attr_min=1,
        attr_max=20,
        asset_min=1,
        asset_max=20,
        cost_min=0,
        cost_max=4,
        endowment_min=110,
        endowment_max=440,
        switch_threshold_to_patri_certain=0.70,
        switch_threshold_to_egal_certain=1.2,
        switch_threshold_to_patri_possible=1,
        switch_threshold_to_egal_possible=0.9,

        switch_rule="no_switch",
        switchable_communities=None,
        parent_attribute_share=0.75,
        digits_after=4,
        digits_show=2,
        random_dist="uniform",
        trait_inheritance="average",
        asset_inheritance_egalitarian=True,
        membership_inheritance="basic",
        utility="linear",
        util_coef_attr=1,
        util_coef_asset=1,
        util_coef_cost=1,
        util_coef_endowment=1,
        proposer="men",
        same_community_bonus=0,
        topography_structure="complete",
    ):

        self.parameters = locals()
        global parameters
        parameters = self.parameters

    def set_communities(self, communities):
        self.communities = communities
        
        self.set_topography(parameters['topography_structure'])

    def create_communities(self, num_coms, com_size, is_egalitarian=True, is_switchable=True):
        self.communities = []

        # here i have a trick. I dont know the wealth rank
        # at the point of creation. So for lock-in and domino cases
        # i cannot create, say the 2nd rich community as egalitarian.
        # And if i switch back to egalitarian right after, women wont get 
        # membership in the first round because it will be like a switch. 
        # so in case of lock in and domino, i create everyone as egalitarian 
        # and then switch the ones that are necessary to patrilineal
        if parameters['switchable_communities'] is not None:
            for i in range(num_coms):
                c_ = Community(size=com_size, is_egalitarian=is_egalitarian, is_switchable=is_switchable)
                self.communities.append(c_)
                self.set_topography(parameters['topography_structure'])
        else:
            for i in range(num_coms):
                c_ = Community(size=com_size, is_egalitarian=is_egalitarian, is_switchable=is_switchable)
                self.communities.append(c_)
                self.set_topography(parameters['topography_structure'])

        sortedcoms =  sorted(self.communities, key=lambda x: -x.endowment)

        for c in sortedcoms:
            c.wealthrank = sortedcoms.index(c)+1

        # setting switchability
        if parameters['switchable_communities'] is not None:
            for c in self.communities:
                if c.wealthrank in parameters['switchable_communities']:
                    c.is_switchable = True
                    # switchables should start from egalitarian
                    # not anymore
                    if not is_egalitarian:
                        c.switch_patrilinear()
                else:
                    c.is_switchable = False
                    if not is_egalitarian:
                        c.switch_patrilinear()


    def set_topography(self, structure):
        self.topography = Topography(self.communities, structure)
        
    def get_people(self, order="random"):
        people = []

        for c in self.communities:
            people.extend(list(c.agents))

        # randomizatin is crucial here because cohort are formed in order
        if order == "random":
            random.shuffle(people)

        return people

    def get_men(self, order="random"):
        people = self.get_people(order)
        people_men = [a for a in people if a.gender == "male"]
        return people_men

    def get_women(self, order="random"):
        people = self.get_people(order)
        people_women = [a for a in people if a.gender == "female"]
        return people_women

    def create_cohorts(self, num_cohorts):
        cohorts_men = np.array_split(
            self.get_men(), num_cohorts
        )  # good that it keeps the objects
        cohorts_women = np.array_split(
            self.get_women(), num_cohorts
        )  # good that it keeps the objects

        cohorts_zipped = zip(cohorts_men, cohorts_women)
        return cohorts_zipped

    def write_agent_data(self, gen_no, coh_no, round_no):
        parameter_dict = dict(
            sim=self.sim_no,
            generation=gen_no,  # generation
            cohort=coh_no,  # cohort
            round=round_no,
            par_attr_min=parameters["attr_min"],
            par_attr_max=parameters["attr_max"],
            par_asset_min=parameters["asset_min"],
            par_asset_max=parameters["asset_max"],
            par_cost_min=parameters["cost_min"],
            par_cost_max=parameters["cost_max"],
            par_endowment_min=parameters["endowment_min"],
            par_endowment_max=parameters["endowment_max"],
            par_switch_threshold_to_patri_certain=parameters["switch_threshold_to_patri_certain"],
            par_switch_threshold_to_egal_certain=parameters["switch_threshold_to_egal_certain"],
            par_switch_threshold_to_patri_possible=parameters["switch_threshold_to_patri_possible"],
            par_switch_threshold_to_egal_possible=parameters["switch_threshold_to_egal_possible"],
            par_switch_rule=parameters["switch_rule"],
            par_parent_attribute_share=parameters["parent_attribute_share"],
            par_random_dist=parameters["random_dist"],
            par_trait_inheritance=parameters["trait_inheritance"],
            par_asset_inheritance_egalitarian=parameters[
                "asset_inheritance_egalitarian"
            ],
            par_membership_inheritance=parameters["membership_inheritance"],
            par_utility=parameters["utility"],
            par_util_coef_attr=parameters["util_coef_attr"],
            par_util_coef_asset=parameters["util_coef_asset"],
            par_util_coef_cost=parameters["util_coef_cost"],
            par_util_coef_endowment=parameters["util_coef_endowment"],
            par_proposer=parameters["proposer"],
            par_same_community_bonus=parameters["same_community_bonus"],
            par_topography_structure=parameters["topography_structure"],
        )

        for a in self.get_people(order=None):
            agent_dict = a.get_info_dict()
            global combined
            combined = {**parameter_dict, **agent_dict}

            self.fileobj.write_line(**combined)

    def run_sim(self, num_gen, num_cohorts, write_to_data=True):

        # The initial round
        if write_to_data:
            self.write_agent_data(gen_no=0, coh_no=0, round_no=0)

        round_no = 1

        for g in range(num_gen):

            # Creating cohorts
            cohorts = self.create_cohorts(num_cohorts)

            # For each cohort
            for i, (men, women) in enumerate(cohorts):

                # Go to marriage market. Matching and moving happends in the market
                market = MarriageMarket(men, women)
                del market

                # Decision by cities
                for c in self.communities:
                    # new switch method
                    if parameters["switch_rule"] in ["to_patri", "to_egal", "both"]:
                        c.decide_switch(restriction=parameters["switch_rule"])
                    
                    # old switch method
                    #if parameters["switch_rule"] in ["to_patri", "both"]:
                    #    c.decide_rule_to_patrililar(round_no) # haha it worked with a typo for a long time!

                    #if parameters["switch_rule"] in ["to_egal", "both"]:
                    #    c.decide_rule_to_egalitarian(round_no)

                    # no_switch condition is missing but we dont need that

                if write_to_data:
                    self.write_agent_data(gen_no=g + 1, coh_no=i + 1, round_no=round_no)

                round_no += 1

            for w in self.get_women():
                w.couple.reproduce_and_die()


class MarriageMarket:
    """
    # Matching and relocatin happens at market generation
    """
    def __init__(self, men, women):
        self.men = list(men)
        self.women = list(women)

        self.couples = []
        self.solution = None

        self._get_matching()
        self._pair_solution()

    def _get_matching(self):
        self.solution = match_couples(self.men, self.women, proposer=parameters['proposer'])
        
    def _pair_solution(self):
        for f, m, c in self.solution:
            #print(f)
            current_couple = Couple(f,m)
            current_couple._relocate_couple_in(c)
            self.couples.append(current_couple)


class Topography:
    def __init__(self, communities, structure="manual"):
        if structure not in ["manual", "complete", "circular"]:
            raise ValueError("structure must be one of 'manual', 'complete', 'circular'")

        self.set_communities(communities)

        
        self.connections = list()
        if structure != "manual":
            self.set_connections(structure=structure)
        pass

    def set_communities(self, communities):
        self.communities = communities
        for c in self.communities:
            c.topography = self
 

    def set_connections(self, connections=None, structure="complete"):
        if structure == "complete":
            community_pairs_iter = combinations(self.communities, 2) # all possible pairs

            for community1, community2 in community_pairs_iter:
                self.connections.append({community1, community2})

        if structure == "circular":
            for i,j in enumerate(self.communities):
                self.connections.append({self.communities[i], 
                                         self.communities[(i+1)%len(self.communities)]})
                                        # next one, if doesnt exist, first one. thanks copilot!
            

    def get_connection_multiplier(self, community1, community2):
        # level 0: no cost
        # level 1: standard cost
        # level 9999: extreme cost

        if community1 == community2:
            return 0
        elif {community1, community2} in self.connections:
            return 1
        else:
            return -1