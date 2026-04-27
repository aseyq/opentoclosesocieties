import random

from .config import global_config as parameters
from .helpers import Helpers
from .agent import Agent
from .family import Family


class Couple:
    def __init__(self, female, male):
        if female.gender != "female":
            raise ValueError("First argument must be female.")
        if male.gender != "male":
            raise ValueError("Second argument must be male.")
        if male.spouse or female.spouse or male.couple or female.couple:
            raise ValueError("One or both agents are already in a couple.")

        self.female = female
        self.male = male
        self.spouses = [female, male]
        self.host = None
        self.city_to_move = None

        self._marry_couple()

        family = Family()
        family.add_member(female)
        family.add_member(male)

    def __str__(self):
        return f"({self.female.name}, {self.male.name})"

    def __repr__(self):
        return self.__str__()

    def _marry_couple(self):
        self.male.spouse = self.female
        self.male.couple = self
        self.female.spouse = self.male
        self.female.couple = self

    def member_in_residence(self):
        return self.female.residence in self.get_couple_memberships()

    def _relocate_couple_in(self, com):
        self.male.set_residence(com)
        self.female.set_residence(com)

        if self.female.membership == com:
            com.grant_membership(self.male)

        if self.male.membership == com and com.is_egalitarian:
            com.grant_membership(self.female)

    def get_couple_memberships(self):
        return list(filter(None, {self.female.membership, self.male.membership}))

    def get_child_attr(self):
        if parameters["trait_inheritance"] == "average":
            inherited_component = (self.male.attr + self.female.attr) / 2
        elif parameters["trait_inheritance"] == "randomparent":
            inherited_component = random.choice([self.male, self.female]).attr
        else:
            raise ValueError("Invalid trait_inheritance strategy.")

        random_component = Helpers.get_random_comp(
            parameters["attr_min"],
            parameters["attr_max"],
            parameters["random_dist"]
        )

        parent_share = parameters["parent_attribute_share"]

        return round(
            inherited_component * parent_share + random_component * (1 - parent_share),
            parameters.get("", 2)
        )

    def get_child_asset(self, gender):
        if parameters["asset_inheritance_egalitarian"]:
            return (self.male.asset + self.female.asset) / 2
        return self.male.asset + self.female.asset if gender == "male" else 0


    def get_child_membership(self, gender):
        inheritance_type = parameters["membership_inheritance"]
        couple_memberships = self.get_couple_memberships()
        residence = self.female.residence

        if inheritance_type == "sticky":
            if residence.is_egalitarian:
                if residence in couple_memberships:
                    return residence
                elif self.male.membership:
                    if self.male.membership.is_egalitarian or gender == "male":
                        return self.male.membership
            else:
                if self.male.membership == residence and gender == "male":
                    return residence
                elif self.male.membership:
                    if self.male.membership.is_egalitarian or gender == "male":
                        return self.male.membership

        elif inheritance_type == "basic":
            if residence in couple_memberships and (gender == "male" or residence.is_egalitarian):
                return residence
            for c in random.sample(couple_memberships, len(couple_memberships)):
                if gender == "male" or c.is_egalitarian:
                    return c

        elif inheritance_type == "none":
            return None

        elif inheritance_type == "single":
            if residence in couple_memberships and (gender == "male" or residence.is_egalitarian):
                return residence

        return None

    def give_birth(self, gender, attr=None, asset=None):
        if gender not in ["male", "female"]:
            raise ValueError("Invalid gender for child.")

        if attr is None:
            attr = self.get_child_attr()
        if asset is None:
            asset = self.get_child_asset(gender)

        child = Agent(
            gender=gender,
            attr=attr,
            asset=asset,
            father_name=self.male.name,
            mother_name=self.female.name,
            residence=self.female.residence,
            membership=self.get_child_membership(gender),
            family=self.female.family,
        )
        return child

    def reproduce(self):
        if parameters["trait_inheritance"] == "comb":
            attr_set = [self.male.attr, self.female.attr]
            random.shuffle(attr_set)

            self.give_birth("male", attr=attr_set[0])
            self.give_birth("female", attr=attr_set[1])
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
