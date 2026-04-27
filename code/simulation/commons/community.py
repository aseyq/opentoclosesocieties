import uuid
import random

from .helpers import Helpers
from .config import global_config as parameters
from .agent import Agent
from .family import Family


class Community:
    def __init__(
        self,
        name=None,
        endowment=None,
        agents=None,
        create_agents=True,
        size=None,
        is_egalitarian=True,
        is_switchable=True,
    ):
        self.agents = set(agents) if agents else set()
        self.is_egalitarian = is_egalitarian
        self.topography = None
        self.is_switchable = is_switchable
        self.wealthrank = None

        self.name = name or uuid.uuid4().hex[:5].upper()

        self.endowment = (
            endowment
            if endowment is not None
            else Helpers.get_random_comp(
                parameters["endowment_min"],
                parameters["endowment_max"],
                parameters["random_dist"],
            )
        )

        for a in self.agents:
            a.residence = self

        if create_agents and size and not agents:
            self.create_community(size)

        if size and agents:
            print("Warning: custom agents and size provided. Ignoring size.")

        self.initial_endowment_pc = self.get_endowment_pc()

    def __str__(self):
        return f"({self.name}, Eg:{self.is_egalitarian}, W:{self.endowment}, P:{self.get_size()})"

    def __repr__(self):
        return self.name

    def assign_location(self):
        pass

    def get_members(self):
        return [a for a in self.agents if a.membership == self]

    def get_member_families(self):
        families = set()
        for a in self.agents:
            if a.membership == self and a.family:
                families.add(a.family)
        return families

    def get_num_families(self):
        return len(self.get_member_families())

    def get_member_size(self):
        return len(self.get_members())

    def get_male_member_size(self):
        return len([a for a in self.get_members() if a.gender == "male"])

    def get_female_member_size(self):
        return len([a for a in self.get_members() if a.gender == "female"])

    def grant_membership(self, a):
        a.membership = self

    def add_agent(self, a, with_membership=False):
        if self.get_agent_by_name(a.name):
            raise ValueError("Agent with the same name already exists.")

        if a.residence and (a.residence is not self):
            a.residence.remove_agent(a)

        a.residence = self
        self.agents.add(a)

        if with_membership:
            self.grant_membership(a)

    def remove_agent(self, a):
        if a.residence == self:
            a.residence = None
        self.agents.remove(a)

    def get_size(self):
        return len(self.agents)

    def get_male_size(self):
        return len([a for a in self.agents if a.gender == "male"])

    def get_female_size(self):
        return len([a for a in self.agents if a.gender == "female"])

    def get_agent_by_name(self, name):
        return next((a for a in self.agents if a.name == name), None)

    def get_agents_by_name(self, names):
        return [a for a in self.agents if a.name in names] or None

    def get_endowment_pc(self, level="couple"):
        if level == "individual":
            members = self.get_member_size()
            return self.endowment if members == 0 else self.endowment / members

        if level == "family":
            families = self.get_num_families()
            return self.endowment if families == 0 else self.endowment / (2 * families)

        if level == "couple":
            members = self.get_member_size()
            if members == 0:
                return self.endowment
            # this was the previous version but it makes more sense to divide by members
            #return (2 * self.endowment / members) if self.is_egalitarian else (self.endowment / (members))

            return (self.endowment / members) if self.is_egalitarian else (self.endowment / (2 * members))

        if level == "male":
            male_members = self.get_male_member_size()
            return self.endowment if male_members == 0 else 2 * self.endowment / male_members

    def repopulate(self):
        pass

    def create_community(self, size):
        for _ in range(int(size / 2)):
            family = Family()
            male = Agent(gender="male", residence=self, first_gen=True)
            female = Agent(gender="female", residence=self, first_gen=True)
            family.add_member(male)
            family.add_member(female)

    def show_members(self):
        return [a for a in self.agents if a.membership == self]

    def get_num_members(self):
        return len(self.show_members())

    def switch_patrilinear(self):
        self.is_egalitarian = False
        for a in self.agents:
            if a.gender == "female":
                a.membership = None

    def switch_egalitarian(self):
        self.is_egalitarian = True
        # for a in self.agents:
        #     if a.gender == "female" and a.birthplace == self:
        #         self.grant_membership(a)

    def get_prob_egal(self):
        ratio = Helpers.util(self.get_endowment_pc()) / Helpers.util(self.initial_endowment_pc)
        a = parameters["switch_threshold_to_egal_possible"]
        b = parameters["switch_threshold_to_egal_certain"]

        if ratio <= a:
            return 0
        elif ratio >= b:
            return 1
        else:
            return (ratio - a) / (b - a)

    def get_prob_patri(self):
        ratio = Helpers.util(self.get_endowment_pc()) / Helpers.util(self.initial_endowment_pc)
        a = parameters["switch_threshold_to_patri_certain"]
        b = parameters["switch_threshold_to_patri_possible"]

        if ratio <= a:
            return 1
        elif ratio >= b:
            return 0
        else:
            return 1 - (ratio - a) / (b - a)

    def get_decision_probs(self, restriction=None):
        p_egal = 0 if restriction == "to_patri" else self.get_prob_egal()
        p_patri = 0 if restriction == "to_egal" else self.get_prob_patri()
        p_statusquo = 1 - p_egal - p_patri
        return dict(egal=p_egal, patri=p_patri, statusquo=p_statusquo)

    def get_result_from_probs(self, probs: dict):
        return random.choices(list(probs.keys()), weights=probs.values(), k=1)[0]

    def get_switch_decision(self, restriction=None):
        return self.get_result_from_probs(self.get_decision_probs(restriction))

    def decide_switch(self, restriction=None):
        if self.is_switchable:
            decision = self.get_switch_decision(restriction)
            if decision == "egal":
                self.switch_egalitarian()
            elif decision == "patri":
                self.switch_patrilinear()
