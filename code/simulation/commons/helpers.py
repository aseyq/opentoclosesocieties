import numpy as np
from .config import global_config as parameters
from datetime import datetime
import random
import math

class Helpers:
    @staticmethod
    def datetime_now():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def roundif(number, ndigits=2):
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
            "residence_location",
            "membership_name",
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
