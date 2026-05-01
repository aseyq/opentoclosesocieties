class Config:
    def __init__(self, overrides=None):
        self.params = {
            "random_dist": "beta",
            "attr_min": 0,
            "attr_max": 1,
            "asset_min": 0,
            "asset_max": 1,
            "init_egalitarian": True,
            "asset_inheritance_egalitarian": True,
            "endowment_min": 0,
            "endowment_max": 160,
            "utility": "sqrt",
            "util_coef_attr": 1,
            "util_coef_asset": 1,
            "util_coef_cost": 0.1,
            "util_coef_endowment": 1,
            "same_community_bonus": 0,
            "switch_threshold_to_egal_possible": 0.98,
            "switch_threshold_to_egal_certain": 1.58, 
            "switch_threshold_to_patri_possible": 0.93,
            "switch_threshold_to_patri_certain": 0.33, 
            "trait_inheritance": "average",
            "parent_attribute_share": 0.8,
            "membership_inheritance": "sticky", 
            "proposer": "men", 
            "topography_structure": "metric",
            "switch_rule": "both",  # "to_patri", "to_egal", "no_switch", "both"
            "switchable_communities": None, # set to a list of switchable communities by rank, for instance: [5],
            # IF none, all communities are switchable/none are switchable depending on the switch_rule
        }

        if overrides:
            self.params.update(overrides)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def get(self, key, default=None):
        return self.params.get(key, default)


global_config = Config()
