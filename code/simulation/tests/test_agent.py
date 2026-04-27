import unittest
import commons
import random
import numpy as np


class Attractiveness(unittest.TestCase):

    def test_initial_attractiveness_fixed_0(self):
        s = commons.Simulation()
        a = commons.Agent(gender="random", attr=0)
        self.assertEqual(a.attr, 0, "Should be 0")

    def test_initial_attractiveness_fixed_10(self):
        s = commons.Simulation()
        a = commons.Agent(gender="random", attr=10)
        self.assertEqual(a.attr, 10, "Should be 10")

    def test_initial_attractiveness_fixed_42(self):
        s = commons.Simulation()
        a = commons.Agent(gender="random", attr=42)
        self.assertEqual(a.attr, 42, "Should be 42")


    def test_initial_attractiveness_random_default_20(self):
        s = commons.Simulation()

        agents = [commons.Agent(gender="random") for i in range(100000)]
        attr_agents = [a.attr for a in agents]
        attr_mean = np.mean(attr_agents)
        attr_min = min(attr_agents)
        attr_max = max(attr_agents)

        self.assertAlmostEqual(attr_mean,10.5,1,"Mean is off. This is a probablistic test though.")
        self.assertAlmostEqual(attr_max,20,1,"Max is off. This is a probablistic test though.")
        self.assertAlmostEqual(attr_min,1,1,"Min is off. This is a probablistic test though.")


    def test_initial_attractiveness_random_20(self):
        s = commons.Simulation(attr_min=0, attr_max=20)

        agents = [commons.Agent(gender="random") for i in range(100000)]
        attr_agents = [a.attr for a in agents]
        attr_mean = np.mean(attr_agents)
        attr_min = min(attr_agents)
        attr_max = max(attr_agents)

        self.assertAlmostEqual(attr_mean,10,1,"Mean is off. This is a probablistic test though.")
        self.assertAlmostEqual(attr_max,20,1,"Max is off. This is a probablistic test though.")
        self.assertAlmostEqual(attr_min,0,1,"Min is off. This is a probablistic test though.")


    def test_initial_attractiveness_random_10(self):
        s = commons.Simulation(attr_min=0, attr_max=10)

        agents = [commons.Agent(gender="random") for i in range(10000)]
        attr_agents = [a.attr for a in agents]
        attr_mean = np.mean(attr_agents)
        attr_min = min(attr_agents)
        attr_max = max(attr_agents)

        self.assertAlmostEqual(attr_mean,5,1,"Mean is off. This is a probablistic test though.")
        self.assertAlmostEqual(attr_max,10,1,"Max is off. This is a probablistic test though.")
        self.assertAlmostEqual(attr_min,0,1,"Min is off. This is a probablistic test though.")




class ResidenceMembership(unittest.TestCase):

    def test_initial_residence(self):
        s = commons.Simulation()
        c = commons.Community()

        a = commons.Agent("male", residence = c)

        self.assertTrue(a in c.agents, "Agent is not in community")
        self.assertIs(a.residence, c, "Agent's residence var is not linked")


    def test_initial_member_by_default_egal(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian = True)

        a_male = commons.Agent("male", residence = c, first_gen = True)
        a_female = commons.Agent("female", residence = c, first_gen = True)

        self.assertTrue(a_male in c.get_members(), "Agent is not a member")
        self.assertIs(a_male.membership, c, "Agent's membership is not community")

        self.assertTrue(a_female in c.get_members(), "Agent is not a member")
        self.assertIs(a_female.membership, c, "Agent's membership is not community")


    def test_initial_member_by_default_patri(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian = False)

        a_male = commons.Agent("male", residence = c, first_gen = True)
        a_female = commons.Agent("female", residence = c, first_gen = True)

        self.assertTrue(a_male in c.get_members(), "Agent is not a member")
        self.assertIs(a_male.membership, c, "Agent's membership is not community")

        self.assertFalse(a_female in c.get_members(), "Agent is a member but shouldnt be")
        self.assertIs(a_female.membership, None, "Agent's membership is not community")


    def test_initial_residence_explicit_no_membership(self):
        """ 
        When you tell explicitely no membership 
        """
        s = commons.Simulation()
        c = commons.Community()

        a = commons.Agent("male", residence = c, membership = None)
        self.assertTrue(not a in c.get_members(), "Agent is not a member")
        self.assertIs(a.membership, None, "Agent's membership is not community")


    def test_initial_residence_another_membership(self):
        """
        When agent is created with a membership different from residence
        """
        s = commons.Simulation()
        c_res = commons.Community()
        c_mem = commons.Community()

        a = commons.Agent("male", residence = c_res, membership = c_mem)

        self.assertTrue(a in c_res.agents, "Agent is not in community")
        self.assertIs(a.residence, c_res, "Agent's residence var is not linked")
        self.assertFalse(a in c_res.get_members(), "Agent is a member, but shouldnt be")

        self.assertIs(a.membership, c_mem, "Agent's membership is not correct")

        



if __name__ == '__main__':
    unittest.main()
