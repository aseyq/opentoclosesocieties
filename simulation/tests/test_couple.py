import unittest
import commons
import random
import numpy as np

class CoupleInit(unittest.TestCase):

    def test_couple_formation(self):
        s = commons.Simulation()
        a_male = commons.Agent(gender="male")
        a_female = commons.Agent(gender="female")

        couple = commons.Couple(a_female, a_male)


        # Couple items
        self.assertIs(couple.male, a_male, "male is not correct")
        self.assertIs(couple.female, a_female, "female is not correct")
        self.assertEqual(couple.spouses, [a_female, a_male], "spouse list is not correct")

        # Agent links spouses
        self.assertIs(a_male.spouse, a_female, "spouse is not correct")
        self.assertIs(a_male.couple,couple, "self.couple is wrong")

        self.assertIs(a_female.spouse, a_male, "spouse is not correct")
        self.assertIs(a_female.couple,couple, "self.couple is wrong")


class CoupleChildAttr(unittest.TestCase):

    def test_get_child_attr_default(self):
        s = commons.Simulation()
        a_male = commons.Agent(gender="male", attr=10)
        a_female = commons.Agent(gender="female", attr= 20)

        couple = commons.Couple(a_female, a_male)

        # Default parent_weight = 0.75
        # Default random_weight = 0.25
        # Default expected_random = 10.5

        expected_value = 13.875 
        expected_min = 11.5
        expected_max = 16.25


        attr = [couple.get_child_attr() for i in range(100000)]
        attr_mean = np.mean(attr)
        attr_min = min(attr)
        attr_max = max(attr)

        self.assertAlmostEqual(attr_mean,expected_value,1,
                               "Mean is off. This is a probablistic test though.")

        self.assertAlmostEqual(attr_max,expected_max,1,
                               "Max is off. This is a probablistic test though.")

        self.assertAlmostEqual(attr_min,expected_min,1,
                               "Min is off. This is a probablistic test though.")


    def test_get_child_attr_parent_share_05(self):
        s = commons.Simulation(parent_attribute_share = 0.5)
        a_male = commons.Agent(gender="male", attr=10)
        a_female = commons.Agent(gender="female", attr= 20)

        couple = commons.Couple(a_female, a_male)


        expected_value = 12.75 
        expected_min = 8
        expected_max = 17.5


        attr = [couple.get_child_attr() for i in range(100000)]
        attr_mean = np.mean(attr)
        attr_min = min(attr)
        attr_max = max(attr)

        self.assertAlmostEqual(attr_mean,expected_value,1,
                               "Mean is off. This is a probablistic test though.")

        self.assertAlmostEqual(attr_max,expected_max,1,
                               "Max is off. This is a probablistic test though.")

        self.assertAlmostEqual(attr_min,expected_min,1,
                               "Min is off. This is a probablistic test though.")




class CoupleMemberships(unittest.TestCase):
    def test_no_member(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)

        a_male = commons.Agent(gender="male", residence=c, membership=None)
        a_female = commons.Agent(gender="female", residence=c, membership=None)

        couple = commons.Couple(a_female, a_male)

        self.assertEqual(couple.get_couple_memberships(), [])


    def test_male_member(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)

        a_male = commons.Agent(gender="male", residence=c, membership=c)
        a_female = commons.Agent(gender="female", residence=c, membership=None)

        couple = commons.Couple(a_female, a_male)

        self.assertEqual(couple.get_couple_memberships(), [c])

    def test_female_member(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)

        a_male = commons.Agent(gender="male", residence=c, membership=None)
        a_female = commons.Agent(gender="female", residence=c, membership=c)

        couple = commons.Couple(a_female, a_male)

        self.assertEqual(couple.get_couple_memberships(), [c])


    def test_both_members(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)

        a_male = commons.Agent(gender="male", residence=c, membership=c)
        a_female = commons.Agent(gender="female", residence=c, membership=c)

        couple = commons.Couple(a_female, a_male)

        self.assertEqual(couple.get_couple_memberships(), [c])


    def test_two_different_memberships(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)
        d = commons.Community(is_egalitarian=True)


        a_male = commons.Agent(gender="male", residence=c, membership=c)
        a_female = commons.Agent(gender="female", residence=c, membership=d)

        couple = commons.Couple(a_female, a_male)

        # We use sets here because order matters in lists
        self.assertEqual(set(couple.get_couple_memberships()), set([d,c]))


    def test_two_different_memberships_with_another_residence(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)
        d = commons.Community(is_egalitarian=True)
        e = commons.Community(is_egalitarian=True)


        a_male = commons.Agent(gender="male", residence=e, membership=c)
        a_female = commons.Agent(gender="female", residence=e, membership=d)

        couple = commons.Couple(a_female, a_male)

        # We use sets here because order matters in lists
        self.assertEqual(set(couple.get_couple_memberships()), set([d,c]))


class ChildMembership(unittest.TestCase):

    def test_both_parents_member_egalitarian(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=True)

        a_male = commons.Agent(gender="male", residence=c,  membership=c)
        a_female = commons.Agent(gender="female", residence=c, membership=c)

        couple = commons.Couple(a_female, a_male)
        self.assertIs(couple.get_child_membership("male"), c, "Male membership is incorrect")
        self.assertIs(couple.get_child_membership("female"), c, "Female membership is incorrect")


    def test_father_member_patrilinear(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)

        a_male = commons.Agent(gender="male", residence=c,  membership=c)
        a_female = commons.Agent(gender="female", residence=c, membership=None)

        couple = commons.Couple(a_female, a_male)
        self.assertIs(couple.get_child_membership("male"), c, "Male membership is incorrect")
        self.assertIs(couple.get_child_membership("female"), None, "Female membership is incorrect")


    def test_father_member_abroad_patrilinear(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)
        d = commons.Community(is_egalitarian=False)

        a_male = commons.Agent(gender="male", residence=c,  membership=d)
        a_female = commons.Agent(gender="female", residence=c, membership=None)

        couple = commons.Couple(a_female, a_male)
        self.assertIs(couple.get_child_membership("male"), d, "Male membership is incorrect")
        self.assertIs(couple.get_child_membership("female"), None, "Female membership is incorrect")

    def test_father_member_abroad_egalitarian(self):
        s = commons.Simulation()
        c = commons.Community(is_egalitarian=False)
        d = commons.Community(is_egalitarian=True)

        a_male = commons.Agent(gender="male", residence=c,  membership=d)
        a_female = commons.Agent(gender="female", residence=c, membership=None)

        couple = commons.Couple(a_female, a_male)
        self.assertIs(couple.get_child_membership("male"), d, "Male membership is incorrect")
        self.assertIs(couple.get_child_membership("female"), d, "Female membership is incorrect")








if __name__ == '__main__':
    unittest.main()
