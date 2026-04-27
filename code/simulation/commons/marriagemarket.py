from commons.matchers import match_couples
from commons.config import global_config as parameters
from commons.couple import Couple

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


