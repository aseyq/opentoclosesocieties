import random 

class Matcher:
    def __init__(self, agent, candidates):
        self.agent = agent # linked object 
        self.agent.matcher = self
        self.prefs = agent.get_prefs_tuples(candidates)
        self.current_match = None

class Proposee(Matcher):
    def __init__(self, agent, candidates):
        super().__init__(agent, candidates)
        self.offers = list()
        self.current_accepted_offer = None

    def receive_offer(self, offer):
        self.offers.append(offer)

    def get_offer_rank(self, offer):
        return self.prefs.index(offer)

    def evaluate_offers(self):
        if self.offers:
            ranked_offers = sorted(self.offers, key=self.get_offer_rank)
            
            best_offer = ranked_offers[0] 
            best_proposer = best_offer[0]

            if self.current_match:
                # I have a match
                if self.get_offer_rank(best_offer) > self.get_offer_rank(self.current_accepted_offer):
                    # But i like my match better
                    return None
                if self.get_offer_rank(best_offer) < self.get_offer_rank(self.current_accepted_offer):
                    #print("blocking pair")
                    #print("me", self.agent.name)
                    #print("best_offer:", best_offer)
                    #print("current offer", self.current_accepted_offer)
                    #print("-"*50)
                    # No the best offer is better. So I will break the current matc
                    self.current_match.current_match = None
                    self.current_match.temporarily_matched = False

            self.current_match = best_proposer.matcher
            self.current_accepted_offer = best_offer

            best_proposer.matcher.current_match = self
            best_proposer.matcher.temporarily_matched = True

class Proposer(Matcher):
    def __init__(self, agent, candidates):
        super().__init__(agent, candidates)
        self.current_offer_rank = -1 # indexing start from zero
        self.temporarily_matched = False

    def next_proposal(self):
        current_pref = self.prefs[self.current_offer_rank]

        proposee = current_pref[0]
        proposed_location = current_pref[1]

        proposal = (self.agent, proposed_location)        

        return (proposal, proposee)

    def propose_next(self):
        self.current_offer_rank += 1
        proposal, proposee  = self.next_proposal()
        proposee.matcher.receive_offer(proposal)

def match_couples(men, women, proposer="men"):
    if proposer == "random":
        proposer = random.choice(["men","women"])

    if proposer == "men":
        proposers = [Proposer(m, women) for m in men]      
        proposees = [Proposee(w, men) for w in women]      

    if proposer == "women":
        proposees = [Proposee(m, women) for m in men]      
        proposers = [Proposer(w, men) for w in women]      
    
    unmatched_proposers = [p for p in proposers if not p.temporarily_matched]

    while unmatched_proposers:
        for pr in unmatched_proposers:
            pr.propose_next()

        for pe in proposees:
            pe.evaluate_offers()  

        unmatched_proposers = [p for p in proposers if not p.temporarily_matched]
        #for p in proposees:
        #if p.current_match:
        #        print(p.agent.name, p.current_match.agent.name, p.current_accepted_offer[1].name)

    matchings = []
    for p in proposees:
        if proposer == "men":
            matchings.append((p.agent, p.current_match.agent, p.current_accepted_offer[1]))

        if proposer == "women":
            matchings.append((p.current_match.agent, p.agent, p.current_accepted_offer[1]))  # reversing men and women

    return matchings