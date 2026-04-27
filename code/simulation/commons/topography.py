import math
import random
from itertools import combinations


class Topography:
    def __init__(self, simulation, structure="complete"):
        if structure not in ["complete", "metric"]:
            raise ValueError("structure must be one of 'complete', 'metric'")
        
        self.simulation = simulation
        self.communities = simulation.communities

        self.set_communities_topography()

        self.connections = list()
        self.distances = list()
        self.positions = dict()  # Store positions for metric structure
        self.structure = structure

        self.set_connections(structure)

    def set_communities_topography(self):
        for c in self.communities:
            c.topography = self
 

    def set_connections(self, structure="complete"):
        if structure == "complete":
            community_pairs_iter = combinations(self.communities, 2) # all possible pairs

            for community1, community2 in community_pairs_iter:
                self.connections.append({community1, community2})

            # distances (all equal to 0)
            for community1, community2 in combinations(self.communities, 2):
                self.distances.append({"communities": {community1, community2}, "distance": 0})

        if structure == "metric":

            for community in self.communities:
                r = 2 * math.sqrt(random.uniform(0, 1))  # Random radius within the circle
                angle = random.uniform(0, 2 * math.pi)  # Random angle
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                self.positions[community] = (x, y)
                community.location = (x, y)  # Store position in community

            # Calculate distances between each pair of communities
            for community1, community2 in combinations(self.communities, 2):
                x1, y1 = community1.location
                x2, y2 = community2.location
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                self.distances.append({"communities": {community1, community2}, "distance": distance})
            
            #print(self.distances)

            community_pairs_iter = combinations(self.communities, 2)
            for community1, community2 in community_pairs_iter:
                self.connections.append({community1, community2})
                # print(f"Connection: {community1.name} - {community2.name}", 
                #       f"Distance: {self.get_distance(community1, community2)}")
                
            
            

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
        
    def get_distance(self, community1, community2):
        #print(f"get_distance: {community1}, {community2}")
        if community1 == community2:
            return 0
        for distance_entry in self.distances:
            if distance_entry["communities"] == {community1, community2}:
                return distance_entry["distance"]
        return -1