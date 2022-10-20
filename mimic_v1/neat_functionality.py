import numpy as np
import random
from network_generation import Agent
from network_training import Network
#random.seed(10)

class Neat():
    def information(self, species1, species2):
        species1_connections = []
        species2_connections = []
        species1_weights = []
        species2_weights = []

        for i in range(2):
            if i == 0: species = species1
            else: species = species2

            for layer in range(len(species.network_topology)):
                for node in range(len(species.network_topology[layer])):
                    for connection in range(len(species.network_topology[layer][node].connections_out)):
                        if species.network_topology[layer][node].connections_out[connection].enabled == True:
                            start_number = species.network_topology[layer][node].number
                            end_number = species.network_topology[layer][node].connections_out[connection].output_node_number

                            connection_attribute = [start_number, end_number]
                            weight = species.network_topology[layer][node].connections_out[connection].weight

                            if i == 0: 
                                species1_connections.append(connection_attribute)
                                species1_weights.append(weight)
                            else:
                                species2_connections.append(connection_attribute)
                                species2_weights.append(weight)
        
        return species1_connections, species2_connections, species1_weights, species2_weights

    def speciation(self, species1, species2):
        species1_connections, species2_connections, species1_weights, species2_weights = self.information(species1, species2)

        N = max(len(species1_connections), len(species2_connections))
        if N == 0:
            N = 1
        E = abs(species1.node_count - species2.node_count)

        D = 0
        both = species1_connections + species2_connections
        for i in range(len(both)):
            if both.count(both[i]) == 1:
                D += 1
        
        W = 0 # sum of weight differences for connections shared
        shorter_species = species1_connections
        if species1_connections <= species2_connections:
            shorter_species = species1_connections

        for i in range(len(shorter_species)):
            connection_identified = shorter_species[i]
            if connection_identified in species1_connections and connection_identified in species2_connections:
                index_species_one = species1_connections.index(connection_identified)
                index_species_two = species2_connections.index(connection_identified)
                
                W += abs(species1_weights[index_species_one] - species2_weights[index_species_two])

        number = E/N + D/N + 0.5*W
        #print(number)
        return number
    
    def making_children(self, species1, species2, observation_space, action_space):
        species1_connections, species2_connections, species1_weights, species2_weights = self.information(species1, species2)
        
        parents = [species1, species2]
        index = np.argmax([int(species1.fitness), int(species2.fitness)])
        fit_parent = parents[index] # take the fitest parent
        less_fit_parent = parents[abs(1-index)]

        child = Agent(observation_space, action_space) 
        child.network_topology = fit_parent.network_topology # take all the nodes from fittest parent
        child.node_count = fit_parent.node_count

        both = species1_connections + species2_connections
        connections_both = []
        parent_chosen = []
        copy_indexs = []
        both_weights = species1_weights + species2_weights
        
        for i in range(len(both)): 
            if both.count(both[i]) == 2:
                if both[i] not in connections_both:
                    connections_both.append(both[i])

        for i in range(len(connections_both)):
            parent_chosen.append(random.randint(0, 1))

        for i in range(len(connections_both)):
            indices = [index for index, element in enumerate(both) if element == connections_both[i]]
            copy_indexs.append(indices)

        for i in range(len(child.network_topology)):
            for j in range(len(child.network_topology[i])):
                for k in range(len(child.network_topology[i][j].connections_out)):
                    if child.network_topology[i][j].connections_out[k].enabled == True:
                        for p in range(len(connections_both)):
                            if connections_both[p][0] == child.network_topology[i][j].number and connections_both[p][1] == child.network_topology[i][j].connections_out[k].output_node_number:
                                child.network_topology[i][j].connections_out[k].weight = both_weights[copy_indexs[p][parent_chosen[p]]]

        network = Network()
        network.setting(child)
        return network
    
    def selecting_score(self, all_parents):
        fitness_scores = []
        for i in range(len(all_parents)):
            fitness_scores.append(all_parents[i].fitness)

        probabilities = []
        added_probs = []
        total_score = 0

        for i in range(len(fitness_scores)):
            total_score += fitness_scores[i]
        factor = 1/total_score

        for i in range(len(fitness_scores)):
            probabilities.append(round(fitness_scores[i] * factor, 2))
        
        added_probs.append(probabilities[0])
        for i in range(1, len(probabilities)):
            added_probs.append(added_probs[i-1] + probabilities[i])
        added_probs = [0] + added_probs

        roll = round(random.random(), 2)

        for i in range(1, len(added_probs)):
            if added_probs[i-1] <= roll <= added_probs[i]:
                return i-1
            if roll > added_probs[len(added_probs)-1]:
                return len(added_probs)-2
