from copy import deepcopy
import random
import numpy as np
import math
#random.seed(10)
all_connections = []

class Node():
    def __init__(self):
        self.number = None
        self.layer_number = None
        self.value = None
        self.delta = None

        self.connections_out = []

class Connection():
    def __init__(self):
        self.output_node_number = None
        self.output_node_layer = None
        self.weight = random.uniform(-1, 1)
        self.enabled = True
        self.innovation = None

class Agent():
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.node_count = self.observation_space + self.action_space

        self.network = [[], []]
        self.connections = []
        self.fitness = None

        for i in range(self.observation_space):
            node = Node()
            node.number = i
            node.layer_number = 0
            self.network[0].append(deepcopy(node))
        for i in range(self.action_space):
            node = Node()
            node.number = self.observation_space + i
            node.layer_number = 1
            self.network[1].append(deepcopy(node))

        for input_node in range(len(self.network[0])):
            for output_node in range(len(self.network[1])):
                connection = Connection()

                connection.output_node_number = self.observation_space + output_node
                connection.output_node_layer = 1
                connection.innovation = self.finding_innovation(self.network[0][input_node], connection.output_node_number)
                
                self.network[0][input_node].connections_out.append(deepcopy(connection))
    
    def finding_innovation(self, starting_node_number, ending_node_number):
        connection_innovation = [starting_node_number, ending_node_number]

        if connection_innovation not in all_connections: all_connections.append(connection_innovation)
        innovation_number = all_connections.index(connection_innovation)

        return innovation_number

    def printing_stats(self):
        # this was just for me to print out the entire network and all the connections to see that everything works
        for i in range(len(self.network)):
            print("--")
            print("layer {}".format(i))
            for j in range(len(self.network[i])):
                print("node {} layer {} number {}".format(self.network[i][j], self.network[i][j].layer_number, self.network[i][j].number))

                for k in range(len(self.network[i][j].connections_out)):
                    if self.network[i][j].connections_out[k].enabled == True:
                        print("layer {} number {} innovation {}".format(self.network[i][j].connections_out[k].output_node_layer, self.network[i][j].connections_out[k].output_node_number, self.network[i][j].connections_out[k].innovation))

    def mutate_node(self): # adding node between two connected
        random_layer = random.randint(0, len(self.network)-2)
        
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        random_connection = random_node.connections_out[random_connection_index]

        random_node.connections_out[random_connection_index].enabled = False
        output_node_number = random_connection.output_node_number
        output_node_layer = random_connection.output_node_layer

        node = Node()
        node.number = self.node_count
        self.node_count += 1
        node.layer_number = random_layer + 1

        if abs(random_layer - output_node_layer) == 1: # layer in between needed
            self.network.insert(random_layer + 1, [])
            self.network[node.layer_number].append(deepcopy(node))
            new_node_index = len(self.network[random_layer + 1]) - 1

            connection = Connection()
            connection.output_node_number = node.number
            connection.output_node_layer = node.layer_number - 1
            connection.innovation = self.finding_innovation(random_node.number, node.number)
            self.network[random_layer][random_node_index].connections_out.append(deepcopy(connection))

            connection = Connection()
            connection.output_node_number = output_node_number
            connection.output_node_layer = output_node_layer 
            connection.innovation = self.finding_innovation(node.number, output_node_number)
            self.network[node.layer_number][new_node_index].connections_out.append(deepcopy(connection))

            for layer in range(node.layer_number+1, len(self.network)):
                for element in range(len(self.network[layer])):
                    self.network[layer][element].layer_number += 1
            
            for layer in range(len(self.network)):
                for element in range(len(self.network[layer])):
                    for connection in range(len(self.network[layer][element].connections_out)):
                        if self.network[layer][element].connections_out[connection].output_node_layer >= node.layer_number-1:
                            self.network[layer][element].connections_out[connection].output_node_layer += 1

        else: # layer in between already exists
            self.network[node.layer_number].append(deepcopy(node))
            new_node_index = len(self.network[random_layer + 1]) - 1

            connection = Connection()
            connection.output_node_number = node.number
            connection.output_node_layer = node.layer_number
            connection.innovation = self.finding_innovation(random_node.number, node.number)
            self.network[random_layer][random_node_index].connections_out.append(deepcopy(connection))

            connection = Connection()
            connection.output_node_number = output_node_number
            connection.output_node_layer = output_node_layer 
            connection.innovation = self.finding_innovation(node.number, output_node_number)
            self.network[node.layer_number][new_node_index].connections_out.append(deepcopy(connection))

    def mutate_link(self): # connect two unconnected nodes
        if len(self.network) > 2:

            first_random_layer = random.randint(0, len(self.network)-2)
            first_random_node_index = random.randint(0, len(self.network[first_random_layer])-1)
            first_random_node = self.network[first_random_layer][first_random_node_index]

            for layer in range(first_random_layer+1, len(self.network)):
                for node in range(len(self.network[layer])):
                    node_number = self.network[layer][node].number
                    node_layer = layer

                    found = False
                    for connection in range(len(first_random_node.connections_out)):
                        if first_random_node.connections_out[connection].output_node_number == node_number and first_random_node.connections_out[connection].output_node_layer == node_layer:
                            found = True
                    
                    if found == False:
                        break
                
                if found == False:
                    break
                
            if found == False:
                connection = Connection()
                connection.output_node_number = node_number
                connection.output_node_layer = node_layer
                self.network[layer][node].connections_out.append(deepcopy(connection))
    
    def mutate_enable_disable(self):
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        
        if random_node.connections_out[random_connection_index].enabled == False:
            self.network[random_layer][random_node_index].connections_out[random_connection_index].enabled = True
        else:
            self.network[random_layer][random_node_index].connections_out[random_connection_index].enabled = False

    def mutate_weight_shift(self):
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        number = random.uniform(0, 1)

        self.network[random_layer][random_node_index].connections_out[random_connection_index].weight *= number

    def mutate_weight_random(self):
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        number = random.uniform(-2, 2)

        self.network[random_layer][random_node_index].connections_out[random_connection_index].weight = random.uniform(-1, 1)

    def mutation(self):
        choice = random.randint(0, 10)
        if choice == 0:
            #print("mutate link")
            self.mutate_link()
        if choice == 1:
            #print("mutate node")
            self.mutate_node()
        if choice == 2 or choice == 5 or choice == 8:
            #print("enable disable")
            self.mutate_enable_disable()
        if choice == 3 or choice == 6 or choice == 9:
            #print("weight shift")
            self.mutate_weight_shift()
        if choice == 4 or choice == 7 or choice == 10:
            #print("weight random")
            self.mutate_weight_random()
