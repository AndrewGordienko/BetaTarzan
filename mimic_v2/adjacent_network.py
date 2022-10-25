import numpy as np
from copy import deepcopy
import random

class Neuron():
    def __init__(self):
        self.number = None
        self.layer_number = None
        self.value = None
        self.connections_out = []

class Connection():
    def __init__(self):
        self.output_node_number = None
        self.output_node_layer = None
        self.weight = random.uniform(-0.1, 0.1)

class Network:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.network_topology = []
        neuron_total = 0
        self.lr = 0.01

        for i in range(len(self.dimensions)): self.network_topology.append([])

        for i in range(len(self.network_topology)):
            for j in range(self.dimensions[i]):
                neuron = Neuron()
                neuron.number = neuron_total
                neuron.layer_number = i
                self.network_topology[i].append(deepcopy(neuron))
                neuron_total += 1

        for i in range(len(self.network_topology)-1):
            for j in range(len(self.network_topology[i])):
                for q in range(len(self.network_topology[i+1])):
                    connection = Connection()
                    connection.output_node_number = self.network_topology[i+1][q].number
                    connection.output_node_layer = i+1
                    self.network_topology[i][j].connections_out.append(deepcopy(connection))
    
    def forward(self, state):
        for i in range(len(self.network_topology)):
            for j in range(len(self.network_topology[i])):
                self.network_topology[i][j].value = 0

        for i in range(len(self.network_topology[0])):
            self.network_topology[0][i].value = state[0][i]
        
        for i in range(len(self.network_topology)-1):
            if i != 0 and i != len(self.network_topology)-1:
                for j in range(len(self.network_topology[i])):
                        self.network_topology[i][j].value = self.ReLU(self.network_topology[i][j].value)

            for j in range(len(self.network_topology[i])):
                for k in range(len(self.network_topology[i][j].connections_out)):
                    output_node_layer = self.network_topology[i][j].connections_out[k].output_node_layer
                    output_node_number = self.network_topology[i][j].connections_out[k].output_node_number
                    weight = self.network_topology[i][j].connections_out[k].weight

                    for p in range(len(self.network_topology[output_node_layer])):
                        if self.network_topology[output_node_layer][p].number == output_node_number:
                            self.network_topology[output_node_layer][p].value += self.network_topology[i][j].value * weight

        last_values = []
        maximum_index = len(self.network_topology)-1
        for i in range(len(self.network_topology[maximum_index])):
            last_values.append(np.tanh(self.network_topology[maximum_index][i].value))

        return last_values
    
    def backprop(self, value, ground_truth):
        all_deltas = []
        all_deltas.append(value - ground_truth)
        delta_counter = 0
        temporary_delta = []

        layer = len(self.network_topology) - 2

        for layer in range(len(self.network_topology)-2, 0, -1):
            for i in range(len(self.network_topology[layer])):
                for j in range(len(self.network_topology[layer][i].connections_out)):
                    weight = self.network_topology[layer][i].connections_out[j].weight
                    deriv = self.relu2deriv(self.network_topology[layer][i].value)

                    node_number = self.network_topology[layer][i].connections_out[j].output_node_number
                    for p in range(len(self.network_topology[layer+1])):
                        if self.network_topology[layer+1][p].number == node_number:
                            index = p
                            break
                    
                    temporary_delta.append(all_deltas[delta_counter][0][index] * weight * deriv)

            all_deltas.append(deepcopy([temporary_delta]))
            delta_counter += 1
        
        delta_counter = 0
        for layer in range(len(self.network_topology)-2, -1, -1):
            for i in range(len(self.network_topology[layer])):
                for j in range(len(self.network_topology[layer][i].connections_out)):
                    node_number = self.network_topology[layer][i].connections_out[j].output_node_number
                    for p in range(len(self.network_topology[layer+1])):
                            if self.network_topology[layer+1][p].number == node_number:
                                index = p
                                break
                    self.network_topology[layer][i].connections_out[j].weight -= self.lr * self.network_topology[layer][i].value * all_deltas[delta_counter][0][index]
            delta_counter += 1

    def ReLU(self, x):
        return x * (x > 0)
    
    def relu2deriv(self, input):
        return int(input > 0)
        
    def printing_stats(self):
        for i in range(len(self.network_topology)):
            print("--")
            print("layer {}".format(i))
            for j in range(len(self.network_topology[i])):
                print("node {} layer {} number {}".format(self.network_topology[i][j], self.network_topology[i][j].layer_number, self.network_topology[i][j].number))
                for k in range(len(self.network_topology[i][j].connections_out)):
                    print("connected to node on layer {} number {} weight {}".format(self.network_topology[i][j].connections_out[k].output_node_layer, self.network_topology[i][j].connections_out[k].output_node_number, self.network_topology[i][j].connections_out[k].weight))
