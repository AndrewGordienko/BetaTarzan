import numpy as np
from copy import deepcopy
import random 
import math

observation_space = 4
action_space = 4

MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
random.seed(1)

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, observation_space),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, action_space),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class Network:
    def __init__(self):
        self.agent = None
        self.network_topology = None
        self.node_count = None
        self.fitness = None
        self.lr = 0.01
        self.second_lr = 0.0001

        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.learn_step_counter = 0
        self.net_copy_interval = 10
    
    def setting(self, agent):
        self.agent = agent
        self.network_topology = agent.network
        self.node_count = agent.node_count
        self.fitness = agent.fitness
        
    def setting_learning(self):
        self.action_agent = deepcopy(self.agent)
        self.target_agent = deepcopy(self.agent)

        self.forward_run = []
        for layer in range(len(self.network_topology)):
            self.forward_run.append([])
            for node in range(len(self.network_topology[layer])):
                self.forward_run[layer].append(0)
    
    def forward(self, state):
        for i in range(len(self.network_topology)):
            for j in range(len(self.network_topology[i])):
                self.network_topology[i][j].value = 0

        for i in range(len(self.network_topology[0])):
            self.network_topology[0][i].value = state[0][i]
        
        for i in range(len(self.network_topology)-1):
            for j in range(len(self.network_topology[i])):
                for k in range(len(self.network_topology[i][j].connections_out)):
                    if self.network_topology[i][j].connections_out[k].enabled == True:
                        output_node_layer = self.network_topology[i][j].connections_out[k].output_node_layer
                        output_node_number = self.network_topology[i][j].connections_out[k].output_node_number
                        weight = self.network_topology[i][j].connections_out[k].weight

                        for p in range(len(self.network_topology[output_node_layer])):
                            if self.network_topology[output_node_layer][p].number == output_node_number:
                                self.network_topology[output_node_layer][p].value += self.network_topology[i][j].value * weight

            if i != 0 and i != len(self.network_topology)-1:
                for j in range(len(self.network_topology[i])):
                        self.network_topology[i][j].value = self.ReLU(self.network_topology[i][j].value)

        last_values = []
        value_array = []
        maximum_index = len(self.network_topology)-1
        for i in range(len(self.network_topology[maximum_index])):
            value_array.append(self.network_topology[maximum_index][i].value)

        item = np.array(value_array)
        normalized_value = np.interp(item, (item.min(), item.max()), (-1,+1))
        last_values.append(normalized_value)

        return last_values

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backprop(self, value, ground_truth):
        for layer in range(len(self.network_topology)):
            for node in range(len(self.network_topology[layer])):
                self.network_topology[layer][node].delta = 0
        
        last_delta = value - ground_truth

        last_layer = len(self.network_topology)-1
        for node in range(len(self.network_topology[last_layer])):
            self.network_topology[last_layer][node].delta = last_delta[0][node]
        
        for layer in range(len(self.network_topology)-2, -1, -1):
            for node in range(len(self.network_topology[layer])):
                for connection in range(len(self.network_topology[layer][node].connections_out)):
                    if self.network_topology[layer][node].connections_out[connection].enabled:
                        node_number = self.network_topology[layer][node].connections_out[connection].output_node_number
                        node_layer = self.network_topology[layer][node].connections_out[connection].output_node_layer
                        weight = self.network_topology[layer][node].connections_out[connection].weight

                        for element in range(len(self.network_topology[node_layer])):
                            if self.network_topology[node_layer][element].number == node_number:
                                delta = self.network_topology[node_layer][element].delta
                        
                        self.network_topology[layer][node].delta += weight * delta

            self.network_topology[layer][node].delta *= self.relu2deriv(self.network_topology[layer][node].value)  

        for layer in range(len(self.network_topology)-2, -1, -1):
            for node in range(len(self.network_topology[layer])):
                for connection in range(len(self.network_topology[layer][node].connections_out)):
                    node_number = self.network_topology[layer][node].connections_out[connection].output_node_number
                    node_layer = self.network_topology[layer][node].connections_out[connection].output_node_layer

                    for element in range(len(self.network_topology[node_layer])):
                        if self.network_topology[node_layer][element].number == node_number:
                            delta = self.network_topology[node_layer][element].delta
                    
                    change = self.network_topology[layer][node].value * delta
                
                    self.network_topology[layer][node].connections_out[connection].weight -= self.lr * change

    def forward_learn(self, state, run, agent):
        for i in range(len(agent.network)):
            for j in range(len(agent.network[i])):
                agent.network[i][j].value = 0

        for i in range(len(agent.network[0])):
            agent.network[0][i].value = state[0][i]
        
        for i in range(len(agent.network)-1):
            for j in range(len(agent.network[i])):
                for k in range(len(agent.network[i][j].connections_out)):
                    if agent.network[i][j].connections_out[k].enabled == True:
                        output_node_layer = agent.network[i][j].connections_out[k].output_node_layer
                        output_node_number = agent.network[i][j].connections_out[k].output_node_number
                        weight = agent.network[i][j].connections_out[k].weight

                        for p in range(len(agent.network[output_node_layer])):
                            if agent.network[output_node_layer][p].number == output_node_number:
                                agent.network[output_node_layer][p].value += agent.network[i][j].value * weight

            if i != 0 and i != len(agent.network)-1:
                for j in range(len(agent.network[i])):
                        self.network_topology[i][j].value = self.ReLU(self.network_topology[i][j].value)

                        if run == True:
                            self.forward_run[i][j] = self.ReLU(self.network_topology[i][j].value)

        last_values = []
        value_array = []
        maximum_index = len(self.network_topology)-1
        for i in range(len(self.network_topology[maximum_index])):
            value_array.append(self.network_topology[maximum_index][i].value)

        item = np.array(value_array)
        normalized_value = np.interp(item, (item.min(), item.max()), (-1,+1))
        last_values.append(normalized_value)

        return last_values

    def backprop_learn(self, mse, agent):
        for layer in range(len(agent.network)):
            for node in range(len(agent.network[layer])):
                agent.network[layer][node].delta = 0

        last_layer = len(self.network_topology)-1
        for node in range(len(self.network_topology[last_layer])):
            agent.network[last_layer][node].delta = mse * self.forward_run[last_layer][node]

        for layer in range(len(agent.network)-2, -1, -1):
            for node in range(len(agent.network[layer])):
                for connection in range(len(self.network_topology[layer][node].connections_out)):
                    if agent.network[layer][node].connections_out[connection].enabled:
                        node_number = agent.network[layer][node].connections_out[connection].output_node_number
                        node_layer = agent.network[layer][node].connections_out[connection].output_node_layer
                        weight = agent.network[layer][node].connections_out[connection].weight

                        for element in range(len(agent.network[node_layer])):
                            if agent.network[node_layer][element].number == node_number:
                                delta = agent.network[node_layer][element].delta
                        
                        agent.network[layer][node].delta += weight * delta

            agent.network[layer][node].delta *= self.relu2deriv(self.forward_run[layer][node]) 

        for layer in range(len(agent.network)-2, -1, -1):
            for node in range(len(agent.network[layer])):
                for connection in range(len(agent.network[layer][node].connections_out)):
                    node_number = agent.network[layer][node].connections_out[connection].output_node_number
                    node_layer = agent.network[layer][node].connections_out[connection].output_node_layer

                    for element in range(len(agent.network[node_layer])):
                        if agent.network[node_layer][element].number == node_number:
                            delta = agent.network[node_layer][element].delta
                    
                    change = self.forward_run[layer][node] * delta
                
                    agent.network[layer][node].connections_out[connection].weight -= self.second_lr * change

    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()

        for i in range(BATCH_SIZE):
            q_value = self.forward_learn([states[i]], True, self.action_agent)[actions[i]]
            next_q_value = self.forward_learn([states_[i]], False, self.target_agent)
            action_ = np.argmax(self.forward_learn([states_[i]], False, self.action_agent))
            action_ = next_q_value[action_]

            q_target = rewards[i] + GAMMA * action_ * dones[i]
            td = q_target - q_value
            
            loss = ((td ** 2.0)).mean()
            self.backprop_learn(loss, self.action_agent)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        if self.learn_step_counter % self.net_copy_interval == 0:
            self.target_agent = deepcopy(self.action_agent)

        self.learn_step_counter += 1

    def choose_action(self, state):
        last_values = self.forward(state)
        action = np.argmax(last_values)
        return last_values

    def ReLU(self, x):
        #return x
        #return math.tanh(x)
        return x * (x > 0)
    
    def relu2deriv(self, input):
        #return input
        #return 1/(math.cos(input)**2)
        return int(input > 0)

    def printing_stats(self):
        # this was just for me to print out the entire network and all the connections to see that everything works
        for i in range(len(self.network_topology)):
            print("--")
            print("layer {}".format(i))
            for j in range(len(self.network_topology[i])):
                print("node {} layer {} number {}".format(self.network_topology[i][j], self.network_topology[i][j].layer_number, self.network_topology[i][j].number))

                for k in range(len(self.network_topology[i][j].connections_out)):
                    print("layer {} number {} innovation {} enabled {}".format(self.network_topology[i][j].connections_out[k].output_node_layer, self.network_topology[i][j].connections_out[k].output_node_number, self.network_topology[i][j].connections_out[k].innovation, self.network_topology[i][j].connections_out[k].enabled))
