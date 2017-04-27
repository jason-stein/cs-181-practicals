# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


# GOOD ALGO FOR GRAVITY 4
class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_pruned_state = None
        self.epsilon = 0.05
        self.gamma = 0.8
        self.eta = 0.2
        self.q_vals = {}
        self.action_space = [0, 1]
        self.state_space = [-2, 0, 2]
        self.vel_space = [-10, 0, 10]
        self.dist_space = [0,100]
        # self.grav_space = [0, 1]
        for k in self.action_space:
            for i in self.state_space:
                for j in self.vel_space + [100]:
                    self.q_vals[((i,j), k)] = 0
        self.epoch = 0
        self.count = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch += 1

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # State processing
        tree_top = state["tree"]["top"]
        tree_bot = state["tree"]["bot"]
        tree_mid = (tree_top + tree_bot) / 2
        tree_botq = (tree_mid + tree_bot) / 2
        tree_topq = (tree_top + tree_mid) / 2
        m_top = state["monkey"]["top"]
        m_bot = state['monkey']['bot']
        m_mid = (m_top + m_bot) / 2 
        m_vel= state['monkey']['vel']
        tree_dist = state['tree']['dist']

        # State logic
        # curr_state = (0,0,0)
        curr_state = (0,0)
        if m_top >= tree_topq:
            curr_state = (2, curr_state[1])
        elif m_bot <= tree_botq:
            curr_state = (-2, curr_state[1])
        else:
            curr_state = (0, curr_state[1])

        for i in self.vel_space[::-1]:
            if m_vel <= i:
                curr_state = (curr_state[0], i)
        if m_vel > self.vel_space[-1]:
            curr_state = (curr_state[0], 100)
        # print m_vel

        # Q-learning
        if self.last_reward:
            future = self.eta * (self.last_reward + self.gamma * np.max([self.q_vals[(curr_state, k)] for k in self.action_space]))
            self.q_vals[(self.last_pruned_state, self.last_action)] = (1-self.eta) * self.q_vals[(self.last_pruned_state, self.last_action)] + future

        if np.random.rand() > self.epsilon:
            self.last_action = self.action_space[np.argmax([self.q_vals[(curr_state, k)] for k in self.action_space])]
        else:
            self.last_action = np.random.choice(self.action_space)

        # self.last_action = new_action
        self.last_state = state
        self.last_pruned_state = curr_state

        # Reduces learning rate
        if self.epoch < 30:
            self.epsilon = self.epsilon * 0.99
        else:
            self.epsilon = 0

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

# EXPERIMENTAL ALGO FOR GRAVITY 1
class Learner2(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_pruned_state = None
        self.epsilon = 0.05
        self.gamma = 0.8
        self.eta = 0.2
        self.q_vals = {}
        self.action_space = [0, 1]
        self.state_space = [-2, -1, 0, 1, 2]
        self.vel_space = [-15, 0, 15]
        self.dist_space = [0,200]
        self.grav_space = [1, 4]
        for k in self.action_space:
            for i in self.state_space:
                for j in self.vel_space + [100]:
                    for l in self.grav_space:
                        for m in self.dist_space:
                    # for l in self.grav_space:
                            self.q_vals[((i,j,l,m), k)] = 0
        self.epoch = 0
        self.gravity_est = None
        self.last_y_vel = None
        self.count = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch += 1
        self.gravity_est = None
        self.last_y_vel = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # State processing
        tree_top = state["tree"]["top"]
        tree_bot = state["tree"]["bot"]
        tree_mid = (tree_top + tree_bot) / 2
        tree_botq = (tree_mid + tree_bot) / 2
        tree_topq = (tree_top + tree_mid) / 2
        tree_1_3 = (tree_top - tree_mid) / 3
        tree_topt = tree_bot + 2 * tree_1_3
        tree_bott = tree_bot + tree_1_3
        m_top = state["monkey"]["top"]
        m_bot = state['monkey']['bot']
        m_mid = (m_top + m_bot) / 2 
        m_vel= state['monkey']['vel']
        tree_dist = state['tree']['dist']

        # State logic
        curr_state = (0,0,1,0)
        # curr_state = (0,0)
        # if m_mid >= tree_bott and m_mid <= tree_topt:
        #     curr_state = (0, curr_state[1], curr_state[2])
        # elif m_mid > tree_topt and m_top <= tree_top:
        #     curr_state = (1, curr_state[1], curr_state[2])
        # elif m_top > tree_top:
        #     curr_state = (2, curr_state[1], curr_state[2])
        # elif m_bot < tree_bot:
        #     curr_state = (-2, curr_state[1], curr_state[2])
        # else:
        #     curr_state = (-1, curr_state[1], curr_state[2])
        if m_mid >= tree_botq and m_mid <= tree_topq:
            curr_state = (0, curr_state[1], curr_state[2], curr_state[3])
        elif m_mid >= tree_topq:
            curr_state = (2, curr_state[1], curr_state[2], curr_state[3])
        elif m_mid <= tree_botq:
            curr_state = (-2, curr_state[1], curr_state[2], curr_state[3])

        for i in self.vel_space[::-1]:
            if m_vel <= i:
                curr_state = (curr_state[0], i, curr_state[2], curr_state[3])
                # curr_state = (curr_state[0], i)
        if m_vel > self.vel_space[-1]:
            curr_state = (curr_state[0], 100, curr_state[2], curr_state[3])
            # curr_state = (curr_state[0], 100)
        # print m_vel

        for i in self.dist_space:
            if tree_dist >= i:
                curr_state = (curr_state[0], curr_state[1], curr_state[2], i)

        if self.last_y_vel and self.last_action == 0 and not self.gravity_est:
            self.gravity_est = self.last_y_vel - m_vel
            curr_state = (curr_state[0], curr_state[1], self.gravity_est, curr_state[3])
            print self.gravity_est
        if not self.last_y_vel:
            self.last_y_vel = m_vel

        # Q-learning
        if self.last_reward:
            future = self.eta * (self.last_reward + self.gamma * np.max([self.q_vals[(curr_state, k)] for k in self.action_space]))
            self.q_vals[(self.last_pruned_state, self.last_action)] = (1-self.eta) * self.q_vals[(self.last_pruned_state, self.last_action)] + future

        if np.random.rand() > self.epsilon:
            self.last_action = self.action_space[np.argmax([self.q_vals[(curr_state, k)] for k in self.action_space])]
        else:
            self.last_action = np.random.choice(self.action_space)
        # new_action = npr.rand() < 1
        new_state  = state

        # self.last_action = new_action
        self.last_state = new_state
        self.last_pruned_state = curr_state

        # if self.count == 0 and self.last_action == 1:
        #     self.count += 1
        # if self.count > 0:
        #     self.count = (self.count + 1) % 3
        #     self.last_action = 0
        # print self.count

        if self.epoch < 30:
            self.epsilon = self.epsilon * 0.99
        else:
            self.epsilon = 0

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        # if reward == -10:
        #     self.last_reward = -100
        # elif reward == -5:
        #     self.last_reward = -50
        # else:
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        print "Epoch: " + str(ii) + " Score: " + str(swing.score) + " Gravity: " + str(swing.gravity)
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 30, 1)

	# Save history. 
	np.save('hist',np.array(hist))


