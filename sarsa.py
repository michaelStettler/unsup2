import numpy as np

class Sarsa():
    def __init__(self, state, action, network_size=20, tau=1, eta=0.001, lambda_=0.5, gamma=0.95):
        """Constant declaration"""

        x_range = [-150,30]
        x_dot_range = [-15,15]
        self.state = state
        self.old_state = state
        self.action = action
        self.old_action = action
        self.tau = tau
        self.sigma_x = (x_range[1]-x_range[0])/network_size
        self.sigma_x_dot = (x_dot_range[1]-x_dot_range[0])/network_size
        self.gamma = gamma
        self.eta = eta
        self.lambda_ = lambda_

        """Neuronal model declarations"""
        self.network = np.zeros([network_size,network_size,2])
        for x, x_value in enumerate(np.linspace(x_range[0], x_range[1], network_size)):
            for y, y_value in enumerate(np.linspace(x_dot_range[0], x_dot_range[1], network_size)):
                self.network[x,y] = [x_value,y_value]

        self.input_activation = np.zeros([network_size,network_size])
        self.Q = np.random.rand(180,30,3)
        self.weight = np.zeros([network_size,network_size,3])
        self.e = np.zeros([180,30,3])

        return

    def _update_activation(self, x, x_d):
        self.input_activation = np.exp(-np.pow(self.network[:,:,0]-x,2)/np.pow(self.sigma_x,2) -
                                       np.pow(self.network[:,:,1]-x_d,2)/np.pow(self.sigma_x_dot,2))

        return

    def _getQ(self,state,a):
        x = np.round(state[0])+150
        y = np.round(state[1])+15
        a_ = np.sign(a)+1
        return self.Q[x,y,a_]

    def _action_probability(self,state):
        """compute probability of each action choice with softmax
        """
        a0 = np.exp(self._getQ(state,-1) / float(self.tau))
        a1 = np.exp(self._getQ(state,0) / float(self.tau))
        a2 = np.exp(self._getQ(state,1) / float(self.tau))
        #return np.exp(self.Q / float(self.tau)) / const

    def _delta_td_error(self, R):
        return R+self.gamma*self._getQ(self.state,self.action) - self._getQ(self.old_state, self.old_action)

    def update(self):
        #observe r
        self._update_activation(self.state[0],self.state[1])
        #observe s'
        #todo figure out what to call for s'
        #state = ....
        self.action = np.argmax(self._action_probability())


if __name__ == "__main__":
    net = Sarsa(state=[0,0],action=0)
