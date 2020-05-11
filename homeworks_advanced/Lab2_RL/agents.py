import torch
import torch.nn as nn
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        state_dim = state_shape[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(state_dim, 16, 4, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(64*144, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.network = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.dense
        )
        

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        qvalues = self.network(state_t)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
        
        
class DuelingQAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        state_dim = state_shape[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(state_dim, 16, 3, 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.conv_layers = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )
        self.head_V = nn.Sequential(
            nn.Linear(64*49, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.head_A = nn.Sequential(
            nn.Linear(64*49, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        conved_features = self.conv_layers(state_t)
        A = self.head_A(conved_features)
        V = self.head_V(conved_features).repeat(1, n_actions)
        mean_A = torch.mean(A, dim=-1, keepdim=True).repeat(1, n_actions)
        qvalues = A + V - mean_A

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, factorised=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.matrix = nn.Parameter(torch.Tensor(out_features, in_features))
        self.var_matrix = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.var_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.factorised = factorised
        self.reset_parameters()

    def forward(self, input):
        if self.factorised:
            eps_input = torch.randn(self.in_features, 
                                    device=self.matrix.device)
            eps_output = torch.randn(self.out_features, 
                                     device=self.matrix.device)
            eps_matrix = torch.matmul(eps_output.unsqueeze(1), 
                                      eps_input.unsqueeze(0))
        else:
            eps_output = torch.randn(self.out_features, 
                                     device=self.matrix.device)
            eps_matrix = torch.randn(self.out_features, self.in_features, 
                                     device=self.matrix.device)

        perturbed_matrix = (self.matrix + self.var_matrix * eps_matrix).t()
        if input.dim() == 2 and self.bias is not None:
            perturbed_bias = (self.bias + self.var_bias * eps_output)
            ret = torch.addmm(perturbed_bias, input, perturbed_matrix) 
        else:
            output = input.matmul(perturbed_matrix)
            if self.bias is not None:
                perturbed_bias = (self.bias + self.var_bias * eps_output)
                output += perturbed_bias
            ret = output
        return ret
    
    def reset_parameters(self):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.matrix)
        bound = 1 / math.sqrt(fan_in)
        nn.init.kaiming_uniform_(self.matrix, a=math.sqrt(5))
        nn.init.constant_(self.var_matrix, val=0.5*bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant_(self.var_bias, val=0.5*bound)
            
            
class NoisyDQNAgent(DQNAgent):
    def __init__(self, state_shape, n_actions):

        super(DQNAgent).__init__(state_shape, n_actions)
        
        self.dense = nn.Sequential(
            NoisyLinear(64*144, 256),
            nn.ReLU(),
            NoisyLinear(256, n_actions)
        )
        

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        conved_1 = self.conv1(state_t)
        conved_2 = self.conv2(conved_1)
        conved_3 = self.conv3(conved_2)
        qvalues = self.dense(conved_3)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == n_actions

        return qvalues

    def sample_actions(self, qvalues):
        """
        Pick actions given qvalues. We always take argmax since it's NoiseNet.
        """
        batch_size, n_actions = qvalues.shape
        best_actions = qvalues.argmax(axis=-1)

        return qvalues.argmax(axis=-1)