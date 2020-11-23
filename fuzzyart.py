import torch

def minmax_scaler(x):
    _min = x.min()
    return (x - _min) / (x.max() - _min)

def range_shift_sym_to_assym(x, border=1):
    return (x+abs(border)) * 0.5

class FuzzyART(object):
    def __init__(self, rho=0.5,  alpha=0.01, learning_rate=0.1,
                 restrict_nodes=15, complement_coding=True,
                 input_range_shift_type = None):       
        self.alpha = alpha  # choice parameter
        self.rho = rho  # vigilance
        self.lr = learning_rate
        
        self._restrict_nodes = max(0, restrict_nodes)
        self._complement_coding = complement_coding
        self._init_shift_input_range(input_range_shift_type)
        self.reset()

    def reset(self):
        self._n_nodes = 0
        self._allow_node_grow = True
        self.weights = None


    @property
    def n_nodes(self):
        if self.weights is None:
            return 0
        return self.weights.shape[0]

    @property
    def node_dim(self):
        if self.weights is None:
            return 0
        return self.weights.shape[-1]

    def _init_shift_input_range(self, stype):
        if stype is None:
            self._preproc_func = lambda x: x
        elif stype == 'simple':
            self._preproc_func = lambda x: range_shift_sym_to_assym(x,1)
        elif stype == 'minmax':
            self._preproc_func = minmax_scaler
        else:
            raise ValueError(f'input_range_shift_type: {stype} not defined. Allowed: None, "simple", "minmax"')

    def preproc_input(self, inp):
        inp = inp.squeeze().detach()
        assert len(inp.shape) == 1, f'Input tensor should be 1D, after squeezing, but got {inp.shape}'
        inp = self._preproc_func(inp)
        return torch.cat((inp, 1-inp)) if self._complement_coding else inp

    def set_weight(self, i, val):
        if self.weights is None:
            if self._restrict_nodes:
                self.weights = torch.zeros(self._restrict_nodes, val.shape[-1], requires_grad=False)
            else:
                self.weights = torch.tensor([], requires_grad=False)
        if self._restrict_nodes:
            if i == -1 and (self._n_nodes < self._restrict_nodes):
                self._n_nodes += 1
                i = self._n_nodes - 1
            if self._n_nodes >= self._restrict_nodes:
                self._allow_node_grow = False
        
        if not self._allow_node_grow and i == -1:
            return
        if i == -1:
            self.weights = torch.cat((self.weights,val.unsqueeze(0)), dim=0)
        else:
            assert i < self.n_nodes, f'index {i} exceeds node count {self.n_nodes}'
            assert val.shape[-1] == self.node_dim, f'Dim of val {val.shape[-1]} not correspond to node dim {self.node_dim}'
            self.weights[i] = val

    def category_activation(self, inp):
        if self.weights is None:
            best_score = torch.norm(inp,p=1) / (self.alpha + inp.shape[-1])
            return -1, best_score.item()
        F1act_norm = torch.norm(torch.min(self.weights, inp), p=1, dim=-1)
        scores = F1act_norm / (self.alpha + torch.norm(self.weights, p=1, dim=-1))
        match = F1act_norm / torch.norm(inp, p=1)
        best_score = torch.max(scores).item()
        thresholds = (match >= self.rho).int()
        J = -1
        if thresholds.sum() != 0:
            J = torch.argmax(scores*thresholds).item()
        return J, best_score


    def infer(self, x, rtl=True):
        x = self.preproc_input(x)
        J, best_score = self.category_activation(x)
        if not rtl:
            return J, best_score
        new_x = x
        if J >= 0:
            new_x = (1-self.lr) * self.weights[J] +\
                    self.lr * torch.min(self.weights[J],x)
        self.set_weight(J, new_x)
        return J, best_score