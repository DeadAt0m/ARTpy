import torch

def minmax_scaler(x):
    _min = x.min()
    return (x - _min) / (x.max() - _min)

def range_shift_sym_to_assym(x, border=1):
    return (x+abs(border)) * 0.5

dist = lambda x,y: torch.norm(x-y, p=2, dim=-1)

class HSART(object):
    def __init__(self, rho=0.5, alpha=0.01, learning_rate=0.1,
                 radius_extend = 2,  restrict_nodes=15, 
                 input_range_shift_type = None):       
        self.alpha = alpha  # choice parameter
        self.radius_extend = radius_extend
        self.rho = rho  # vigilance
        self.lr = learning_rate

        self._restrict_nodes = max(0, restrict_nodes)
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
        return self.weights['centroid'].shape[0]
    
    @property
    def node_dim(self):
        if self.weights is None:
            return 0
        return self.weights['centroid'].shape[-1]

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
        return self._preproc_func(inp)
 
    def set_weight(self, i, C, rad):
        if self.weights is None:
            if self._restrict_nodes:
                self.weights = {}
                self.weights['radius'] = torch.zeros(self._restrict_nodes, requires_grad=False)
                self.weights['centroid'] = torch.empty(self._restrict_nodes, C.shape[-1], requires_grad=False)
            else:
                self.weights = {}
                self.weights['centroid'] = torch.tensor([], requires_grad=False)
                self.weights['radius'] = torch.tensor([], requires_grad=False)
        if self._restrict_nodes:
            if i == -1 and (self._n_nodes < self._restrict_nodes):
                self._n_nodes += 1
                i = self._n_nodes - 1
            if self._n_nodes >= self._restrict_nodes:
                self._allow_node_grow = False
        if not self._allow_node_grow and i == -1:
            return
        if i == -1:
            self.weights['centroid'] = torch.cat((self.weights['centroid'], C.unsqueeze(0)), dim=0)
            self.weights['radius'] = torch.cat((self.weights['radius'], torch.tensor([rad])), dim=0)
        else:
            assert i < self.n_nodes, f'index {i} exceeds node count {self.n_nodes}'
            assert C.shape[-1] == self.node_dim, f'Dim of val {C.shape[-1]} not correspond to node dim {self.node_dim}'
            self.weights['centroid'][i] = C
            self.weights['radius'][i] = rad

    def category_activation(self, inp):
        if self.weights is None:
            best_score = self.radius_extend /  (self.alpha + self.radius_extend)
            return -1, best_score
        F1act_radius = torch.max(self.weights['radius'],
                                 dist(inp,self.weights['centroid']))
        scores = (self.radius_extend - F1act_radius) / (self.radius_extend - self.weights['radius'] + self.alpha)
        match = 1 - F1act_radius/self.radius_extend
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
        new_centroid = x
        new_radius = 0
        if J >= 0:
            centroid = self.weights['centroid'][J]
            radius = self.weights['radius'][J]
            c_norm = dist(x,centroid)
            new_centroid = centroid + (x-centroid)*(1 - min(radius, c_norm)/(c_norm+1e-8))*self.lr*0.5
            new_radius = radius + (max(radius, c_norm) - radius)*self.lr*0.5
        self.set_weight(J, new_centroid, new_radius)
        return J, best_score
