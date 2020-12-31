import numpy as np
from .utils import rolling_window

# parameters (except epsilon) offered by https://link.springer.com/article/10.1007/s00521-016-2314-8
class GBLM:
    def __init__(
        self,
        dimension=10,
        epsilon=5e-3,
        forgetting_rate=.59,
        p_learning_rate=.008,
        s_learning_rate=.001,
        decay_rate=.25,
        oe_penalty=-1.5,
        ue_penalty=-1.5,
        reward=1,
        epochs=1
    ):
        self.dimension = dimension
        self.epsilon = epsilon
        self.forgetting_rate = forgetting_rate
        self.p_learning_rate = p_learning_rate
        self.s_learning_rate = s_learning_rate
        self.decay_rate = decay_rate
        self.oe_penalty = oe_penalty
        self.ue_penalty = ue_penalty
        self.reward = reward
        self.epochs = epochs
    
    def fit(self, price_indices):
        self.models = {}
        initial_weight = np.random.uniform(-1, 1, self.dimension)
        initial_weight /= np.sum(initial_weight)
        for capital_name in price_indices:
            self.models[capital_name] = {'w': initial_weight, 'rlF': 0, 'rlR': 0}
            seq_price = price_indices[capital_name].copy()
            seq_price[1:] = (seq_price[1:] - seq_price[:-1]) / seq_price[:-1]
            seq_price[0] = 0
            X = rolling_window(seq_price, self.dimension)
            f = seq_price[self.dimension:]
            for _ in range(self.epochs):
                for x, y in zip(X, f):
                    w = self.models[capital_name]['w']
                    yHat = np.dot(w, x)
                    self.models[capital_name]['rlF'] = self.models[capital_name]['rlF']*self.decay_rate + yHat
                    diff = yHat - y
                    reward = self.oe_penalty if diff > self.epsilon else self.ue_penalty if diff < -self.epsilon else self.reward
                    self.models[capital_name]['rlR'] = self.models[capital_name]['rlR']*self.decay_rate + reward
                    self.models[capital_name]['w'] = self.forgetting_rate*w + self.p_learning_rate*x*self.models[capital_name]['rlF']*self.models[capital_name]['rlR']
            


    def predict(self, recent_prices, update=True, true_values=None, loss_functions=None):
        if true_values is None and update:
            raise Exception('True values must be provided if update parameter is set to true')

        if loss_functions is None:
            loss_functions = {'MSE': lambda truth, estimate, _prices: np.sqrt(np.mean((truth-estimate)**2))}
        
        loss_results = {}
        result = {}
        for capital_name in recent_prices:
            result[capital_name] = np.array([])
            row_number = 0

            if recent_prices[capital_name].shape[1] != self.dimension+1:
                raise Exception('The matrix to be predicted must be of the shape (*, dimension+1)')

            all_daily_changes = (recent_prices[capital_name][:,1:] - recent_prices[capital_name][:,:-1]) / recent_prices[capital_name][:,:-1]
            for row in recent_prices[capital_name]:
                w = self.models[capital_name]['w']
                daily_changes = all_daily_changes[row_number]
                change_ratio = np.dot(w, daily_changes)
                res = (change_ratio + 1) * row[-1]

                result[capital_name] = np.concatenate((result[capital_name], [res]))

                if update:
                    newF = (true_values[capital_name][row_number] - row[-1]) / row[-1]
                    self.models[capital_name]['rlF'] = self.models[capital_name]['rlF']*self.decay_rate + change_ratio
                    diff = change_ratio - newF
                    reward = self.oe_penalty if diff > self.epsilon else self.ue_penalty if diff < -self.epsilon else self.reward
                    self.models[capital_name]['rlR'] = self.models[capital_name]['rlR']*self.decay_rate + reward
                    self.models[capital_name]['w'] = self.forgetting_rate*w + self.s_learning_rate*daily_changes*self.models[capital_name]['rlF']*self.models[capital_name]['rlR']
                    
                row_number += 1
            
            if true_values is not None:
                loss_results[capital_name] = {}
                for loss_name in loss_functions:
                    loss_results[capital_name][loss_name] = loss_functions[loss_name](true_values[capital_name], result[capital_name], recent_prices[capital_name])
        
        return result, loss_results

