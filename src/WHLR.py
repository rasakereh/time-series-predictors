import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.kernel_approximation import RBFSampler
from .utils import rolling_window

class WHLR:
    def __init__(
        self,
        future_scope=3,
        dimension=10,
        avr_elemwise_dist=0.04,
        learning_rate=1e-2
    ):
        self.future_scope = future_scope
        self.dimension = dimension
        self.avr_elemwise_dist = avr_elemwise_dist
        self.learning_rate = learning_rate

        self.gamma = (.33 / (dimension**.5 * avr_elemwise_dist))**2
    
    def fit(self, price_indices):
        self.models = {}
        for capital_name in price_indices:
            self.models[capital_name] = {'rbf': RBFSampler(gamma=self.gamma), 'lr': LR()}
            curr_prices = price_indices[capital_name]
            seq_price = price_indices[capital_name].copy()
            seq_price[1:] = (seq_price[1:] - seq_price[:-1]) / seq_price[:-1]
            seq_price[0] = 0
            X = rolling_window(seq_price, self.dimension)
            padding = np.ones((self.dimension-1, self.dimension)) #np.random.normal(0, self.epsilon, (self.dimension-1, self.dimension))
            X = np.vstack((padding, X))
            cum_sum = np.cumsum(curr_prices)
            moving_avr = curr_prices.copy()
            moving_avr[:-self.future_scope] = (cum_sum[self.future_scope:] - cum_sum[:-self.future_scope]) / self.future_scope
            f = np.zeros((curr_prices.shape[0],))
            f[:] = (moving_avr - curr_prices) / curr_prices
            X = self.models[capital_name]['rbf'].fit_transform(X)
            self.models[capital_name]['lr'].fit(X, f)

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
                daily_changes = all_daily_changes[row_number]
                daily_changes = self.models[capital_name]['rbf'].transform([daily_changes])[0]
                change_ratio = self.models[capital_name]['lr'].predict([daily_changes])[0]
                res = (change_ratio + 1) * row[-1]

                result[capital_name] = np.concatenate((result[capital_name], [res]))

                if update:
                    newF = (true_values[capital_name][row_number] - row[-1]) / row[-1]
                    augmentedX = np.concatenate([[1], daily_changes])
                    w = np.concatenate([[self.models[capital_name]['lr'].intercept_], self.models[capital_name]['lr'].coef_])
                    w = w - self.learning_rate * (np.dot(w, augmentedX) - newF) * augmentedX
                    self.models[capital_name]['lr'].intercept_ = w[0]
                    self.models[capital_name]['lr'].coef_ = w[1:]
                
                row_number += 1
            
            if true_values is not None:
                loss_results[capital_name] = {}
                for loss_name in loss_functions:
                    loss_results[capital_name][loss_name] = loss_functions[loss_name](true_values[capital_name], result[capital_name], recent_prices[capital_name])
        
        return result, loss_results

