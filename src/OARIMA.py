import numpy as np
import pmdarima as pm
from math import sqrt
from .utils import rolling_window

class OARIMA:
    def __init__(
        self,
        dimension=10,
        lrate=1e-2,
        epsilon=1e-10,
        method='ogd'
    ):
        self.dimension = dimension
        self.lrate = lrate
        self.epsilon = epsilon
        self.method = method
    
    def fit(self, price_indices):
        self.model = {}
        self.order = None
        with_intercept = None
        for capital_name in price_indices:
            self.model[capital_name] = {'ma': None, 'data': None, 'size': 0}
            seq_price = price_indices[capital_name].copy()
            seq_price[1:] = (seq_price[1:] - seq_price[:-1]) / seq_price[:-1]
            seq_price[0] = 0
            if self.order is None:
                model = pm.auto_arima(seq_price, seasonal=False, start_p=0, max_p=0, start_q=3, max_q=50, trace=True)
                params = model.get_params()
                self.order, _with_intercept = params['order'], params['with_intercept']
                if self.order[2] < 4:
                    self.order = (self.order[0], self.order[1], 4)
            model = pm.ARIMA(order=self.order, with_intercept=False)
            model.fit(seq_price)
            
            self.model[capital_name]['ma'] = model.maparams()
            self.model[capital_name]['data'] = seq_price[-(self.order[1] + self.order[2]):]
            self.model[capital_name]['size'] = seq_price.shape[0] - (self.order[1] + self.order[2])
        
        pass


    def predict(self, recent_prices, update=True, true_values=None, loss_functions=None):
        if true_values is None and update:
            raise Exception('True values must be provided if update parameter is set to true')

        if loss_functions is None:
            loss_functions = {'MSE': lambda truth, estimate, _prices: np.sqrt(np.mean((truth-estimate)**2))}
        
        loss_results = {}
        result = {}
        for capital_name in recent_prices:
            result[capital_name] = np.array([])
            A_trans = np.identity(self.order[2]) * self.epsilon
            row_number = 0

            if recent_prices[capital_name].shape[1] != self.dimension+1:
                raise Exception('The matrix to be predicted must be of the shape (*, dimension+1)')

            for row in recent_prices[capital_name]:
                diffed_values = self.model[capital_name]['data']
                if self.order[1]:
                    diffed_sum = diffed_values[-1]
                else:
                    diffed_sum = 0
                for _ in range(self.order[1]):
                    diffed_values = diffed_values[1:] - diffed_values[:-1]
                    diffed_sum += diffed_values[-1]
                estimate = (self.model[capital_name]['ma'] @ diffed_values)
                change_ratio = diffed_sum + estimate
                res = (change_ratio + 1) * row[-1]
                result[capital_name] = np.concatenate((result[capital_name], [res]))
                self.model[capital_name]['data'] = np.concatenate((self.model[capital_name]['data'][1:], [change_ratio]))
                self.model[capital_name]['size'] += 1

                if update:
                    exact = (true_values[capital_name][row_number] - row[-1])/row[-1]
                    self.model[capital_name]['data'][-1] = exact
                    diff = estimate - exact
                    if self.method == 'ogd':
                        s = self.model[capital_name]['size']
                        self.model[capital_name]['ma'] = self.model[capital_name]['ma'] - diffed_sum*2*diff/sqrt(row_number+s+1)*self.lrate
                    elif self.method == 'ons':
                        grad = (2*diffed_values*diff).reshape((1, -1))
                        # print(A_trans, A_trans.shape)
                        # print(grad, grad.shape)
                        A_trans = A_trans - A_trans @ grad.T @ grad @ A_trans/(1 + grad @ A_trans @ grad.T)
                        self.model[capital_name]['ma'] = self.model[capital_name]['ma'] - self.lrate * grad @ A_trans
                        self.model[capital_name]['ma'] = self.model[capital_name]['ma'].reshape((-1,))
                    self.model[capital_name]['ma'] = self.model[capital_name]['ma'] / np.sum(self.model[capital_name]['ma'])
                    
                row_number += 1
            
            if true_values is not None:
                loss_results[capital_name] = {}
                for loss_name in loss_functions:
                    loss_results[capital_name][loss_name] = loss_functions[loss_name](true_values[capital_name], result[capital_name], recent_prices[capital_name])
        
        return result, loss_results

