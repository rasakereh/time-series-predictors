import numpy as np
import faiss
from .utils import rolling_window

class MLShepard:
    def __init__(
        self,
        future_scope=3,
        dimension=10,
        minor_days=3,
        trust_treshold=4,
        max_point_usage=5,
        avr_elemwise_dist=0.04,
        epsilon=1e-10
    ):
        self.future_scope = future_scope
        self.dimension = dimension
        self.minor_days = minor_days
        self.trust_threshold = trust_treshold
        self.max_point_usage = max_point_usage
        self.avr_elemwise_dist = avr_elemwise_dist
        self.epsilon = epsilon

        self.relevance_threshold = dimension**.5 * avr_elemwise_dist
    
    def fit(self, price_indices):
        self.price_indices = {}
        for capital_name in price_indices:
            self.price_indices[capital_name] = {'X': np.array([]), 'f': np.array([]), 'data': np.array([])}
            if price_indices[capital_name].shape[0] == 0:
                continue
            curr_prices = price_indices[capital_name]
            seq_price = price_indices[capital_name].copy()
            seq_price[1:] = (seq_price[1:] - seq_price[:-1]) / seq_price[:-1]
            seq_price[0] = 0
            self.price_indices[capital_name]['data'] = seq_price
            X = rolling_window(seq_price, self.dimension)
            padding = np.ones((self.dimension-1, self.dimension)) #np.random.normal(0, self.epsilon, (self.dimension-1, self.dimension))
            X = np.vstack((padding, X))
            self.price_indices[capital_name]['X'] = faiss.IndexFlatL2(X.shape[1])
            self.price_indices[capital_name]['X'].add(X.astype(np.float32))
            cum_sum = np.cumsum(curr_prices)
            moving_avr = curr_prices.copy()
            moving_avr[:-self.future_scope] = (cum_sum[self.future_scope:] - cum_sum[:-self.future_scope]) / self.future_scope
            self.price_indices[capital_name]['f'] = np.zeros((curr_prices.shape[0],))
            self.price_indices[capital_name]['f'][:] = (moving_avr - curr_prices) / curr_prices


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
            distances, indices = self.price_indices[capital_name]['X'].search(all_daily_changes.astype(np.float32), k=self.max_point_usage)
            closeDays = distances < self.relevance_threshold
            for row in recent_prices[capital_name]:
                proximity = distances[row_number][closeDays[row_number]]
                if proximity.shape[0] < self.trust_threshold:
                    res = row[-1]
                else:
                    daily_changes = all_daily_changes[row_number]
                    currIndices = indices[row_number][closeDays[row_number]]
                    fluctuations = np.vstack([self.price_indices[capital_name]['data'][(i-self.dimension+1):(i+1)] for i in currIndices])
                    changes = self.price_indices[capital_name]['f'][currIndices]

                    general_w = self.dimension/(2*self.dimension - self.minor_days)
                    major_w = 1 - general_w

                    ws = np.dot(fluctuations, daily_changes)*general_w + np.dot(fluctuations[:, self.minor_days:], daily_changes[self.minor_days:])*major_w
                    ws /= (np.sum(ws)+self.epsilon)

                    change_ratio = np.sum(ws * changes)
                    res = (change_ratio + 1) * row[-1]

                result[capital_name] = np.concatenate((result[capital_name], [res]))

                if update:
                    newF = (true_values[capital_name][row_number] - row[-1]) / row[-1]
                    self.price_indices[capital_name]['X'].add(daily_changes.reshape((1,-1)).astype(np.float32))
                    self.price_indices[capital_name]['f'] = np.concatenate((self.price_indices[capital_name]['f'], [newF]))
                    
                
                row_number += 1
            
            if true_values is not None:
                loss_results[capital_name] = {}
                for loss_name in loss_functions:
                    loss_results[capital_name][loss_name] = loss_functions[loss_name](true_values[capital_name], result[capital_name], recent_prices[capital_name])
        
        return result, loss_results

