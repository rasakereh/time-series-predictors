import numpy as np
import math
from .MF.mondrianforest import MondrianForest as MondrianForest_Main
from .MF.mondrianforest_utils import precompute_minimal
from .utils import rolling_window

def prepareData(X, f, settings, single=False):
    data = {
        'x_train': X,
        'y_train': f,
        'n_dim': X.shape[1],
        'n_train': f.shape[0],
        'x_test': None,
        'y_test': None,
        'n_test': 0,
        'is_sparse': False
    }
    # ------ beginning of hack ----------
    is_mondrianforest = True
    n_minibatches = 1 if single else settings.n_minibatches
    if is_mondrianforest:
        # creates data['train_ids_partition']['current'] and data['train_ids_partition']['cumulative'] 
        #    where current[idx] contains train_ids in minibatch "idx", cumulative contains train_ids in all
        #    minibatches from 0 till idx  ... can be used in gen_train_ids_mf or here (see below for idx > -1)
        data['train_ids_partition'] = {'current': {}, 'cumulative': {}}
        train_ids = np.arange(data['n_train'])
        try:
            draw_mondrian = settings.draw_mondrian
        except AttributeError:
            draw_mondrian = False
        # if is_mondrianforest and (not draw_mondrian):
            # reset_random_seed(settings)
            # np.random.shuffle(train_ids)
            # NOTE: shuffle should be the first call after resetting random seed
            #       all experiments would NOT use the same dataset otherwise
        train_ids_cumulative = np.arange(0)
        n_points_per_minibatch = data['n_train'] / n_minibatches
        assert n_points_per_minibatch > 0
        idx_base = np.arange(n_points_per_minibatch)
        for idx_minibatch in range(n_minibatches):
            is_last_minibatch = (idx_minibatch == n_minibatches - 1)
            idx_tmp = idx_base + idx_minibatch * n_points_per_minibatch
            if is_last_minibatch:
                # including the last (data[n_train'] % settings.n_minibatches) indices along with indices in idx_tmp
                idx_tmp = np.arange(idx_minibatch * n_points_per_minibatch, data['n_train'])
            idx_tmp = list(map(int, idx_tmp))
            train_ids_current = train_ids[idx_tmp]
            # print idx_minibatch, train_ids_current
            data['train_ids_partition']['current'][idx_minibatch] = train_ids_current
            train_ids_cumulative = np.append(train_ids_cumulative, train_ids_current)
            data['train_ids_partition']['cumulative'][idx_minibatch] = train_ids_cumulative
    return data

class Settings:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class MondrianForest:
    def __init__(
        self,
        future_scope=3,
        dimension=10,
        later_values=None
    ):
        self.future_scope = future_scope
        self.dimension = dimension
        self.settings = Settings(**{
            "bagging": 0,
            "budget": -1,
            "budget_to_use": float('inf'),
            "data_path": "../../process_data/",
            "dataset": "msg-4dim",
            "debug": 0,
            "discount_factor": 10,
            "draw_mondrian": 0,
            "init_id": 1,
            "min_samples_split": 2,
            "batch_size": 128,
            "n_minibatches": 10,
            "n_mondrians": 10,
            "name_metric": "mse",
            "normalize_features": 0,
            "op_dir": "results",
            "optype": "real",
            "perf_dataset_keys": ["train", "test"],
            "perf_metrics_keys": ["log_prob", "mse"],
            "perf_store_keys": ["pred_mean", "pred_prob"],
            "save": 0,
            "select_features": 0,
            "smooth_hierarchically": 1,
            "store_every": 0,
            "tag": "",
            "verbose": 1,
        })
        if later_values is not None:
            self.laterX = later_values['X']
            self.laterF = later_values['f']
        else:
            self.laterX = None
            self.laterF = None
    
    def fit(self, price_indices):
        self.models = {}
        self.aux_data = {}
        for capital_name in price_indices:
            trainPortion = 1
            curr_prices = price_indices[capital_name]
            seq_price = price_indices[capital_name].copy()
            seq_price[1:] = (seq_price[1:] - seq_price[:-1]) / seq_price[:-1]
            seq_price[0] = 0
            X = rolling_window(seq_price, self.dimension)
            padding = np.ones((self.dimension-1, self.dimension))
            X = np.vstack((padding, X))
            cum_sum = np.cumsum(curr_prices)
            moving_avr = curr_prices.copy()
            moving_avr[:-self.future_scope] = (cum_sum[self.future_scope:] - cum_sum[:-self.future_scope]) / self.future_scope
            f = np.zeros((curr_prices.shape[0],))
            f[:] = (moving_avr - curr_prices) / curr_prices
            ##### if update is expected
            if self.laterX is not None:
                trainSize = price_indices[capital_name].shape[0]
                testSize = self.laterX[capital_name].shape[0]
                trainPortion = trainSize / (trainSize + testSize)
                all_daily_changes = (self.laterX[capital_name][:,1:] - self.laterX[capital_name][:,:-1]) / self.laterX[capital_name][:,:-1]
                X = np.vstack([X, all_daily_changes])
                f = np.concatenate([f, self.laterF[capital_name]])
            self.settings.n_minibatches = math.ceil(f.shape[0] / self.settings.batch_size)
            data = prepareData(X, f, self.settings)
            param, cache = precompute_minimal(data, self.settings)
            self.aux_data[capital_name] = {'param': param, 'cache': cache, 'untrainedBatch': self.settings.n_minibatches, 'data': data}
            self.models[capital_name] = MondrianForest_Main(self.settings, data)
            for idx_minibatch in range(self.settings.n_minibatches):
                if idx_minibatch/self.settings.n_minibatches > trainPortion:
                    self.aux_data[capital_name]['untrainedBatch'] = idx_minibatch
                    break

                if idx_minibatch % 10 == 0:
                    print('========== %d/%d minibaches =========='%(idx_minibatch, self.settings.n_minibatches))
                train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
                if idx_minibatch == 0:
                    self.models[capital_name].fit(data, train_ids_current_minibatch, self.settings, param, cache)
                else:
                    self.models[capital_name].partial_fit(data, train_ids_current_minibatch, self.settings, param, cache)

    def predict(self, recent_prices, update=True, true_values=None, loss_functions=None):
        if true_values is None and update:
            raise Exception('True values must be provided if update parameter is set to true')

        if loss_functions is None:
            loss_functions = {'MSE': lambda truth, estimate, _prices: np.sqrt(np.mean((truth-estimate)**2))}
        
        loss_results = {}
        result = {}
        ZERO_ARR = np.array([0])
        weights_prediction = np.ones(self.settings.n_mondrians) * 1.0 / self.settings.n_mondrians
        for capital_name in recent_prices:
            result[capital_name] = np.array([])
            row_number = 0

            if recent_prices[capital_name].shape[1] != self.dimension+1:
                raise Exception('The matrix to be predicted must be of the shape (*, dimension+1)')

            all_daily_changes = (recent_prices[capital_name][:,1:] - recent_prices[capital_name][:,:-1]) / recent_prices[capital_name][:,:-1]
            for row in recent_prices[capital_name]:
                daily_changes = all_daily_changes[row_number]
                change_ratio = self.models[capital_name].evaluate_predictions({}, daily_changes.reshape((1,-1)), ZERO_ARR, self.settings, self.aux_data[capital_name]['param'], weights_prediction, False)['pred_mean'][0]
                change_ratio = self.models[capital_name].evaluate_predictions({}, daily_changes.reshape((1,-1)), np.array([true_values[capital_name][row_number]]), self.settings, self.aux_data[capital_name]['param'], weights_prediction, False)['pred_mean'][0]
                res = (change_ratio + 1) * row[-1]

                result[capital_name] = np.concatenate((result[capital_name], [res]))

                if update and ((row_number + 1) % self.settings.batch_size == 0):
                    idx_minibatch = self.aux_data[capital_name]['untrainedBatch']
                    try:
                        train_ids_current_minibatch = self.aux_data[capital_name]['data']['train_ids_partition']['current'][idx_minibatch]
                        self.models[capital_name].partial_fit(self.aux_data[capital_name]['data'], train_ids_current_minibatch, self.settings, self.aux_data[capital_name]['param'], self.aux_data[capital_name]['cache'])
                        self.aux_data[capital_name]['untrainedBatch'] += 1
                    except:
                        # idx is not aligned
                        pass
                
                row_number += 1
            
            if true_values is not None:
                loss_results[capital_name] = {}
                for loss_name in loss_functions:
                    loss_results[capital_name][loss_name] = loss_functions[loss_name](true_values[capital_name], result[capital_name], recent_prices[capital_name])
        
        return result, loss_results
