import numpy as np
import math
import torch
from torch import nn, optim
# from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from collections import OrderedDict
import pandas as pd 

from .RCLSTM import rclstm
from .RCLSTM.rclstm import RNN

from .utils import rolling_window

loss_fn = nn.MSELoss()

def compute_loss_accuracy(model, data, label):
    hx = None
    _, (h_n, _) = model[0](input_=data, hx=hx)
    logits = model[1](h_n[-1])
    loss = torch.sqrt(loss_fn(input=logits, target=label))
    return loss, logits

#learning rate decay
def exp_lr_scheduler(optimizer, epoch, init_lr=1e-2, lr_decay_epoch=3):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class RandConLSTM:
    def __init__(
        self,
        future_scope=3,
        dimension=10,
        epochs=2,
        batch_size=128,
        num_layers=1,
        epsilon=1e-10,
        hidden_size=100,
        connectivity=.2
    ):
        self.future_scope = future_scope
        self.dimension = dimension
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.connectivity = connectivity
    
    def fit(self, price_indices):
        self.models = {}
        for capital_name in price_indices:
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
            rnn_model = RNN(device='cpu', cell_class='rclstm', input_size=1,
                hidden_size=self.hidden_size, connectivity=self.connectivity, 
                num_layers=self.num_layers, batch_first=True, dropout=1)

            fc2 = nn.Linear(in_features=self.hidden_size, out_features=1)
            self.models[capital_name] = nn.Sequential(OrderedDict([
                    ('rnn', rnn_model),
                    ('fc2', fc2),
                    ]))

            # if use_gpu:
            #     model.cuda()
            self.models[capital_name].to('cpu')

            optim_method = optim.Adam(params=self.models[capital_name].parameters())

            iter_cnt = 0
            num_batch = int(math.ceil(len(X) // self.batch_size))
            while iter_cnt < self.epochs:
                optimizer = exp_lr_scheduler(optim_method, iter_cnt, init_lr=0.01, lr_decay_epoch=3)
                for i in range(num_batch):
                    low_index = self.batch_size*i
                    high_index = self.batch_size*(i+1)
                    if low_index <= len(X)-self.batch_size:
                        batch_inputs = X[low_index:high_index].reshape(self.batch_size, self.dimension, 1).astype(np.float32)
                        batch_targets = f[low_index:high_index].reshape((self.batch_size, 1)).astype(np.float32)
                    else:
                        batch_inputs = X[low_index:].astype(float)
                        batch_targets = f[low_index:].astype(float)

                    batch_inputs = torch.from_numpy(batch_inputs).to('cpu')
                    batch_targets = torch.from_numpy(batch_targets).to('cpu')
                    
                    # if use_gpu:
                    #     batch_inputs = batch_inputs.cuda()
                    #     batch_targets = batch_targets.cuda()

                    self.models[capital_name].train(True)
                    self.models[capital_name].zero_grad()
                    train_loss, _logits = compute_loss_accuracy(self.models[capital_name], data=batch_inputs, label=batch_targets)
                    train_loss.backward()
                    optimizer.step()
                    
                    if i % 100 == 0:
                        print('the %dth iter, the %d/%dth batch, train loss is %.4f' % (iter_cnt, i, num_batch, train_loss.item()))

                # save model
                # save_path = '{}/{}'.format(save_dir, int(round(connectivity/.01)))
                # if os.path.exists(save_path):
                #     torch.save(model, os.path.join(save_path, str(iter_cnt)+'.pt'))
                # else:
                #     os.makedirs(save_path)
                #     torch.save(model, os.path.join(save_path, str(iter_cnt)+'.pt'))
                iter_cnt += 1

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
                self.models[capital_name].train(False)
                tdc = daily_changes.reshape((1,-1,1)).astype(np.float32)
                torch_daily_changes = torch.from_numpy(tdc).to('cpu')
                lstm_out = self.models[capital_name][0](torch_daily_changes)[0]
                change_ratio = self.models[capital_name][1](lstm_out)[0].detach().numpy().reshape((-1,))[0]
                res = (change_ratio + 1) * row[-1]

                result[capital_name] = np.concatenate((result[capital_name], [res]))

                if update:
                    self.models[capital_name].train(True)
                    optim_method = optim.Adam(params=self.models[capital_name].parameters())
                    newF = (true_values[capital_name][row_number] - row[-1]) / row[-1]

                    iter_cnt = 0
                    while iter_cnt < self.epochs:
                        optimizer = exp_lr_scheduler(optim_method, iter_cnt, init_lr=0.01, lr_decay_epoch=3)
                        batch_inputs = daily_changes.reshape((1, -1, 1)).astype(np.float32)
                        batch_targets = np.array([newF]).reshape((1, 1)).astype(np.float32)
                        batch_inputs = torch.from_numpy(batch_inputs).to('cpu')
                        batch_targets = torch.from_numpy(batch_targets).to('cpu')
                        
                        self.models[capital_name].train(True)
                        self.models[capital_name].zero_grad()
                        train_loss, _logits = compute_loss_accuracy(self.models[capital_name], data=batch_inputs, label=batch_targets)
                        train_loss.backward()
                        optimizer.step()

                        iter_cnt += 1
                
                row_number += 1
            
            if true_values is not None:
                loss_results[capital_name] = {}
                for loss_name in loss_functions:
                    loss_results[capital_name][loss_name] = loss_functions[loss_name](true_values[capital_name], result[capital_name], recent_prices[capital_name])
        
        return result, loss_results



