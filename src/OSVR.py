"""Implementation of Online Support Vector Regression (OSVR) as library for a class project in 16-831 
Statistical Techniques in Robotics.

Requires Python 3.5

Author: Adam Werries, awerries@cmu.edu, 12/2015.
Adapted from MATLAB code available at http://onlinesvr.altervista.org/

Parameters defined in main() below. C is the regularization parameter, essentially defining the limit on how close the learner must adhere to the dataset (smoothness). Epsilon is the acceptable error, and defines the width of what is sometimes called the "SVR tube". The kernel parameter is the scaling factor for comparing feature distance (this implementation uses a Radial Basis Function). 
"""

import sys
import numpy as np
from .utils import rolling_window
from time import time

def sign(x):
    """ Returns sign. Numpys sign function returns 0 instead of 1 for zero values. :( """
    if x >= 0:
        return 1
    else:
        return -1

class OnlineSVR:
    def __init__(self, numFeatures, C, eps, kernelParam, bias = 0, debug = False):
        # Configurable Parameters
        self.numFeatures = numFeatures
        self.C = C
        self.eps = eps
        self.kernelParam = kernelParam
        self.bias = bias
        self.debug = debug
        
        print('SELF',self.C,self.eps,self.kernelParam, file=sys.stderr)
        # Algorithm initialization
        self.numSamplesTrained = 0
        self.weights = np.array([])
        
        # Samples X (features) and Y (truths)
        self.X = list()
        self.Y = list()
        # Working sets, contains indices pertaining to X and Y
        self.supportSetIndices = list()
        self.errorSetIndices = list()
        self.remainderSetIndices = list()
        self.R = np.matrix([])

    def findMinVariation(self, H, beta, gamma, i):
        """ Finds the variations of each sample to the new set.
        Lc1: distance of the new sample to the SupportSet
        Lc2: distance of the new sample to the ErrorSet
        Ls(i): distance of the support samples to the ErrorSet/RemainingSet
        Le(i): distance of the error samples to the SupportSet
        Lr(i): distance of the remaining samples to the SupportSet
        """
        # Find direction q of the new sample
        q = -sign(H[i])
        # Compute variations
        Lc1 = self.findVarLc1(H, gamma, q, i)
        q = sign(Lc1)
        Lc2 = self.findVarLc2(H, q, i)
        Ls = self.findVarLs(H, beta, q)
        Le = self.findVarLe(H, gamma, q)
        Lr = self.findVarLr(H, gamma, q)
        # Check for duplicate minimum values, grab one with max gamma/beta, set others to inf
        # Support set
        if Ls.size > 1:
            minS = np.abs(Ls).min()
            results = np.array([k for k,val in enumerate(Ls) if np.abs(val)==minS])
            if len(results) > 1:
                betaIndex = beta[results+1].argmax()
                Ls[results] = q*np.inf
                Ls[results[betaIndex]] = q*minS
        # Error set
        if Le.size > 1:
            minE = np.abs(Le).min()
            results = np.array([k for k,val in enumerate(Le) if np.abs(val)==minE])
            if len(results) > 1:
                errorGamma = gamma[self.errorSetIndices]
                gammaIndex = errorGamma[results].argmax()
                Le[results] = q*np.inf
                Le[results[gammaIndex]] = q*minE
        # Remainder Set
        if Lr.size > 1:
            minR = np.abs(Lr).min()
            results = np.array([k for k,val in enumerate(Lr) if np.abs(val)==minR])
            if len(results) > 1:
                remGamma = gamma[self.remainderSetIndices]
                gammaIndex = remGamma[results].argmax()
                Lr[results] = q*np.inf
                Lr[results[gammaIndex]] = q*minR
        
        # Find minimum absolute variation of all, retain signs. Flag determines set-switching cases.
        minLsIndex = np.abs(Ls).argmin()
        minLeIndex = np.abs(Le).argmin()
        minLrIndex = np.abs(Lr).argmin()
        minIndices = [None, None, minLsIndex, minLeIndex, minLrIndex]
        minValues = np.array([Lc1, Lc2, Ls[minLsIndex], Le[minLeIndex], Lr[minLrIndex]])

        if np.abs(minValues).min() == np.inf:
            print('No weights to modify! Something is wrong.', file=sys.stderr)
            sys.exit()
        flag = np.abs(minValues).argmin()
        if self.debug:
            print('MinValues',minValues, file=sys.stderr)
        return minValues[flag], flag, minIndices[flag]

    def findVarLc1(self, H, gamma, q, i):
        # weird hacks below
        Lc1 = np.nan
        if gamma.size < 2:
            g = gamma
        else:
            g = gamma.item(i)
        # weird hacks above

        if  g <= 0:
            Lc1 = np.array(q*np.inf)
        elif H[i] > self.eps and -self.C < self.weights[i] and self.weights[i] <= 0:
            Lc1 = (-H[i] + self.eps) / g
        elif H[i] < -self.eps and 0 <= self.weights[i] and self.weights[i] <= self.C:
            Lc1 = (-H[i] - self.eps) / g
        else:
            print('Something is weird.', file=sys.stderr)
            print('i',i, file=sys.stderr)
            print('q',q, file=sys.stderr)
            print('gamma',gamma, file=sys.stderr)
            print('g',g, file=sys.stderr)
            print('H[i]',H[i], file=sys.stderr)
            print('weights[i]',self.weights[i], file=sys.stderr)
        
        if np.isnan(Lc1):
            Lc1 = np.array(q*np.inf)
        return Lc1.item()

    def findVarLc2(self, H, q, i):
        if len(self.supportSetIndices) > 0:
            if q > 0:
                Lc2 = -self.weights[i] + self.C
            else:
                Lc2 = -self.weights[i] - self.C
        else:
            Lc2 = np.array(q*np.inf)
        if np.isnan(Lc2):
            Lc2 = np.array(q*np.inf)
        return Lc2

    def findVarLs(self, H, beta, q):
        if len(self.supportSetIndices) > 0 and len(beta) > 0:
            Ls = np.zeros([len(self.supportSetIndices),1])
            supportWeights = self.weights[self.supportSetIndices]
            supportH = H[self.supportSetIndices]
            for k in range(len(self.supportSetIndices)):
                if q*beta[k+1] == 0:
                    Ls[k] = q*np.inf
                elif q*beta[k+1] > 0:
                    if supportH[k] > 0:
                        if supportWeights[k] < -self.C:
                            Ls[k] = (-supportWeights[k] - self.C) / beta[k+1]
                        elif supportWeights[k] <= 0:
                            Ls[k] = -supportWeights[k] / beta[k+1]
                        else:
                            Ls[k] = q*np.inf
                    else:
                        if supportWeights[k] < 0:
                            Ls[k] = -supportWeights[k] / beta[k+1]
                        elif supportWeights[k] <= self.C:
                            Ls[k] = (-supportWeights[k] + self.C) / beta[k+1]
                        else:
                            Ls[k] = q*np.inf
                else:
                    if supportH[k] > 0:
                        if supportWeights[k] > 0:
                            Ls[k] = -supportWeights[k] / beta[k+1]
                        elif supportWeights[k] >= -self.C:
                            Ls[k] = (-supportWeights[k] - self.C) / beta[k+1]
                        else:
                            Ls[k] = q*np.inf
                    else:
                        if supportWeights[k] > self.C:
                            Ls[k] = (-supportWeights[k] + self.C) / beta[k+1]
                        elif supportWeights[k] >= self.C:
                            Ls[k] = -supportWeights[k] / beta[k+1]
                        else:
                            Ls[k] = q*np.inf
        else:
            Ls = np.array([q*np.inf])

        # Correct for NaN
        Ls[np.isnan(Ls)] = q*np.inf
        if Ls.size > 1:
            Ls.shape = (len(Ls),1)
            # Check for broken signs
            for val in Ls:
                if sign(val) == -sign(q) and val != 0:
                    print('Sign mismatch error in Ls! Exiting.', file=sys.stderr)
                    sys.exit()
        # print('findVarLs',Ls, file=sys.stderr)
        return Ls
        
    def findVarLe(self, H, gamma, q):
        if len(self.errorSetIndices) > 0:
            Le = np.zeros([len(self.errorSetIndices),1])
            errorGamma = gamma[self.errorSetIndices]
            errorWeights = self.weights[self.errorSetIndices]
            errorH = H[self.errorSetIndices]
            for k in range(len(self.errorSetIndices)):
                if q*errorGamma[k] == 0:
                    Le[k] = q*np.inf
                elif q*errorGamma[k] > 0:
                    if errorWeights[k] > 0:
                        if errorH[k] < -self.eps:
                            Le[k] = (-errorH[k] - self.eps) / errorGamma[k]
                        else:
                            Le[k] = q*np.inf
                    else:
                        if errorH[k] < self.eps:
                            Le[k] = (-errorH[k] + self.eps) / errorGamma[k]
                        else:
                            Le[k] = q*np.inf
                else:
                    if errorWeights[k] > 0:
                        if errorH[k] > -self.eps:
                            Le[k] = (-errorH[k] - self.eps) / errorGamma[k]
                        else:
                            Le[k] = q*np.inf
                    else:
                        if errorH[k] > self.eps:
                            Le[k] = (-errorH[k] + self.eps) / errorGamma[k]
                        else:
                            Le[k] = q*np.inf
        else:
            Le = np.array([q*np.inf])

        # Correct for NaN
        Le[np.isnan(Le)] = q*np.inf
        if Le.size > 1:
            Le.shape = (len(Le),1)
            # Check for broken signs
            for val in Le:
                if sign(val) == -sign(q) and val != 0:
                    print('Sign mismatch error in Le! Exiting.', file=sys.stderr)
                    sys.exit()
        # print('findVarLe',Le, file=sys.stderr)
        return Le

    def findVarLr(self, H, gamma, q):
        if len(self.remainderSetIndices) > 0:
            Lr = np.zeros([len(self.remainderSetIndices),1])
            remGamma = gamma[self.remainderSetIndices]
            remH = H[self.remainderSetIndices]
            for k in range(len(self.remainderSetIndices)):
                if q*remGamma[k] == 0:
                    Lr[k] = q*np.inf
                elif q*remGamma[k] > 0:
                    if remH[k] < -self.eps:
                        Lr[k] = (-remH[k] - self.eps) / remGamma[k]
                    elif remH[k] < self.eps:
                        Lr[k] = (-remH[k] + self.eps) / remGamma[k]
                    else:
                        Lr[k] = q*np.inf
                else:
                    if remH[k] > self.eps:
                        Lr[k] = (-remH[k] + self.eps) / remGamma[k]
                    elif remH[k] > -self.eps:
                        Lr[k] = (-remH[k] - self.eps) / remGamma[k]
                    else:
                        Lr[k] = q*np.inf
        else:
            Lr = np.array([q*np.inf])

        # Correct for NaN
        Lr[np.isnan(Lr)] = q*np.inf
        if Lr.size > 1:
            Lr.shape = (len(Lr),1)
            # Check for broken signs
            for val in Lr:
                if sign(val) == -sign(q) and val != 0:
                    print('Sign mismatch error in Lr! Exiting.', file=sys.stderr)
                    sys.exit()
        # print('findVarLr',Lr, file=sys.stderr)
        return Lr

    def computeKernelOutput(self, set1, set2):
        """Compute kernel output. Uses a radial basis function kernel."""
        X1 = np.matrix(set1)
        X2 = np.matrix(set2).T
        # Euclidean distance calculation done properly
        [S,R] = X1.shape
        [R2,Q] = X2.shape
        X = np.zeros([S,Q])
        if Q < S:
            copies = np.zeros(S,dtype=int)
            for q in range(Q):
                if self.debug:
                    print('X1',X1, file=sys.stderr)
                    print('X2copies',X2.T[q+copies,:], file=sys.stderr)
                    print('power',np.power(X1-X2.T[q+copies,:],2), file=sys.stderr)
                xsum = np.sum(np.power(X1-X2.T[q+copies,:],2),axis=1)
                xsum.shape = (xsum.size,)
                X[:,q] = xsum
        else:
            copies = np.zeros(Q,dtype=int)
            for i in range(S):
                X[i,:] = np.sum(np.power(X1.T[:,i+copies]-X2,2),axis=0)
        X = np.sqrt(X)
        y = np.matrix(np.exp(-self.kernelParam*X**2))
        if self.debug:
            print('distance',X, file=sys.stderr)
            print('kernelOutput',y, file=sys.stderr)
        return y
    
    def predict(self, newSampleX):
        X = np.array(self.X)
        newX = np.array(newSampleX)
        weights = np.array(self.weights)
        weights.shape = (weights.size,1)
        if self.numSamplesTrained > 0:
            y = self.computeKernelOutput(X, newX)
            return (weights.T @ y).T + self.bias
        else:
            return np.zeros_like(newX) + self.bias

    def computeMargin(self, newSampleX, newSampleY):
        fx = self.predict(newSampleX)
        newSampleY = np.array(newSampleY)
        newSampleY.shape = (newSampleY.size, 1)
        if self.debug:
            print('fx',fx, file=sys.stderr)
            print('newSampleY',newSampleY, file=sys.stderr)
            print('hx',fx-newSampleY, file=sys.stderr)
        return fx-newSampleY

    def computeBetaGamma(self,i):
        """Returns beta and gamma arrays."""
        # Compute beta vector
        X = np.array(self.X)
        Qsi = self.computeQ(X[self.supportSetIndices,:], X[i,:])
        if len(self.supportSetIndices) == 0 or self.R.size == 0:
            beta = np.array([])
        else:
            beta = -self.R @ np.append(np.matrix([1]),Qsi,axis=0)
        # Compute gamma vector
        Qxi = self.computeQ(X, X[i,:])
        Qxs = self.computeQ(X, X[self.supportSetIndices,:])
        if len(self.supportSetIndices) == 0 or Qxi.size == 0 or Qxs.size == 0 or beta.size == 0:
            gamma = np.array(np.ones_like(Qxi))
        else:
            gamma = Qxi + np.append(np.ones([self.numSamplesTrained,1]), Qxs, 1) @ beta

        # Correct for NaN
        beta[np.isnan(beta)] = 0
        gamma[np.isnan(gamma)] = 0
        if self.debug:
            print('R',self.R, file=sys.stderr)
            print('beta',beta, file=sys.stderr)
            print('gamma',gamma, file=sys.stderr)
        return beta, gamma

    def computeQ(self, set1, set2):
        set1 = np.matrix(set1)
        set2 = np.matrix(set2)
        Q = np.matrix(np.zeros([set1.shape[0],set2.shape[0]]))
        for i in range(set1.shape[0]):
            for j in range(set2.shape[0]):
                Q[i,j] = self.computeKernelOutput(set1[i,:],set2[j,:])
        return np.matrix(Q)
        
    def adjustSets(self, H, beta, gamma, i, flag, minIndex):
        print('Entered adjustSet logic with flag {0} and minIndex {1}.'.format(flag,minIndex), file=sys.stderr)
        if flag not in range(5):
            print('Received unexpected flag {0}, exiting.'.format(flag), file=sys.stderr)
            sys.exit()
        # add new sample to Support set
        if flag == 0:
            print('Adding new sample {0} to support set.'.format(i), file=sys.stderr)
            H[i] = np.sign(H[i])*self.eps
            self.supportSetIndices.append(i)
            self.R = self.addSampleToR(i,'SupportSet',beta,gamma)
            return H,True
        # add new sample to Error set
        elif flag == 1: 
            print('Adding new sample {0} to error set.'.format(i), file=sys.stderr)
            self.weights[i] = np.sign(self.weights[i])*self.C
            self.errorSetIndices.append(i)
            return H,True
        # move sample from Support set to Error or Remainder set
        elif flag == 2: 
            index = self.supportSetIndices[minIndex]
            weightsValue = self.weights[index]
            if np.abs(weightsValue) < np.abs(self.C - abs(weightsValue)):
                self.weights[index] = 0
                weightsValue = 0
            else:
                self.weights[index] = np.sign(weightsValue)*self.C
                weightsValue = self.weights[index]
            # Move from support to remainder set
            if weightsValue == 0:
                print('Moving sample {0} from support to remainder set.'.format(index), file=sys.stderr)
                self.remainderSetIndices.append(index)
                self.R = self.removeSampleFromR(minIndex)
                self.supportSetIndices.pop(minIndex)
            # move from support to error set
            elif np.abs(weightsValue) == self.C:
                print('Moving sample {0} from support to error set.'.format(index), file=sys.stderr)
                self.errorSetIndices.append(index)
                self.R = self.removeSampleFromR(minIndex)
                self.supportSetIndices.pop(minIndex)
            else:
                print('Issue with set swapping, flag 2.','weightsValue:',weightsValue, file=sys.stderr)
                sys.exit()
        # move sample from Error set to Support set
        elif flag == 3: 
            index = self.errorSetIndices[minIndex]
            print('Moving sample {0} from error to support set.'.format(index), file=sys.stderr)
            H[index] = np.sign(H[index])*self.eps
            self.supportSetIndices.append(index)
            self.errorSetIndices.pop(minIndex)
            self.R = self.addSampleToR(index, 'ErrorSet', beta, gamma)
        # move sample from Remainder set to Support set
        elif flag == 4: 
            index = self.remainderSetIndices[minIndex]
            print('Moving sample {0} from remainder to support set.'.format(index), file=sys.stderr)
            H[index] = np.sign(H[index])*self.eps
            self.supportSetIndices.append(index)
            self.remainderSetIndices.pop(minIndex)
            self.R = self.addSampleToR(index, 'RemainingSet', beta, gamma)
        return H, False

    def addSampleToR(self, sampleIndex, sampleOldSet, beta, gamma):
        print('Adding sample {0} to R matrix.'.format(sampleIndex), file=sys.stderr)
        X = np.array(self.X)
        sampleX = X[sampleIndex,:]
        sampleX.shape = (sampleX.size//self.numFeatures,self.numFeatures)
        # Add first element
        if self.R.shape[0] <= 1:
            Rnew = np.ones([2,2])
            Rnew[0,0] = -self.computeKernelOutput(sampleX,sampleX)
            Rnew[1,1] = 0
        # Other elements
        else:
            # recompute beta/gamma if from error/remaining set
            if sampleOldSet == 'ErrorSet' or sampleOldSet == 'RemainingSet':
                # beta, gamma = self.computeBetaGamma(sampleIndex)
                Qii = self.computeKernelOutput(sampleX, sampleX)
                Qsi = self.computeKernelOutput(X[self.supportSetIndices[0:-1],:], sampleX)
                beta = -self.R @ np.append(np.matrix([1]),Qsi,axis=0)
                beta[np.isnan(beta)] = 0
                beta.shape = (len(beta),1)
                gamma[sampleIndex] = Qii + np.append(1,Qsi.T)@beta
                gamma[np.isnan(gamma)] = 0
                gamma.shape = (len(gamma),1)
            # add a column and row of zeros onto right/bottom of R
            r,c = self.R.shape
            Rnew = np.append(self.R, np.zeros([r,1]), axis=1)
            Rnew = np.append(Rnew, np.zeros([1,c+1]), axis=0)
            # update R
            if gamma[sampleIndex] != 0:
                # Numpy so wonky! SO WONKY.
                beta1 = np.append(beta, [[1]], axis=0)
                Rnew = Rnew + 1/gamma[sampleIndex].item()*beta1@beta1.T
            if np.any(np.isnan(Rnew)):
                print('R has become inconsistent. Training failed at sampleIndex {0}'.forma, file=sys.stderrt(sampleIndex))
                sys.exit()
        return Rnew

    def removeSampleFromR(self, sampleIndex):
        print('Removing sample {0} from R matrix.'.format(sampleIndex), file=sys.stderr)
        sampleIndex += 1
        I = list(range(sampleIndex))
        I.extend(range(sampleIndex+1,self.R.shape[0]))
        I = np.array(I)
        I.shape = (1,I.size)
        if self.debug:
            print('I',I, file=sys.stderr)
            print('RII',self.R[I.T,I], file=sys.stderr)
        # Adjust R
        if self.R[sampleIndex,sampleIndex] != 0:
            Rnew = self.R[I.T,I] - (self.R[I.T,sampleIndex]*self.R[sampleIndex,I]) / self.R[sampleIndex,sampleIndex].item()
        else:
            Rnew = np.copy(self.R[I.T,I])
        # Check for bad things
        if np.any(np.isnan(Rnew)):
            print('R has become inconsistent. Training failed removing sampleIndex {0}'.forma, file=sys.stderrt(sampleIndex))
            sys.exit()
        if Rnew.size == 1:
            print('Time to annhilate R? R:',Rnew, file=sys.stderr)
            Rnew = np.matrix([])
        return Rnew

    def learn(self, newSampleX, newSampleY):
        self.numSamplesTrained += 1
        self.X.append(newSampleX)
        self.Y.append(newSampleY)
        self.weights = np.append(self.weights,0)
        i = self.numSamplesTrained - 1 # stupid off-by-one errors
        H = self.computeMargin(self.X, self.Y)

        # correctly classified sample, skip the rest of the algorithm!
        if (abs(H[i]) <= self.eps):
            print('Adding new sample {0} to remainder set, within eps.'.format(i), file=sys.stderr)
            if self.debug:
                print('weights',self.weights, file=sys.stderr)
            self.remainderSetIndices.append(i)
            return

        newSampleAdded = False
        iterations = 0
        while not newSampleAdded:
            # Ensure we're not looping infinitely
            iterations += 1
            if iterations > self.numSamplesTrained*100:
                print('Warning: we appear to be in an infinite loop.', file=sys.stderr)
                sys.exit()
                iterations = 0
            # Compute beta/gamma for constraint optimization
            beta, gamma = self.computeBetaGamma(i)
            # Find minimum variation and determine how we should shift samples between sets
            deltaC, flag, minIndex = self.findMinVariation(H, beta, gamma, i)
            # Update weights and bias based on variation
            if len(self.supportSetIndices) > 0 and len(beta)>0:
                self.weights[i] += deltaC
                delta = beta*deltaC
                self.bias += delta.item(0)
                # numpy is wonky...
                weightDelta = np.array(delta[1:])
                weightDelta.shape = (len(weightDelta),)
                self.weights[self.supportSetIndices] += weightDelta
                H += gamma*deltaC
            else:
                self.bias += deltaC
                H += deltaC
            # Adjust sets, moving samples between them according to flag
            H,newSampleAdded = self.adjustSets(H, beta, gamma, i, flag, minIndex)
        if self.debug:
            print('weights',self.weights, file=sys.stderr)

class OSVR:
    def __init__(
        self,
        future_scope=3,
        dimension=10,
        C=10,
        kernelParam=30,
        epsilon=1e-10
    ):
        self.future_scope = future_scope
        self.dimension = dimension
        self.C = C
        self.kernelParam = kernelParam
        self.epsilon = epsilon

    
    def fit(self, price_indices):
        self.models = {}
        for capital_name in price_indices:
            times = []
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
            self.models[capital_name] = OnlineSVR(numFeatures = X.shape[1], C = self.C, eps = self.epsilon, kernelParam = self.kernelParam, bias = 0, debug = False)
            for i in range(X.shape[0]):
                if i % 20 == 0:
                    times.append((i,time()))
                if i % 40 == 0:
                    # print('%%%%%%%%%%%%%%% Data point {0} %%%%%%%%%%%%%%%'.format(i))
                    print(times)
                self.models[capital_name].learn(X[i,:], f[i])


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
                change_ratio = self.models[capital_name].predict(np.array([daily_changes]))[0]
                res = (change_ratio + 1) * row[-1]

                result[capital_name] = np.concatenate((result[capital_name], [res]))

                if update:
                    newF = (true_values[capital_name][row_number] - row[-1]) / row[-1]
                    self.models[capital_name].learn(daily_changes)
                
                row_number += 1
            
            if true_values is not None:
                loss_results[capital_name] = {}
                for loss_name in loss_functions:
                    loss_results[capital_name][loss_name] = loss_functions[loss_name](true_values[capital_name], result[capital_name], recent_prices[capital_name])
        
        return result, loss_results
