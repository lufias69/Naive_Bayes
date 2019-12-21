import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import operator
import math
pi_ = math.pi
from statistics import stdev, mean
from scipy import sparse


def vars(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))

def stds(a, axis=None):
    """ Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    return np.sqrt(vars(a, axis))

def mean_fitur(X):
    return sparse.csr_matrix(X.todense().mean(0))
    # return np.mean(X.A, axis=0)
        # mean_0 = list()
        # for i in X.transpose().A:
        #     mean_0.append(i.mean())
        # return np.array(mean_0)

def stdev_fitur(X):
    return sparse.csr_matrix(X.todense().std(0))
    # stdev_0 = list()
    # for i in X.transpose().A:
    #     stdev_0.append(stdev(i.tolist()))
    # return np.array(stdev_0)

def mean_fitur_(X):
    return sparse.csr_matrix(X.mean(0))

def stdev_fitur_(X):
    return sparse.csr_matrix(stds(X, 0))

def prior_(y):
    unik = sorted(set(y))
    dict_p = dict()
    for c in unik:
        count = y.tolist().count(c)
        dict_p.update({c:count/len(y)})
    return dict_p 

def data_separate(y, complement=False):
    kelas = sorted(set(y))
    
    dict_index = dict()
    for c in kelas:
        index = list()
        for ix, c_ in enumerate(y):
            if complement==False and c==c_:
                index.append(ix)
            elif complement==True and c!=c_:
                index.append(ix)
        dict_index.update({c:index})
    return dict_index

def get_label_(dc):
    result = list()
    x=pd.DataFrame.from_dict(dc)
    hasil = x.values
    label = x.keys().tolist()
    for i in hasil:
        result.append(label[i.tolist().index(max(i))])
    return result

def get_label(hasil, class__):
    result = list()
    label = class__#x.keys().tolist()
    for i in np.matrix(hasil).transpose().A:
        result.append(class__[i.tolist().index(max(i))])
    return result

class NaiveBayesClassifier:
    def __init__(self, alpha=1):
        self.__alpha = alpha

    @property
    def alpha(self):
        pass
    @alpha.setter
    def alpha(self, input):
        self.__alpha = input
    @alpha.getter
    def alpha(self):
        return self.__alpha
        
    def train(self, X, y):
        # self.__dict_nb = dict()
        vectorizer = CountVectorizer(binary=True)
        self.__model_w = vectorizer.fit(X)
        self.X = self.__model_w.transform(X)
        
        self.__class_ = sorted(set(y))
        self.prior = prior_(y)
        index_data = data_separate(y)
        self.__an = self.X.shape[1] #jumlah fitur
        if self.X.shape[0]!=len(y):
            return "jumlah data dan label tidak seimbang, jumlah data = "+str(self.X.shape[0])+" dan jumlah label = "+str(len(y))
 
        self.__dict_nb = dict()
        for c in self.__class_:
            n_yi = np.sum(self.X[index_data[c]], axis=0)
            # n_y = len(self.X[index_data[c]].A)
            n_y = self.X[index_data[c]].shape[0]
            self.__dict_nb.update({c:{}})
            self.__dict_nb[c]["n_yi"] = sparse.csr_matrix(n_yi)
            self.__dict_nb[c]["n_y"] = n_y
    @property
    def data(self):
        pass
    @data.getter
    def data(self):
        return self.__dict_nb 

    @property
    def classes(self):
        pass
    @classes.getter
    def classes(self):
        return self.__class_

    def predict(self, X_):
        self.X_ = self.__model_w.transform(X_)
        result = list()
        for i in self.X_.A:
            index = list()
            for ix, f in enumerate(i):
                if f>0:
                    index.append(ix)
            if len(index)==0:
                result.append(self.__class_[0])
                continue
            list_pst = list()
            for c in self.__class_:
                ny_i_ = self.__dict_nb[c]["n_yi"].A[0][index]
                # print(ny_i_)
                weight = (ny_i_+self.__alpha)/(self.__dict_nb[c]["n_y"]+self.__an)
                posterior = np.prod(weight)*self.prior[c]
                list_pst.append(posterior)
            result.append(self.__class_[list_pst.index(max(list_pst))])
        return result

    @property
    def data(self):
        pass
    @data.getter
    def data(self):
        return self.__dict_nb 

    def predict_(self, X_):
        self.X_ = self.__model_w.transform(X_)
        dict_posterior = dict()
        list_posterior = list()
        for c in self.__class_:
            ny_i_ = self.__dict_nb[c]["n_yi"].A[0]
            weight = (ny_i_+self.__alpha)/(self.__dict_nb[c]["n_y"]+self.__an)
            proud = np.prod(weight**self.X_.A, axis = 1)
            posterior = proud*self.prior[c]
            # dict_posterior.update({c:posterior})
            list_posterior.append(posterior)
        return get_label(list_posterior, self.__class_)
        

class Gaussian():
    def __init__(self,var_smoothing=.0000000001):
        self.__var_smoothing=var_smoothing

    @property
    def var_smoothing(self):
        pass
    @var_smoothing.getter
    def var_smoothing(self):
        return self.__var_smoothing
    @var_smoothing.getter
    def var_smoothing(self, input):
        self.__var_smoothing = input

    def train(self, X, y):
        self.X = X
        self.y = np.array(y)
        self.__class_ = sorted(set(y))
        self.__nb_dict=dict()

        unik = sorted(set(y))
        self.prior = prior_(y)
        for c in self.__class_:
            index = list()
            for i, c_ in enumerate(self.y):
                if c==c_:
                    index.append(i)
            self.__nb_dict.update({c:{"mean":[]}})
            self.__nb_dict.update({c:{"stdev":[]}})
            self.__nb_dict[c]["mean"] = mean_fitur_(self.X[index])
            self.__nb_dict[c]["stdev"]= stdev_fitur_(self.X[index])

    @property
    def classes(self):
        pass
    @classes.getter
    def classes(self):
        return self.__class_

    @property
    def data(self):
        pass
    @data.getter
    def data(self):
        return self.__nb_dict

    def predict(self, X):
        result = list()
        for d in X.A:
            if d.sum==0:
                result.append(self.__class_[0])
                continue
            post_ = list()
            for c in self.__class_:
                list_prob=list()
                for v, mean_, stdev_ in zip(d.tolist(), self.__nb_dict[c]["mean"].A[0],self.__nb_dict[c]["stdev"].A[0]):
                    if v != 0:
                        kiri = 1/np.sqrt((2*pi_)*((stdev_+self.__var_smoothing)**2))
                        kanan = np.exp(-(((v-mean_)**2)/(2*((stdev_+self.__var_smoothing)**2))))
                        list_prob.append(kanan*kiri)
                post_.append(np.prod(list_prob)*self.prior[c])
            result.append(self.__class_[post_.index(max(post_))])
        return np.array(result)

    def predict_(self, X):
        result = list()
        self.list_posterior = list()
        for c in self.__class_:
            sqrt_2pi = np.sqrt((2*pi_))
            stdev_ = (self.__nb_dict[c]["stdev"].A[0]+self.__var_smoothing)**2
            mean_ = self.__nb_dict[c]["mean"].A[0]
            kiri = 1/(sqrt_2pi*(stdev_))
            kanan = ((X.A-mean_)**2)/(2*stdev_)
            kanan = np.exp(-kanan)
            liklihood = (kanan*kiri)**X.A
            self.list_posterior.append(np.prod(liklihood, axis =1)*self.prior[c])
        return np.array(get_label(self.list_posterior, self.__class_))

class MultinominalNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
           
    def train(self, X, y):
        self.dict_nb = dict()
        self.class_ = sorted(set(y))
        self.prior = prior_(y)
        index_data = data_separate(y)
        self.an = len(X.A[0])
        self.X = X
        
        self.dict_nb = dict()
        for c in self.class_:
            n_yi = np.sum(self.X[index_data[c]], axis=0)
            n_y = self.X[index_data[c]].sum()
            self.dict_nb.update({c:{}})
            self.dict_nb[c]["n_y"] = n_y
            self.dict_nb[c]["n_yi"] = n_yi
            
    def predict(self, X):
        self.X = X
        result = list()
        for i in self.X.A:
            list_pst = list()
            for c in self.class_:
                hat_theta = (self.dict_nb[c]["n_yi"].A[0]+self.alpha)/(self.dict_nb[c]["n_y"]+self.an)
                posterior = np.prod(hat_theta**i)*self.prior[c]
                list_pst.append(posterior)
            result.append(self.class_[list_pst.index(max(list_pst))])
        return np.array(result)

class ComplementNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def train(self, X, y):
        self.dict_nb = dict()
        self.class_ = sorted(set(y))
        self.prior = prior_(y)
        self.index_data = data_separate(y, complement=True) #complement=False
        self.an = len(X.A[0])
        self.X = X

        self.dict_nb = dict()
        for c in self.class_:
            self.n_yi = np.sum(self.X[self.index_data[c]], axis=0)
            self.n_y = self.X[self.index_data[c]].sum()
            self.dict_nb.update({c:{}})
            hat_theta = (self.n_yi+self.alpha)/(self.n_y+self.an)
            w_ci = np.log(hat_theta)
            
            abs_sum_wci= np.sum(np.abs(w_ci))
            norm_wci = w_ci/abs_sum_wci
            self.dict_nb[c]['w_ci'] = norm_wci
            self.dict_nb[c]["n_y"] = self.n_y
            
    def set_alpha(self, alpha =1):
        self.alpha = alpha
        self.dict_nb = dict()
        for c in self.class_:
            self.n_yi = np.sum(self.X[index_data[c]], axis=0)
            self.n_y = self.X[index_data[c]].sum()
            self.dict_nb.update({c:{}})
            hat_theta = (self.n_yi+self.alpha)/(self.n_y+self.an)
            w_ci = np.log(hat_theta)
            abs_sum_wci= np.sum(np.abs(w_ci))
            norm_wci = w_ci/abs_sum_wci
            self.dict_nb[c]['w_ci'] = norm_wci
            self.dict_nb[c]["n_y"] = self.n_y
            
    def predict(self, X):
        try:
            self.X = X
            result = list()
            for i in self.X.A:
                list_pst = list()
                for c in self.class_:
                    # x = self.dict_nb[c]["w_ci"]
                    posterior = np.sum(self.dict_nb[c]["w_ci"].A[0]*i)
                    list_pst.append(posterior)
                result.append(self.class_[list_pst.index(min(list_pst))])
            return result
        except:
            return [self.class_[0]]