import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import operator
import math
pi_ = math.pi

from statistics import stdev, mean
def mean_fitur(X):
        mean_0 = list()
        for i in X.transpose().A:
            mean_0.append(i.mean())
        return np.array(mean_0)

def stdev_fitur(X):
    stdev_0 = list()
    for i in X.transpose().A:
        stdev_0.append(stdev(i.tolist()))
    return np.array(stdev_0)

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

def prior_(y):
    unik = sorted(set(y))
    dict_p = dict()
    for c in unik:
        count = y.tolist().count(c)
        dict_p.update({c:count/len(y)})
    return dict_p 

class MultinominalNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.dict_nb = 0
        
    def train(self, X, y):
        self.class_ = sorted(set(y))
        self.prior = prior_(y)
        index_data = data_separate(y)
        self.an = len(X.A[0])
        self.X = X
        
        self.dict_nb = dict()
        for c in self.class_:
            n_yi = np.sum(self.X[index_data[c]].A, axis=0)
            n_y = self.X[index_data[c]].A.sum()
            self.dict_nb.update({c:{}})
            self.dict_nb[c]['hat_theta'] = n_yi/(n_y+self.an)
            self.dict_nb[c]["n_y"] = n_y
            
    def predict(self, X):
        self.X = X
        result = list()
        for i in self.X.A:
            list_pst = list()
            for c in self.class_:
                laplace = self.alpha/(self.dict_nb[c]["n_y"]+self.an)
                x = self.dict_nb[c]["hat_theta"]+laplace
                posterior = np.prod(x**i)*self.prior[c]
                list_pst.append(posterior)
            result.append(self.class_[list_pst.index(max(list_pst))])
        return result



class GaussianNaiveBayes():
    def __init__(self,alpha=0):
        self.alpha=alpha
        self.prior = 0
        self.class_ = 0
        self.nb_dict=dict()

    def train(self, X, y):
        self.X = X
        self.y = y
        self.class_ = sorted(set(y))
#         self.prior =  prior_(self.y)

        unik = sorted(set(y))
        dict_p = dict()
        for c in unik:
            count = y.tolist().count(c)
            dict_p.update({c:count/len(y)})
        self.prior = dict_p
        for c in self.class_:
            index = list()
            for i, c_ in enumerate(self.y):
                if c==c_:
                    index.append(i)
            self.nb_dict.update({c:{"mean":[]}})
            self.nb_dict.update({c:{"stdev":[]}})
            self.nb_dict[c]["mean"] = mean_fitur(self.X[index])
            self.nb_dict[c]["stdev"]= stdev_fitur(self.X[index])
#             self.nb_dict[c]= "stdev":stdev_fitur(self.X[index])}

    def predict(self, X):
        self.X = X
        self.dict_posterior = dict()
        for c in self.class_:
            list_prob=list()
            for v, mean_, stdev_ in zip(self.X[0].A.tolist()[0], self.nb_dict[c]["mean"],self.nb_dict[c]["stdev"]):
#                 print(v)
                if v != 0:
                    kiri = 1/np.sqrt((2*pi_)*((stdev_+self.alpha)**2))
                    kanan = np.exp(-(((v-mean_)**2)/(2*((stdev_+self.alpha)**2))))
                    list_prob.append(kanan*kiri)
            self.dict_posterior.update({c:np.prod(list_prob)*self.prior[c]})
        return max(self.dict_posterior.items(), key=operator.itemgetter(1))[0]

class ComplementNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.dict_nb = 0
        
    def train(self, X, y):
        self.class_ = sorted(set(y))
        self.prior = prior_(y)
        index_data = data_separate(y, complement=True) #complement=False
        self.an = len(X.A[0])
        self.X = X
        
        self.dict_nb = dict()
        
        for c in self.class_:
            self.n_yi = np.sum(self.X[index_data[c]].A, axis=0)
            self.n_y = self.X[index_data[c]].A.sum()
            self.dict_nb.update({c:{}})
            hat_theta = (self.n_yi+self.alpha)/(self.n_y+self.an)
            w_ci = np.log(hat_theta)
            
            abs_sum_wci= np.sum(np.abs(w_ci))
            norm_wci = w_ci/abs_sum_wci
            self.dict_nb[c]['w_ci'] = norm_wci
            self.dict_nb[c]["n_y"] = self.n_y
            
    def set_alpha(self, alpha =1):
        self.alpha = alpha
#         self.dict_nb = dict()
        for c in self.class_:
            self.dict_nb.update({c:{}})
            hat_theta = (self.n_yi+self.alpha)/(self.n_y+self.an)
            w_ci = np.log(hat_theta)
            
            abs_sum_wci= np.sum(np.abs(w_ci))
            norm_wci = w_ci/abs_sum_wci
            self.dict_nb[c]['w_ci'] = norm_wci
#             self.dict_nb[c]["n_y"] = self.n_y
            
    def predict(self, X):
        try:
            self.X = X
            result = list()
            for i in self.X.A:
                list_pst = list()
                for c in self.class_:
                    x = self.dict_nb[c]["w_ci"]
                    posterior = np.sum(x*i)
                    list_pst.append(posterior)
                result.append(self.class_[list_pst.index(min(list_pst))])
            return result
        except:
            return [self.class_[0]]

# class NaiveBayesClassifier:
#     def __init__(self, alpha = 1):
#         self.dic_data_index = dict()
#         self.dic_data_by_class = dict()
#         self.dic_data_posterior = dict()
#         self.dic_data_prob_fitur = dict()
#         self.prior = dict()
#         self.class_ = 0
#         self.alpha = alpha
#         self.list_post = list()
        
#     def train (self, X, y):
#         self.X = X
#         self.y = y
        
#         vectorizer = CountVectorizer(binary=True)
#         self.X = vectorizer.fit_transform(X).A
#         self.fitur = vectorizer.get_feature_names()
#         self.len_fitur = len(self.fitur)
#         self.y = np.array(self.y)
#         self.class_ = sorted(set(y))
        
#         self.prior_class = dict(zip(Counter(self.y).keys(), Counter(self.y).values()))
#         self.count_doc_c = self.prior_class.copy()
#         for key, value in self.prior_class.items():
# #             self.count_doc_c.update({key:len(self.y)})
#             self.prior_class.update({key:value/len(self.y)})
        
#         #menghitung jumlah class
#         self.len_data = len(self.y)

#         #mencari index data untuk class tertentu
# #         self.dic_data_index = dict()
#         for i in self.class_:
#             isi_list = list()
#             for index, j in enumerate(self.y):
#         #         print(j)
#                 if i == j:
#                     isi_list.append(index)
#             self.dic_data_index.update({i:isi_list})
            
# #         self.dic_data_by_class = dict()
#         for key, value in self.dic_data_index.items():
#             self.dic_data_by_class.update({key:(self.X[value])})
#         # del self.dic_data_index
        
#         self.count_fitur_c = dict()
#         self.dic_data_prob_fitur = dict()
#         for c, value in self.dic_data_by_class.items():
#             count_per_doc = list()
#             prob_fitur = list()
#             sum_value  = value.sum()
#             for ix, f in enumerate(value.transpose()):
#                 count_per_doc.append(f.sum())
#                 prob_fitur.append((f.sum()/(len(value)+self.len_fitur)))

#             _dict_count_fitur = dict(zip(self.fitur, count_per_doc))
#             bobot_fitur = dict(zip(self.fitur,prob_fitur))
            
#             self.count_fitur_c.update({c:_dict_count_fitur})
#             self.dic_data_prob_fitur.update({c:bobot_fitur})
            
    
#     def predict(self, X):
#         if self.alpha <= 0:
#             raise Exception("alpha tidak boleh kurang dari atau sama dengan 0, alpha="+str(self.alpha)) 
#         X = sorted(set(X.split()))
#         try:
#             self.dict_predict = dict()
#             for c, value in self.dic_data_prob_fitur.items():

#                 bobot_fitur = list()
#                 for kata in X:
#                     if kata in value:
#                         bobot_fitur.append((self.count_fitur_c[c][kata]+self.alpha)/(self.count_doc_c[c]+self.len_fitur))
#                         self.list_post.append("P({}|{})=({}+{})/({}+{} = {})"
#                         .format(kata,c,self.count_fitur_c[c][kata],self.alpha,self.count_doc_c[c],self.len_fitur, (self.count_fitur_c[c][kata]+self.alpha)/(self.count_doc_c[c]+self.len_fitur)))
                  
#                 if len(bobot_fitur)>0:
#                     post_prior = np.prod(bobot_fitur)*self.prior_class[c]
#                     bobot_fitur_ = [str(round(x, 3)) for x in bobot_fitur]
#                     self.list_post.append("P({}|{}) = {} x {} = {}".format(c,"X",self.prior_class[c], " x ".join(bobot_fitur_), post_prior,20))
#                     self.list_post.append("===================================")
#                     self.dict_predict.update({c:post_prior})
#             return max(self.dict_predict.items(), key=operator.itemgetter(1))[0]
#         except:
#             print("err404or")
#             return max(self.prior_class.items(), key=operator.itemgetter(1))[0]


    




# #         except:
# #             print("err404or")
# #             return max(self.prior_class.items(), key=operator.itemgetter(1))[0]

# class MultinominalNaiveBayes:
#     def __init__(self,alpha = 1):
#         self.alpha = alpha
        
#     def train(self, X, y, fitur):
#         self.X = X
#         self.y = y
#         self.fitur = fitur
#         # print(X)
#         self.len_fitur = len(X.A[0])
# #         self.fitur = fitur
        
#         self.class_ = sorted(set(self.y))

#         self.prior_class = dict(zip(Counter(self.y).keys(), Counter(self.y).values()))
#         for key, value in self.prior_class.items():
#             self.prior_class.update({key:value/len(self.y)})
        
#         #menghitung jumlah class
#         self.len_data = len(self.y)

#         #mencari index data untuk class tertentu
#         self.dic_data_index = dict()
#         for i in self.class_:
#             isi_list = list()
#             for index, j in enumerate(self.y):
#         #         print(j)
#                 if i == j:
#                     isi_list.append(index)
#             self.dic_data_index.update({i:isi_list})
            
#         self.dic_data_by_class = dict()
#         for key, value in self.dic_data_index.items():
#             self.dic_data_by_class.update({key:(self.X[value])})
# #         dic_data_index

#         self.posterior = dict()
#         for c, value in self.dic_data_by_class.items():
#             prob_fitur = list()
#             sum_value  = value.sum()
#             for ix, f in enumerate(value.transpose()):
#                 prob_fitur.append((f.sum()+self.alpha)/(sum_value+self.len_fitur))
#                 print(c,str(f.sum()), "+",str(self.alpha),"/", str(sum_value), str(self.len_fitur), self.fitur[ix])
#             self.posterior.update({c:prob_fitur})
# #         print(self.posterior)

#     def predict(self, X_predict):
#         self.X_predict = X_predict
#         self.inf_dict = dict()
#         for c in self.class_: 
# #             self.sum_predict = dict()
#             self.inf_dict.update({c:np.prod(np.power(self.posterior[c],self.X_predict))*self.prior_class[c]})
#         return max(self.inf_dict.items(), key=operator.itemgetter(1))[0]

# # class ComplementNaiveBayes:
#     def __init__(self,alpha = 1):
#         self.alpha = alpha
        
#     def train(self, X, y):
#         self.X = X
#         self.y = y
#         self.len_fitur = len(X.A[0])
# #         self.fitur = fitur
        
#         self.class_ = sorted(set(self.y))
#         self.len_fitur = len(self.y)

#         #mencari index data dengan class tertentu
#         self.dic_data_index = dict()
#         for i in self.class_:
#             isi_list = list()
#             for index, j in enumerate(self.y):
#                 if i != j: #salah satu dari tahap complement
#                     isi_list.append(index)
#             self.dic_data_index.update({i:isi_list})
            
#         self.dic_data_by_class = dict()
#         for key, value in self.dic_data_index.items():
#             self.dic_data_by_class.update({key:(self.X[value])})

#         self.complement = dict()
#         for c, value in self.dic_data_by_class.items():
#             prob_wci = list()
#             sum_value  = value.sum()
# #           Hitung Complement
#             for f in value.transpose():
#                 # f = f.sum() + (self.alpha*len(self.dic_data_index[c])) / sum_value+(self.alpha*len(self.dic_data_index[c]))
#                 f = (f.sum() + self.alpha) / (sum_value+self.len_fitur)
#                 wci = np.log(f)
#                 prob_wci.append(wci)
#             print(sum(prob_wci))
#             norm_wci = np.array(prob_wci)#/np.sqrt()
#             self.complement.update({c:norm_wci})

#     def predict(self, X_predict):
#         self.X_predict = X_predict
#         self.inf_dict = dict()
#         for c in self.class_:
#             self.inf_dict.update({c:np.prod(np.power(self.complement[c],self.X_predict))})
#         return min(self.inf_dict.items(), key=operator.itemgetter(1))[0]


# import numpy as np
# import operator


# class NaiveBayesClassifier:
#     def __init__(self):
#         self.dic_data_index = dict()
#         self.dic_data_by_class = dict()
#         self.dic_data_posterior = dict()
#         self.dic_data_prob_fitur = dict()
#         self.prior = dict()
#         self.class_ = 0

#     def train (self, X, y):
# #         print(X)
#         self.X = X
#         self.y = y

#         self.X = [i.split() for i in self.X]

#         self.X = np.array(self.X)
#         self.y = np.array(self.y)
#         self.class_ = sorted(set(y))

#         for i in self.class_:
#             isi_list = list()
#             for index, j in enumerate(y):
#         #         print(j)
#                 if i == j:
#                     isi_list.append(index)
#             self.dic_data_index.update({i:isi_list})

#         for key, value in self.dic_data_index.items():
#             self.dic_data_by_class.update({key:(self.X[value])})
#             self.prior.update({key:len(value)/len(self.y)})

#         for key, value in self.dic_data_by_class.items():
#             words = list()
# #             print(value)
#             for i in value:
#                 for kata in set(i): #biner
#                     words.append(kata)
#             dict_fitur = dict()
# #             fitur_bobot = list()
#             for i in words:
#                 prob_fitur_ = words.count(i)/len(words)
#                 dict_fitur.update({i:prob_fitur_})
# #                 fitur_bobot.append(prob_fitur_)
#     #         print(np.prod(fitur_bobot))
#             self.dic_data_prob_fitur.update({key:dict_fitur})
# #             self.dic_data_posterior.update({key:np.prod(fitur_bobot)})

#     def predict(self, X):
#         X = X.split()
#         try:
#             self.dict_predict = dict()
#             for c, value in self.dic_data_prob_fitur.items():
                
#                 bobot_fitur = list()
#                 for kata in X:
#     #                 print(kata)
#                     if kata in value:
#                         # print(c,kata,value[kata])
#                         bobot_fitur.append(value[kata])
#     #                     print(c, value[kata])
#                 if len(bobot_fitur)>0:
#                     post_prior = np.prod(bobot_fitur)*self.prior[c]
#     #                 print(post_prior)
#                     self.dict_predict.update({c:post_prior})
#             return max(self.dict_predict.items(), key=operator.itemgetter(1))[0]
#         except:
#             print("err404or")
#             return max(self.prior.items(), key=operator.itemgetter(1))[0]
            