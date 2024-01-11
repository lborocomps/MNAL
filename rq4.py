import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--initial_size')
parser.add_argument('--query_size')
parser.add_argument('--start_from_run')
parser.add_argument('--sota')
# parser.add_argument('--epoch')
args = parser.parse_args()

EPOCH = 18
initial_size = int(args.initial_size)
sample_size = int(args.query_size )
start_from_run = int(args.start_from_run)-1
sota = args.sota

if sota == 'Thung et al.'
  sota = 'Tu2022'
if sota == 'hbrPredictor'
  sota = 'Wu2021'
if sota == 'Ge et al.'
  sota = 'Ge2021'
if sota == 'EMBLEM'
  sota = 'Fang2015'


if sota == 'Tu2022':
  # Tu2022
  f1_flag = 'u+c' # 'uncertainty' or 'no' (aggressive undersampling)
  f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f4_flag = 'no'
  f5_flag = 'no'
  MODE = 'single'
  MODEL = 'SVM'
  name = 'Tu2022'

elif sota == 'Wu2021':
  # Wu2021
  f1_flag = 'uncertainty' # 'uncertainty' or 'no'
  f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f4_flag = 'no'
  f5_flag = 'no'
  MODE = 'single'
  MODEL = 'NBM' #(alpha = 10)
  name = 'Wu2021'

elif sota == 'Ge2021':
  # Ge2021
  f1_flag = 'uncertainty' # 'uncertainty' or 'no' (when labeled<candidate)
  f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f4_flag = 'no'
  f5_flag = 'no'
  MODE = 'single'
  MODEL = 'RF' #(100)
  name = 'Ge2021'

elif sota == 'Wang2016':
  # Wang2016
  f1_flag = 'no' # 'uncertainty' or 'no'
  f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f3_flag = 'coreset' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f4_flag = 'no'
  f5_flag = 'no'
  MODE = 'single'
  MODEL = 'SVM'
  name = 'Wang2016'

elif sota == 'Fang2015':
  # Fang2015
  f1_flag = 'uncertainty' # 'uncertainty' or 'no' (aggressive undersampling)
  f2_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f3_flag = 'no' # 'labeled' or 'unlabeled' or 'labeled+unlabeled' or 'no'
  f4_flag = 'no'
  f5_flag = 'no'
  MODE = 'single'
  MODEL = 'SVM'
  name = 'Fang2015'


sample_times = 10
test_size = 5000

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,auc,roc_curve
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

import argparse
import pandas as pd
import re
import nltk
import numpy as np
import string
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
import torch
import random
from sklearn.metrics import (accuracy_score,recall_score,precision_score,f1_score, auc, roc_curve, confusion_matrix)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import time
import datetime
from tqdm import tqdm
import math
import pickle
# import logging
import torch.nn.functional as F
import math
from abc import abstractmethod
import numpy as np
from sklearn.metrics import pairwise_distances
from transformers import BertModel
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import textstat
from collections import Counter
from sklearn.model_selection import GridSearchCV


words_to_search = ['error', 'bug', 'reproduce', 'issue', 'behavior', 'debug', 'failed', 'expected', 'crash', 'add', 'would', 'like', 'use', 'feature', 'request', 'support', 'improvement', 'want', 'documentation']

with open(f'/data_all_test_size{test_size}.pkl','rb') as pklfile:
  files = pickle.load(pklfile)
X_test, y_test, bug_train_ids, X_train, y_train = files

class Coreset_Greedy:
    def __init__(self, all_pts,mode='max'):
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = []
        self.mode = mode

        # reshape
        feature_len = self.all_pts[0].shape[0]
        self.all_pts = self.all_pts.reshape(-1,feature_len)

        # self.first_time = True

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None:
            x = self.all_pts[centers] # pick only centers
            dist = pairwise_distances(self.all_pts, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def sample(self, unlabeled_len, annotated_len, sample_size):

        # initially updating the distances
        self.update_dist(range(unlabeled_len,unlabeled_len+annotated_len), only_new=False, reset_dist=True)
        self.already_selected = range(unlabeled_len,unlabeled_len+annotated_len)

        new_batch = []
        obj = []
        # pdb.set_trace()
        for _ in range(sample_size):
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
              if self.mode == 'max':
                ind = np.argmax(self.min_distances[:unlabeled_len])
                dist = np.max(self.min_distances[:unlabeled_len])
              elif self.mode == 'min':
                ind = np.argmin(self.min_distances[:unlabeled_len])
                dist = np.min(self.min_distances[:unlabeled_len])
            assert ind not in range(unlabeled_len,unlabeled_len+annotated_len)
            self.update_dist([ind],only_new=True, reset_dist=False)
            new_batch.append(ind)
            obj.append(dist)

        return new_batch, obj

def ids2dataloader(ids,tokenized_sentences,batch_size=32):
  data = TensorDataset(tokenized_sentences['input_ids'][ids].clone().detach(), tokenized_sentences['attention_mask'][ids].clone().detach(), torch.tensor(y_train[ids]))
  sampler = SequentialSampler(data)
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  return dataloader

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def pareto_front(costs):
  """
  Find the pareto-efficient points
  :param costs: An (n_points, n_costs) array
  :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
  """
  is_front = np.ones(costs.shape[0], dtype = bool)
  for i, c in enumerate(costs):
      if is_front[i]:
          is_front[is_front] = np.any(costs[is_front]<c, axis=1)  # Keep any point with a lower cost
          is_front[i] = True  # And keep self
  return is_front

def dominated_solution(costs, n_sample):
  count = 0
  indices = np.array(range(costs.shape[0]))
  initial_indices = np.array(range(costs.shape[0]))
  initial_costs = np.copy(costs)
  while True:
    is_front = pareto_front(costs)
    last_count = count
    count += sum(1 for x in is_front if x)
    if count > n_sample:
      costs_selected = costs[is_front]
      indices_selected = indices[is_front]
      inner_indices = np.argsort(costs_selected[:,0])[:n_sample-last_count]

      if len(inner_indices) == n_sample:
        pareto_indices = indices_selected[inner_indices]
      else:
        pareto_indices = np.setdiff1d(initial_indices, indices)
        pareto_indices = np.concatenate((pareto_indices,indices_selected[inner_indices]))
      break
    elif count == n_sample:
      indices = indices[~is_front]
      pareto_indices = np.setdiff1d(initial_indices, indices)
      break
    else:
      indices = indices[~is_front]
      costs = costs[~is_front]
  return pareto_indices

def cal_dists(extreme_points,costs):
  # Example points
  point1 = extreme_points[0]
  point2 = extreme_points[1]
  point3 = extreme_points[2]
  dists = []
  # These two vectors are in the plane
  v1 = point3 - point1
  v2 = point2 - point1
  # Calculate the cross product of the two vectors
  normal = np.cross(v2, v1)
  # Get the coefficients (a, b, c, d) of the plane equation ax + by + cz + d = 0
  a, b, c = normal
  d = -np.dot(normal, point1)
  print(a,b,c,d)
  for other_point in costs:
    dist = abs((a * other_point[0] + b * other_point[1] + c * other_point[2] + d))/(math.sqrt(a * a + b * b + c * c))
    dists.append(dist)
  return np.array(dists)

def getKneePointIndices(costs,num):
  extreme_points = []
  for i in range(costs.shape[1]):
    extreme_points.append(costs[np.argmax(costs[:,i])])
  extreme_points = np.array(extreme_points)
  cost_dists = cal_dists(extreme_points,costs)
  KneePointIndices = np.argsort(-cost_dists)[:num]
  return KneePointIndices


def dominated_knee_solution(costs, n_sample):
  count = 0
  indices = np.array(range(costs.shape[0]))
  initial_indices = np.array(range(costs.shape[0]))
  initial_costs = np.copy(costs)
  while True:
    is_front = pareto_front(costs)
    last_count = count
    count += sum(1 for x in is_front if x)
    if count > n_sample:
      costs_selected = costs[is_front]
      indices_selected = indices[is_front]
      # inner_indices = np.argsort(costs_selected[:,0])[:n_sample-last_count]
      inner_indices = getKneePointIndices(costs_selected,n_sample-last_count)

      if len(inner_indices) == n_sample:
        pareto_indices = indices_selected[inner_indices]
      else:
        pareto_indices = np.setdiff1d(initial_indices, indices)
        pareto_indices = np.concatenate((pareto_indices,indices_selected[inner_indices]))
      break
    elif count == n_sample:
      indices = indices[~is_front]
      pareto_indices = np.setdiff1d(initial_indices, indices)
      break
    else:
      indices = indices[~is_front]
      costs = costs[~is_front]
  return pareto_indices

def single_id_predict(id,return_pooled_output=False):
  # tokens = tokenizer(X_train.iloc[id], max_length=MAX_LEN,truncation=True, padding=True,return_tensors="pt", return_attention_mask=True)
  # with torch.no_grad():
  #   outputs = model(tokens['input_ids'].to(device),token_type_ids=None,attention_mask=tokens['attention_mask'].to(device))
  # prob = F.softmax(outputs[0], dim=-1)[0][0].detach().cpu().numpy().item()
  tokens = tokenizer(X_train.iloc[id], max_length=MAX_LEN,truncation=True, padding=True,return_tensors="pt", return_attention_mask=True)
  with torch.no_grad():
    output = model(tokens['input_ids'].to(device),token_type_ids=None,attention_mask=tokens['attention_mask'].to(device))
  logits = linear_layer(output[1])
  prob = F.softmax(logits, dim=-1)[0][0].detach().cpu().numpy().item()
  if return_pooled_output:
    return prob,output[1][0].detach().cpu().numpy()
  else:
    return prob

def modelpredict(indices,model,tokenized_sentences,batch_size=128,verbose=0,return_pool_result=True):
  start_time = time.time()
  if verbose == 0:
    TQDM_DISABLE = True
  else:
    TQDM_DISABLE = False
  model.eval()
  dataloader = ids2dataloader(indices,tokenized_sentences,batch_size=128)
  probs = np.empty((0,2))
  if return_pool_result:
    pooled_outputs = np.zeros((len(indices),768)).astype(np.float32)
  current_idx = 0
  # Evaluate data for one epoch
  for batch in tqdm(dataloader,disable=TQDM_DISABLE):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():

          _, pooled_output = model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask,return_dict=False)
    logits = linear_layer(pooled_output)
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    probs = np.concatenate((probs,prob))
    if return_pool_result:
      # # method 1
      # pooled_outputs = np.concatenate((pooled_outputs, pooled_output.detach().cpu().numpy()), axis=0)
      # # method 2
      # pooled_outputs = np.append(pooled_outputs, pooled_output.detach().cpu().numpy())
      # method 3
      # pooled_outputs = np.concatenate((pooled_outputs, pooled_output.detach().cpu().numpy()))
      # method 4
      # pooled_outputs_list = []
      # print('pooled_output.shape',pooled_output.shape)
      # print('pooled_output',pooled_output)
      # pooled_outputs_list.append(pooled_output.detach().cpu().numpy())
      # pooled_outputs = np.concatenate(pooled_outputs_list, axis=0)
      # method 5
      batch_size = prob.shape[0]
      pooled_outputs[current_idx:current_idx+batch_size] = pooled_output.detach().cpu().numpy()
      current_idx += batch_size

    # pooled_outputs_list = []
    # pooled_outputs_list.append(pooled_output.detach().cpu().numpy())
    # pooled_outputs = np.concatenate(pooled_outputs_list, axis=0)
  
  end_time = time.time()
  execution_time = end_time - start_time
  print("model predicting time:", execution_time)
  if return_pool_result:
    return probs[:,1],pooled_outputs
  else:
    return probs[:,1]

def modeltest(model,verbose=0):
  X_test_using = np.array(tfidf.transform(X_test).todense())
  y_pred = model.predict(X_test_using)

  y_pred2 = np.zeros(y_pred.shape)
  y_pred2[y_pred>0.5] = 1
  f1 = f1_score(y_test, y_pred2)
  acc = accuracy_score(y_test, y_pred2)
  rec = recall_score(y_test, y_pred2)
  prec = precision_score(y_test, y_pred2)
  fpr, tpr, thresholds = roc_curve(y_test,y_pred,pos_label=1)
  auc_value=auc(fpr,tpr)
  return np.array([f1,acc,auc_value,rec,prec])

def reload_model():
  model = BertModel.from_pretrained(
      "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
      # num_labels = 2, # The number of output labels--2 for binary classification.
      #                 # You can increase this for multi-class tasks.
      # output_attentions = False, # Whether the model returns attentions weights.
      # output_hidden_states = False, # Whether the model returns all hidden-states.
  )

  # Tell pytorch to run this model on the GPU.
  model.cuda()
  return model

def modelfit(indices):
  tfidf = TfidfVectorizer(ngram_range=(1,1),max_features=1000)
  X_using = np.array(tfidf.fit_transform(X_train.iloc[indices]).todense())

  if MODEL == 'NBM':
    params = {'alpha': [0.01,0.1,1,10,100]}
  elif MODEL == 'SVM':
    # params = {'C':[math.pow(2,x) for x in [-5,0,5]],'gamma':[1/(2*math.pow(math.pow(2,x),2)) for x in [-5,0,5]],'kernel':['linear']}
    params = {'C':[math.pow(2,x) for x in [0]],'gamma':[1/(2*math.pow(math.pow(2,x),2)) for x in [0]],'kernel':['linear']}
  elif MODEL == 'RF':
    params = {'n_estimators':[25,50,75,100]}

  if MODEL == 'NBM':
    clf = MultinomialNB()
  elif MODEL == 'SVM':
    clf = SVC(random_state=run)
  elif MODEL == 'RF':
    clf = RandomForestClassifier(random_state=run)

  model = GridSearchCV(clf,params,cv=5,scoring='accuracy')
  model.fit(X_using, y_train[indices])

  return model,tfidf

metrics_all = np.zeros((5,sample_times+1),dtype='float64')

# first run
run = start_from_run
print(f'\nThe {run} run started\n')

indices = random.sample(range(len(bug_train_ids)),sample_size)

bug_annotated_ids = bug_train_ids[indices]
bug_pool_ids = np.delete(bug_train_ids, indices, axis=0)

model,tfidf = modelfit(bug_annotated_ids)
tmp=modeltest(model,verbose=0)
metrics_all[:,0] = metrics_all[:,0]+tmp

y_true_labeled = y_train[bug_annotated_ids]

# left run
for sample_time in range(0,sample_times):
  print(f'\nThe {sample_time} step started\n')
  predicted_ids = []
  dict_id2ypred = {}
  dict_id2poolres = {}
  dict_id2f4 = {}
  dict_id2f5 = {}
  budge_count = 0

  start_time = time.time()


  if MODE == 'random':
    indices = random.sample(range(len(bug_pool_ids)),sample_size)
    result = []
    
  elif MODE == 'dominated' or MODE == 'knee':
    costs = []

    y_pred = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=False)

    if f1_flag=='uncertainty':
      costs.append(np.abs(0.5-y_pred))
    if f4_flag=='read':
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      costs.append(-read)
    if f5_flag=='iden':
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values())/(len(text.split())+1)).values
      costs.append(-iden)
    costs = np.transpose(costs)
    if MODE == 'dominated':
      indices = dominated_solution(costs,sample_size)
    elif MODE == 'knee':
      indices = dominated_knee_solution(costs,sample_size)
    result = costs[indices]

  elif MODE == 'DBScan' or MODE == 'kmeans':

    y_pred, pool_res = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=True)

    if f1_flag=='uncertainty' or f1_flag=='u+c':
      obj = np.abs(0.5-y_pred)
    elif f1_flag=='read':
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      obj = -read
      for i in range(len(obj)):
        if obj[i] < -206.8:
          obj[i] = 9999

    elif f1_flag=='iden':
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values())/(len(text.split())+1)).values
      obj = -iden
      for i in range(len(obj)):
        if obj[i] == 0:
          obj[i] = 9999
    if f4_flag=='normalized_sum':
      cer = np.abs(0.5-y_pred)
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values())/(len(text.split())+1)).values

      costs = np.array([cer, -read, -iden]).T
      scaler = MinMaxScaler()
      scaler.fit(costs)
      normalized_costs = scaler.transform(costs)
      temp = normalized_costs.T
      obj = temp[0]+temp[1]+temp[2]
      # obj = -obj
    # clustering
    if MODE == 'kmeans':
      clustering = KMeans(n_clusters=sample_size).fit(pool_res)
    elif MODE == 'DBScan':
      clustering = DBSCAN(min_samples=sample_size).fit(pool_res)

    # Initialize dictionary to store the index of minimum objective for each cluster
    min_objective_indices = {}

    # Iterate over the label and objective arrays
    for i, (label, objective) in enumerate(zip(clustering.labels_, obj)):
        # If the cluster label is not in the dictionary, or the current objective is lower than the stored minimum
        if label not in min_objective_indices or objective < min_objective_indices[label][1]:
            # Update the dictionary with the current index and objective
            min_objective_indices[label] = (i, objective)

    # Get only the index of minimum objective for each cluster
    min_indices = {cluster: objective[0] for cluster, objective in min_objective_indices.items()}

    # Convert the dictionary values to a list and sort by cluster
    indices = [min_indices[i] for i in sorted(min_indices.keys())]
    result = obj[indices]

  elif MODE == 'single' or MODE == 'normalized_sum' or MODE == 'read+iden':

    X_temp = np.array(tfidf.transform(X_train.iloc[bug_pool_ids]).todense())
    y_pred = model.predict(X_temp)
    if f1_flag=='uncertainty' or f1_flag=='u+c':
        obj = np.abs(0.5-y_pred)
    if f1_flag=='threshold':
        obj = np.abs(0.5-y_pred)
    if f1_flag=='read':
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      obj = -read
      for i in range(len(obj)):
        if obj[i] < -206.8:
          obj[i] = 9999

    if f1_flag=='iden':
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values())/(len(text.split())+1)).values
      obj = -iden
      for i in range(len(obj)):
        if obj[i] == 0:
          obj[i] = 9999
    if f2_flag=='labeled':
        y_pred_labeled, pool_res_labeled = modelpredict(bug_annotated_ids,model,tokenized_sentences,return_pool_result=True)
        y_pred = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=False,verbose=0)
        obj = np.empty(y_pred.shape[0])
        for i,y_temp in enumerate(y_pred):
              obj[i] = np.sum(np.multiply(y_temp-y_pred_labeled,np.log(y_temp/y_pred_labeled)))
    elif f2_flag=='unlabeled':
        y_pred = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=False,verbose=0)
        obj = np.empty(y_pred.shape[0])
        for i,y_temp in enumerate(y_pred):
              obj[i] = np.sum(np.multiply(y_temp-y_pred,np.log(y_temp/y_pred)))
    elif f2_flag=='minus_labeled':
        y_pred_labeled, pool_res_labeled = modelpredict(bug_annotated_ids,model,tokenized_sentences,return_pool_result=True)
        y_pred = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=False,verbose=0)
        obj = np.empty(y_pred.shape[0])
        for i,y_temp in enumerate(y_pred):
              obj[i] = -np.sum(np.multiply(y_temp-y_pred_labeled,np.log(y_temp/y_pred_labeled)))
    elif f2_flag=='minus_unlabeled':
        y_pred = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=False,verbose=0)
        obj = np.empty(y_pred.shape[0])
        for i,y_temp in enumerate(y_pred):
              obj[i] = -np.sum(np.multiply(y_temp-y_pred,np.log(y_temp/y_pred)))
    if f3_flag=='labeled':
        y_pred, pool_res_unlabeled = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=True,verbose=0)
        y_pred_labeled, pool_res_labeled = modelpredict(bug_annotated_ids,model,tokenized_sentences,return_pool_result=True)
        obj = np.empty(y_pred.shape[0])
        for i,feature_temp_u in enumerate(pool_res_unlabeled):
              dist_u2l_list = []
              for j,feature_temp_l in enumerate(pool_res_labeled):
                  dist_u2l_list.append(np.linalg.norm(feature_temp_u-feature_temp_l))
              obj[i] = -sum(dist_u2l_list)/len(dist_u2l_list)
    elif f3_flag=='unlabeled':
        # representative_unlabeled
        y_pred, pool_res_unlabeled = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=True,verbose=0)
        dist_matrix_unlabeled = pairwise_distances(pool_res_unlabeled)
        obj = np.empty(y_pred.shape[0])
        for i in range(len(pool_res_unlabeled)):
          dist_u2u_list = dist_matrix_unlabeled[i]
          obj[i] = -sum(dist_u2u_list)/len(dist_u2u_list)
    elif f3_flag=='minus_unlabeled':
        # representative_unlabeled
        y_pred, pool_res_unlabeled = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=True,verbose=0)
        dist_matrix_unlabeled = pairwise_distances(pool_res_unlabeled)
        obj = np.empty(y_pred.shape[0])
        for i in range(len(pool_res_unlabeled)):
          dist_u2u_list = dist_matrix_unlabeled[i]
          obj[i] = sum(dist_u2u_list)/len(dist_u2u_list)
    elif f3_flag=='minus_labeled':
        y_pred, pool_res_unlabeled = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=True,verbose=0)
        y_pred_labeled, pool_res_labeled = modelpredict(bug_annotated_ids,model,tokenized_sentences,return_pool_result=True)
        obj = np.empty(y_pred.shape[0])
        for i,feature_temp_u in enumerate(pool_res_unlabeled):
              dist_u2l_list = []
              for j,feature_temp_l in enumerate(pool_res_labeled):
                  dist_u2l_list.append(np.linalg.norm(feature_temp_u-feature_temp_l))
              obj[i] = sum(dist_u2l_list)/len(dist_u2l_list)
    elif f3_flag=='BALD':
        model.train()
        dataloader = ids2dataloader(bug_pool_ids,tokenized_sentences,batch_size=128)
        # pooled_outputs = np.empty((0,768))
        # Evaluate data for one epoch

        MC_samples = np.empty((0,len(bug_pool_ids),2))
        nb_MC_samples = 50

        for _ in range(nb_MC_samples):
            probs = np.empty((0,2))

        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                _, pooled_output = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,return_dict=False)
            logits = linear_layer(pooled_output)
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            probs = np.concatenate((probs,prob))
            # pooled_outputs = np.concatenate((pooled_outputs, pooled_output.detach().cpu().numpy()), axis=0)

        MC_samples = np.concatenate((MC_samples,np.reshape(probs,(1,probs.shape[0],probs.shape[1]))))
        expected_entropy = - np.mean(np.sum(MC_samples * np.log(MC_samples + 1e-10), axis=-1), axis=0)  # [batch size]
        expected_p = np.mean(MC_samples, axis=0)
        entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
        f3_arr = -(entropy_expected_p - expected_entropy)
        obj = f3_arr
    elif f3_flag=='coreset':
      X_labeled = np.array(tfidf.transform(X_train.iloc[bug_annotated_ids]).todense())
      process_array = np.concatenate((X_temp,X_labeled))
      coreset = Coreset_Greedy(process_array, mode='max')
      indices,obj = coreset.sample(len(bug_annotated_ids),len(bug_pool_ids), sample_size)
    elif f3_flag=='minus_coreset':
      y_pred, pool_res_unlabeled = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=True,verbose=0)
      y_pred_labeled, pool_res_labeled = modelpredict(bug_annotated_ids,model,tokenized_sentences,return_pool_result=True)
      process_array = np.concatenate((pool_res_unlabeled,pool_res_labeled))
      coreset = Coreset_Greedy(process_array, mode='min')
      indices,obj = coreset.sample(len(bug_annotated_ids),len(bug_pool_ids), sample_size)
    if f4_flag=='normalized_sum':
      y_pred = modelpredict(bug_pool_ids,model,tokenized_sentences,return_pool_result=False,verbose=0)
      cer = np.abs(0.5-y_pred)
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values())/(len(text.split())+1)).values

      costs = np.array([cer, -read, -iden]).T
      scaler = MinMaxScaler()
      scaler.fit(costs)
      normalized_costs = scaler.transform(costs)
      temp = normalized_costs.T
      obj = temp[0]+temp[1]+temp[2]
    if f4_flag=='read+iden':
      read = X_train.iloc[bug_pool_ids].apply(lambda text: textstat.flesch_reading_ease(text)).values
      iden = X_train.iloc[bug_pool_ids].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values())/(len(text.split())+1)).values

      costs = np.array([-read, -iden]).T
      scaler = MinMaxScaler()
      scaler.fit(costs)
      normalized_costs = scaler.transform(costs)
      temp = normalized_costs.T
      obj = temp[0]+temp[1]

    if f3_flag=='coreset' or f3_flag=='minus_coreset':
      indices = np.array(indices)
      result = np.array(obj)
    elif f1_flag=='threshold':
      condition_indices = np.where(obj > threshold)[0]
      indices = condition_indices[np.argsort(obj[condition_indices])[:sample_size]]
      # indices = np.argsort(obj)[threshold:sample_size+threshold]
      result = obj[indices]
    elif f1_flag=='u+c':
      u_indices = np.argpartition(obj,30)[:30]
      u_result = obj[u_indices]
      complement_indices = np.setdiff1d(np.arange(len(obj)), u_indices)
      c_obj = -np.abs(0.5-y_pred[complement_indices])
      c_indices = np.argpartition(c_obj,sample_size-30)[:sample_size-30]
      result = np.append(u_result,c_obj[c_indices])
      indices = np.append(u_indices,c_indices)
    else:
      indices = np.argpartition(obj,sample_size)[:sample_size]
      result = obj[indices]

  read_values = X_train.iloc[bug_pool_ids[indices]].apply(lambda text: textstat.flesch_reading_ease(text)).values
  iden_values = X_train.iloc[bug_pool_ids[indices]].apply(lambda text: sum(Counter(word for word in words_to_search if word in text).values())/(len(text.split())+1)).values

  bug_annotated_ids = np.concatenate((bug_annotated_ids,bug_pool_ids[indices]))
  y_true_labeled = np.concatenate((y_true_labeled,y_train[bug_pool_ids[indices]]))
  saved_indices = bug_pool_ids[indices]
  bug_pool_ids = np.delete(bug_pool_ids, indices, axis=0)

  end_time = time.time()
  execution_time = end_time - start_time
  print("Annotating time:", execution_time)
  
  start_time = time.time()
  model,tfidf = modelfit(bug_annotated_ids)
  end_time = time.time()
  execution_time = end_time - start_time
  print("model fitting time:", execution_time)

  start_time = time.time()
  tmp=modeltest(model,verbose=0)
  end_time = time.time()
  execution_time = end_time - start_time
  print("model testing time:", execution_time)

  # end

  metrics_all[:,sample_time+1] = metrics_all[:,sample_time+1]+tmp
  print('metrics_all')
  print(*metrics_all[0,0:sample_time+2], sep=",")
  print(*metrics_all[1,0:sample_time+2], sep=",")
  print(*metrics_all[2,0:sample_time+2], sep=",")

  files = bug_annotated_ids, bug_pool_ids, y_true_labeled, metrics_all, saved_indices, tmp, read_values, iden_values

  print('run is ',run)

  with open(f'/rq4/data/rq4_initial_size{initial_size}_sample_size{sample_size}_run_{run+1}_step{sample_time+1}-{sample_times}_sota_{sota}.pkl','wb') as pklfile:
    pickle.dump(files, pklfile)

