import os
import numpy as np
import pandas as pd
import random
import time
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct
from sklearn.ensemble import GradientBoostingRegressor
from itertools import product
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import itertools 

def cross_mu(q, l):
    q = list(q)
    l = list(l)
    m = random.randint(0, len(q))
    for i in random.sample(range(len(q)), m):
        q[i], l[i] = l[i], q[i]
        q_mu = random.randint(0, len(q)-1)
        q[q_mu] = np.random.uniform(-1, 1, 1).item()        
        l_mu = random.randint(0, len(l)-1)
        l[l_mu] = np.random.uniform(-1, 1, 1).item() 
    return q, l

def main():

  #####################################################
  ########## Read data and simple processing ########## 
  #####################################################

  parser = argparse.ArgumentParser(description='Configuration extrapolation with GIL/GIL+.')
  parser.add_argument('--target', type=str, default="has", help='Target hardware')
  parser.add_argument('--workload', type=str, default="als", help='Workload')
  parser.add_argument('--outcome', type=str, default="Throughput", help='Outcome metric') 
  parser.add_argument('--n_start', type=int, default=200, help='Start index (default: 200)')
  parser.add_argument('--n_step', type=int, default=20, help='k: number of configurations queried at each round (default: 20)')
  parser.add_argument('--n_iter', type=int, default=20, help='T: number of rounds (default: 20)')

  args = parser.parse_args()
 
  target     = args.target
  workload   = args.workload  
  outcome    = args.outcome
  n_start    = args.n_start  
  n_step     = args.n_step    
  n_iter     = args.n_iter   

  print("target:   {}".format(target))
  print("workload: {}".format(workload))
  print("outcome:  {}".format(outcome))
  print("n_start:  {}".format(n_start))
  print("n_step:   {}".format(n_step))
  print("n_iter:   {}".format(n_iter))

  X = pd.read_csv('data/perf/config_clean.csv', index_col=0)
  Xr = X.copy()
  X = (X-X.mean())/X.std()

  ac_dict = {'sky': 'Skylake', 'has': 'Haswell', 'stg': 'Storage'}
  sn_dict = {'sky': 0, 'has': 1, 'stg': 2}

  ac1, ac2, ac3 = 'sky', 'has', 'stg'

  rp_sky = pd.read_csv('data/perf/{}/{}.csv'.format(ac_dict[ac1], workload), index_col=0)
  rp_has = pd.read_csv('data/perf/{}/{}.csv'.format(ac_dict[ac2], workload), index_col=0)
  rp_stg = pd.read_csv('data/perf/{}/{}.csv'.format(ac_dict[ac3], workload), index_col=0)

  rp = pd.DataFrame(columns=[ac1, ac2, ac3])
  rp[ac1] = rp_sky[outcome].values
  rp[ac2] = rp_has[outcome].values
  rp[ac3] = rp_stg[outcome].values

  rpn = (rp-rp.mean())/rp.std()
      
  ############################# Load LLSM data ############################
  Sn_dict = {}

  S_sky = pd.read_csv('data/llsm/{}/{}.csv'.format(ac_dict['sky'], workload), index_col=0)
  S_has = pd.read_csv('data/llsm/{}/{}.csv'.format(ac_dict['has'], workload), index_col=0)
  S_stg = pd.read_csv('data/llsm/{}/{}.csv'.format(ac_dict['stg'], workload), index_col=0)

  Sn_sky = (S_sky-S_sky.mean())/S_sky.std()
  Sn_dict['sky'] = Sn_sky.copy()

  Sn_has = (S_has-S_has.mean())/S_has.std()
  Sn_dict['has'] = Sn_has.copy()

  Sn_stg = (S_stg-S_stg.mean())/S_stg.std()
  Sn_dict['stg'] = Sn_stg.copy()  


  ac_list = [ac1, ac2, ac3]

  res = {}
  res[ac1], res[ac2], res[ac3] = [], [], []

  n_step_per = n_step//Sn_dict['sky'].shape[1]

  ac_src = [target]
  ac_tar = list(set(ac_list) - set(ac_src)) 

  ###################### Model training ######################

  for ac in ac_tar:

      sort_idx_full = rp.sort_values(by=[ac_src[0]], ascending=True).index    
      sort_idx_sub = sort_idx_full[rpn.shape[0]//2:]
      
      X_sub = X.iloc[sort_idx_sub]
      Y_sub = rpn.loc[sort_idx_sub, ac_src[0]]
      U_sub = Sn_dict[ac_src[0]].iloc[sort_idx_sub]
      X_sub.reset_index(drop=True, inplace=True)
      Y_sub.reset_index(drop=True, inplace=True)
      U_sub.reset_index(drop=True, inplace=True)    
          
      ## Get init_idx from best configurations on another arch
      sort_idx_full_src = rp.sort_values(by=[ac_src[0]], ascending=True).index
      sort_idx_sub_src = sort_idx_full_src[rpn.shape[0]//2:(rpn.shape[0]//2+200)]

      sort_idx_full_tar = rp.sort_values(by=[ac], ascending=True).index
      sort_idx_sub_tar = sort_idx_full_tar[-500:]

      init_idx_pool = list(set(sort_idx_sub_src) & set(sort_idx_sub_tar))
      init_idx_raw = init_idx_pool[:n_step]  
      
      val_init_idx = rpn.loc[init_idx_raw, ac_src[0]].values
      init_idx = np.searchsorted(Y_sub, val_init_idx).tolist()

      
      ###################### 1. GIL ######################
      pool_idx_ex = range(X_sub.shape[0])
      seed_idx_ex = []
      pool_idx_ex = list(set(pool_idx_ex) - set(init_idx))
      seed_idx_ex = seed_idx_ex + init_idx
      for i in range(n_iter):
          ## Fit model between configs and outcome
          X_ex_tr, Y_ex_tr = X_sub.values[seed_idx_ex], Y_sub[seed_idx_ex]    
          pred_ex = Ridge().fit(X_ex_tr, Y_ex_tr).predict(X_sub.values)
          re_ex_tmp = pred_ex.copy()
          re_ex_tmp[seed_idx_ex] = -1000
          new_idx = np.argsort(re_ex_tmp)[-n_step:].tolist()
          ## Update pool indices and seed indices
          pool_idx_ex = list(set(pool_idx_ex)-set(new_idx))
          seed_idx_ex = seed_idx_ex + new_idx 
      top_idx_ex  = np.argmax(pred_ex)
      ## find index and value
      idx_nz_ex = X[X==X_sub.iloc[top_idx_ex,:]].dropna().index.item()
      top_thpt_ex = rp.loc[idx_nz_ex, ac_src[0]]

      ###################### 2. GIL+ ######################
      pool_idx_hi = range(X_sub.shape[0])
      seed_idx_hi = []
      pool_idx_hi = list(set(pool_idx_hi) - set(init_idx))
      seed_idx_hi = seed_idx_hi + init_idx
      for i in range(n_iter):
          ## Fit one model between uarch and outcome
          U_hi_tr, Y_hi_tr = U_sub.values[seed_idx_hi], Y_sub[seed_idx_hi]
          pred_u = Ridge().fit(U_hi_tr, Y_hi_tr).predict(U_sub.values)
          re_u_tmp = pred_u.copy()
          re_u_tmp[seed_idx_hi] = -1000
          top_idx_u = np.argmax(re_u_tmp)
          ## Get uarch vector corresponding to the best predicted Throughput
          top_u_vec = U_sub.values[top_idx_u]  
          ## Fit five models between configs and uarchs
          for j in range(U_sub.shape[1]):
              X_hi_tr, Up_hi_tr = X_sub.values[seed_idx_hi], U_sub.iloc[seed_idx_hi, j]
              pred_up = Ridge().fit(X_hi_tr, Up_hi_tr).predict(X_sub.values)
              re_up_tmp = pred_up.copy()
              re_up_tmp[seed_idx_hi] = -1000
              diff_up_tmp = (re_up_tmp - top_u_vec[j])**2
              new_idx = np.argsort(diff_up_tmp)[:n_step_per].tolist()
              ## Update pool indices and seed indices
              pool_idx_hi = list(set(pool_idx_hi)-set(new_idx))
              seed_idx_hi = seed_idx_hi + new_idx 
      ## Final search. Only happens between configs and Throughput 
      X_hi_tr, Y_hi_tr = X_sub.values[seed_idx_hi], Y_sub[seed_idx_hi]    
      pred_hi = Ridge().fit(X_hi_tr, Y_hi_tr).predict(X_sub.values)
      top_idx_hi  = np.argmax(pred_hi)
      ## find index and value
      idx_nz_hi = X[X==X_sub.iloc[top_idx_hi,:]].dropna().index.item()
      top_thpt_hi = rp.loc[idx_nz_hi, ac_src[0]]


      ###################### 3. DAC (ensemble tree + genetic algorithm) ######################
      seed_idx_dac = list(range(n_start,(n_start+n_step*n_iter)))
      pool_idx_dac = list(set(range(X_sub.shape[0])) - set(seed_idx_dac))
      X_dac_test, Y_dac_test = X_sub.values[pool_idx_dac], Y_sub.values[pool_idx_dac]

      X_dac_raw, Y_dac_raw = X_sub.values[seed_idx_dac], Y_sub.values[seed_idx_dac]
      model_dac = Ridge().fit(X_dac_raw, Y_dac_raw)

      ## Train a model using raw configs first. Why need a model? To evaluate on new crossover and mutant data
      ## to find the optimal config         
      pairs = list(itertools.combinations(range(X_dac_raw.shape[0]), 2))
      list_cm = []
      for pa in pairs:
          q, l = X_dac_raw[pa[0]], X_dac_raw[pa[1]]
          q_cm, l_cm = cross_mu(q, l)
          list_cm.append(q_cm)
          list_cm.append(l_cm)

      X_dac_cm = np.asarray(list_cm)
      pred_dac = GradientBoostingRegressor().fit(X_dac_raw, Y_dac_raw).predict(X_dac_cm)
      X_dac_max = X_dac_cm[np.argmax(pred_dac)]
      X_diff=np.linalg.norm((X_dac_test - X_dac_max), axis=1)
      top_idx_dac = np.argmin(X_diff)
      ## find index and value
      idx_nz_dac = X[X==X_sub.iloc[top_idx_dac,:]].dropna().index.item()
      top_thpt_dac = rp.loc[idx_nz_dac, ac_src[0]]           
      
      ########################## 3. ANN (ASPLOS 2006) ###########################    
      pool_idx_nn = range(X_sub.shape[0])
      seed_idx_nn= init_idx
      pool_idx_nn = list(set(pool_idx_nn)-set(seed_idx_nn))
      X_nn_tr, Y_nn_tr = X_sub.values[seed_idx_nn], Y_sub.values[seed_idx_nn] 

      kf = KFold(n_splits=5)
      X_nn_te_sub = X_sub.values[pool_idx_nn]
      pred_nn_list = []
      ANN = MLPRegressor(random_state=0, hidden_layer_sizes=16, max_iter=500)
      for train, test in kf.split(X_nn_tr):
          X_nn_tr_sub, Y_nn_tr_sub = X_nn_tr[train], Y_nn_tr[train]    
          pred_nn = ANN.fit(X_nn_tr_sub, Y_nn_tr_sub).predict(X_nn_te_sub)
          pred_nn_list.append(pred_nn)
      nn_var = np.var(np.asarray(pred_nn_list), axis=0)
      idx_var = np.argsort(nn_var)[-(n_step*(n_iter-1)):] 
      seed_idx_nn = seed_idx_nn + idx_var.tolist()

      X_nn_tr, Y_nn_tr = X_sub.values[seed_idx_nn], Y_sub.values[seed_idx_nn] 
      test_idx_nn = list(set(range(X_sub.shape[0]))-set(seed_idx_nn))
      mean_nn = ANN.fit(X_nn_tr, Y_nn_tr).predict(X_sub.values[test_idx_nn])    
      top_idx_nn = np.argmax(mean_nn)
      ## find index and value
      idx_nz_nn = X[X==X_sub.iloc[top_idx_nn,:]].dropna().index.item()
      top_thpt_nn = rp.loc[idx_nz_nn, ac_src[0]]        
      
      ########################## 4. BO ########################### 
      pool_idx_bo = range(X_sub.shape[0])
      seed_idx_bo = init_idx
      pool_idx_bo = list(set(pool_idx_bo)-set(init_idx))

      kernel = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e1)) 
      model=GaussianProcessRegressor(kernel=kernel, alpha=1)
      
      for i in range(n_iter):    
          ## Fit GP model
          X_bo_tr, Y_bo_tr = X_sub.values[seed_idx_bo], Y_sub.values[seed_idx_bo]    
          mean_bo, std_bo = model.fit(X_bo_tr, Y_bo_tr).predict(X_sub.values, return_std=True)
          ## Get co reward function
          re_bo = std_bo
          re_bo_tmp = re_bo.copy()
          re_bo_tmp[seed_idx_bo] = -1000
          new_idx = np.argsort(re_bo_tmp)[-n_step:].tolist()
          ## Update pool indices and seed indices
          pool_idx_bo = list(set(pool_idx_bo)-set(new_idx))
          seed_idx_bo = seed_idx_bo + new_idx 
      top_idx_bo  = np.argmax(mean_bo)
      ## find index and value
      idx_nz_bo = X[X==X_sub.iloc[top_idx_bo,:]].dropna().index.item()
      top_thpt_bo = rp.loc[idx_nz_bo, ac_src[0]]       
      
      ###################### 5. Random sampling with linear regression ######################
      seed_idx_rnd = list(range(n_step*n_iter))
      pool_idx_rnd = list(set(range(X_sub.shape[0])) - set(seed_idx_rnd))
      X_rnd_tr, Y_rnd_tr = X_sub.values[seed_idx_rnd], Y_sub[seed_idx_rnd]
      pred_rnd = Ridge().fit(X_rnd_tr, Y_rnd_tr).predict(X_sub.values)
      top_idx_rnd  = np.argmax(pred_rnd)
      ## find index and value
      idx_nz_rnd = X[X==X_sub.iloc[top_idx_rnd,:]].dropna().index.item()
      top_thpt_rnd = rp.loc[idx_nz_rnd, ac_src[0]]                

      ###################### 6. True optimal ######################
      top_thpt_opt = rp.loc[sort_idx_full[-1], ac_src[0]]

      ###################### 7. Default ######################
      top_thpt_de = rp.loc[sort_idx_full[500], ac_src[0]]

      res[ac] = [top_thpt_opt, top_thpt_de, top_thpt_rnd, top_thpt_nn, 
                 top_thpt_dac, top_thpt_bo, top_thpt_ex, top_thpt_hi]            

  ###################### Save results ######################

  methods = ['opt', 'de', 'rs', 'nn', 'dac', 'bo', 'gil', 'gil+']
  np_res = np.column_stack((res[ac_tar[0]],res[ac_tar[1]]))
  df = pd.DataFrame(np_res, index= methods, columns=ac_tar)
  df.loc['n-de',:]=(df.loc['opt',:] - df.loc['de',:])/df.loc['opt',:]
  df.loc['n-rs',:]=(df.loc['opt',:] - df.loc['rs',:])/df.loc['opt',:]
  df.loc['n-nn',:]=(df.loc['opt',:] - df.loc['nn',:])/df.loc['opt',:]
  df.loc['n-dac',:]=(df.loc['opt',:] - df.loc['dac',:])/df.loc['opt',:]
  df.loc['n-bo',:]=(df.loc['opt',:] - df.loc['bo',:])/df.loc['opt',:]
  df.loc['n-gil',:]=(df.loc['opt',:] - df.loc['gil',:])/df.loc['opt',:]
  df.loc['n-gil+',:]=(df.loc['opt',:] - df.loc['gil+',:])/df.loc['opt',:]
  # df.to_csv('output/mid2high/{}.csv'.format(workload), index=True)
  print(df)


if __name__ == '__main__':
  main()














