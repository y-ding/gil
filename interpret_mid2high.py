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

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct
from sklearn.ensemble import GradientBoostingRegressor
from itertools import product
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import itertools 


def main():

  #####################################################
  ########## Read data and simple processing ########## 
  #####################################################

  parser = argparse.ArgumentParser(description='Configuration extrapolation with GIL/GIL+.')
  parser.add_argument('--source', type=str, default="has", help='Source hardware')
  parser.add_argument('--target', type=str, default="stg", help='Target hardware')
  parser.add_argument('--workload', type=str, default="als", help='Workload')
  parser.add_argument('--outcome', type=str, default="Throughput", help='Outcome metric') 
  parser.add_argument('--n_start', type=int, default=200, help='Start index (default: 200)')
  parser.add_argument('--n_step', type=int, default=20, help='k: number of configurations queried at each round (default: 20)')
  parser.add_argument('--n_iter', type=int, default=20, help='T: number of rounds (default: 20)')

  args = parser.parse_args()
 
  source     = args.source
  target     = args.target
  workload   = args.workload  
  outcome    = args.outcome
  n_start    = args.n_start  
  n_step     = args.n_step    
  n_iter     = args.n_iter   

  print("source:   {}".format(source))
  print("target:   {}".format(target))
  print("workload: {}".format(workload))
  print("outcome:  {}".format(outcome))
  print("n_start:  {}".format(n_start))
  print("n_step:   {}".format(n_step))
  print("n_iter:   {}".format(n_iter))

  X = pd.read_csv('data/perf/config_clean.csv', index_col=0)
  X_raw = pd.read_csv('data/perf/config_clean.csv', index_col=0)
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

  ###################### Model training ######################

  ac_tar = [target]  ## end configuration
  ac_src = list(set(ac_list) - set(ac_tar)) 
  ac_src = [source]  ## start configuration

  for ac in ac_src:

      sort_idx_full = rp.sort_values(by=[ac_tar[0]], ascending=True).index    
      sort_idx_sub = sort_idx_full[rpn.shape[0]//2:]
      
      X_sub = X.iloc[sort_idx_sub]
      Y_sub = rpn.loc[sort_idx_sub, ac_tar[0]]
      U_sub = Sn_dict[ac_tar[0]].iloc[sort_idx_sub]
      X_sub.reset_index(drop=True, inplace=True)
      Y_sub.reset_index(drop=True, inplace=True)
      U_sub.reset_index(drop=True, inplace=True)    
          
      ## Get init_idx from best configurations on another arch
      sort_idx_full_tar = rp.sort_values(by=[ac_tar[0]], ascending=True).index
      sort_idx_sub_tar = sort_idx_full_tar[rpn.shape[0]//2:(rpn.shape[0]//2+200)]

      sort_idx_full_src = rp.sort_values(by=[ac], ascending=True).index
      sort_idx_sub_src = sort_idx_full_src[-500:]

      init_idx_pool = list(set(sort_idx_sub_tar) & set(sort_idx_sub_src))
      init_idx_raw = init_idx_pool[:n_step]  
      
      val_init_idx = rpn.loc[init_idx_raw, ac_tar[0]].values
      init_idx = np.searchsorted(Y_sub, val_init_idx).tolist()
          
      ############################## Before extrapolation ##############################
      U_be_tr, Y_be_tr = U_sub.values[init_idx], Y_sub[init_idx]
      uarch_coef_be = Ridge(fit_intercept=False).fit(U_be_tr, Y_be_tr).coef_    
      X_be_tr, Y_be_tr = X_sub.values[init_idx], Y_sub[init_idx]
      conf_coef_be = Ridge(fit_intercept=False).fit(X_be_tr, Y_be_tr).coef_
      
      coef_uarch_be = []
      for j in range(U_sub.shape[1]):
          pred_uarch_be = Ridge(fit_intercept=False).fit(X_be_tr, U_be_tr[:,j]).coef_
          coef_uarch_be.append(pred_uarch_be)        
      
      ############################## Start extrapolating ##############################
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
              
      ############################## After extrapolation ##############################
      seed_idx_after = list(set(seed_idx_hi)-set(init_idx))
      U_af_tr, Y_af_tr = U_sub.values[seed_idx_after], Y_sub[seed_idx_after]
      uarch_coef_af = Ridge(fit_intercept=False).fit(U_af_tr, Y_af_tr).coef_    
      X_af_tr, Y_af_tr = X_sub.values[seed_idx_after], Y_sub[seed_idx_after]
      conf_coef_af = Ridge(fit_intercept=False).fit(X_af_tr, Y_af_tr).coef_ 
      
      coef_uarch_af = []
      for j in range(U_sub.shape[1]):
          pred_uarch_af = Ridge(fit_intercept=False).fit(X_af_tr, U_af_tr[:,j]).coef_
          coef_uarch_af.append(pred_uarch_af)     

  uarch_names = ['BMR', 'CMR', 'IPC', 'CSR', 'PFR']
  uarch_coef_diff=np.linalg.norm((uarch_coef_af.reshape(-1,1) - uarch_coef_be.reshape(-1,1)), axis=1)
  np_uarch = np.row_stack((uarch_coef_be, uarch_coef_af, uarch_coef_diff))
  df_uarch = pd.DataFrame(np_uarch, index=['Before', 'After', 'Diff'], columns=uarch_names)
  df_uarch.rename_axis('Models', inplace=True)
  print(df_uarch)       


  ###################### Draw radar charts ######################
  def plot_radar(df, radar_labels, wl, ac):
      
      # Set up colors
      ORANGE = '#FD7120'
      BLUE = '#00BFFF'
      GREEN = '#009E73'

      # Each attribute we'll plot in the radar chart.
      labels =radar_labels
      cl = ['r', GREEN, 'b', '#ff7f00', ' #377eb8', '#ffff33']
      md = ['Before', 'After']
      
      # Number of variables we're plotting.
      num_vars = len(labels)

      # Split the circle into even parts and save the angles so we know where to put each axis.
      angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

      # The plot is a circle, so we need to "complete the loop" and append the start value to the end.
      angles += angles[:1]

      # ax = plt.subplot(polar=True)
      fig, ax = plt.subplots(figsize=(3,3), subplot_kw=dict(polar=True))

      # Helper function to plot each car on the radar chart.
      def add_to_radar(model, color):
          
          values = df.loc[model].tolist()
          values += values[:1]
          ax.plot(angles, values, color=color, linewidth=2.5, label=model)
          ax.fill(angles, values, color=color, alpha=0.2)

      # Add each car to the chart.
      add_to_radar(md[0], cl[0])
      add_to_radar(md[1], cl[1])

      # Fix axis to go in the right order and start at 12 o'clock.
      ax.set_theta_offset(np.pi / 2)
      ax.set_theta_direction(-1)
          
      ax.set_thetagrids(np.degrees(angles), labels)

      # Ensure radar goes from 0 to 100.
      ax.set_ylim(-1, 1.1)
  #     ax.set_ylim(-0.5, 2.1) ## for haswell lr only
      # You can also set gridlines manually like this:
      ax.set_rgrids([-1, -0.5, 0, 0.5, 1])
  #     ax.set_rgrids([-0.5, 0, 0.5, 1, 1.5, 2]) ## for haswell lr only

      # Set position of y-labels (0-100) to be in the middle
      # of the first two axes.
      ax.set_rlabel_position(180 / num_vars)

      # Add title.
      #ax.set_title('Learned coefficients for each LLSM', y=1.08)

      # Add a legend as well.
      ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.2))
      plt.savefig('output/fig/mid2high/{}/radar_{}.pdf'.format(ac[0], wl),bbox_inches='tight') 

  cause_labels = ['BMR', 'CMR', 'IPC', 'CSR', 'PFR']    
  plot_radar(df_uarch, cause_labels, workload, ac_tar)


  ###################### Draw bar charts ######################

  ## Get top config parameters influencing performance
  coef_diff=np.linalg.norm((conf_coef_be.reshape(-1,1) - conf_coef_af.reshape(-1,1)), axis=1)
  order = coef_diff.argsort()[::-1]
  ranks = order.argsort()
  np_conf = np.column_stack((np.asarray(list(X_raw)).reshape(-1,1), conf_coef_be, conf_coef_af, coef_diff, ranks))
  df_conf = pd.DataFrame(np_conf, columns=['conf', 'before', 'after', 'diff', 'ranks'])
  df_sort = df_conf.sort_values(by=['diff'], ascending=False).copy()
  df_list = list(df_sort.iloc[:,0:3].itertuples(index=False, name=None))
  conf = [x[0] for x in df_list]

  ## Get corresponding llsm diff coefs
  df_uarch_be = pd.DataFrame(data=np.asarray(np.abs(coef_uarch_be)).T, columns=list(U_sub), index=list(X_raw))
  df_uarch_af = pd.DataFrame(data=np.asarray(np.abs(coef_uarch_af)).T, columns=list(U_sub), index=list(X_raw))
  df_uarch_diff = df_uarch_af - df_uarch_be
  df_uarch_diff.columns = ['BMR', 'CMR', 'IPC', 'CSR', 'PFR']
  df_uarch_cut = df_uarch_diff.loc[conf, :]
  df_uarch_cut[np.abs(df_uarch_cut)<0.05] = 0
  df_uarch_cut
  df_final = df_uarch_cut[(df_uarch_cut.T != 0).any()]

  plt.style.use('classic')
  fs = 20
  df_final.T.plot.barh(figsize=(5.5,8), width=1, colormap='gist_ncar')
  plt.xlabel('Linear coefficient differences', fontsize=fs)
  plt.tick_params(axis="x", labelsize=fs)
  plt.xticks(rotation='horizontal')
  plt.tick_params(axis="y", labelsize=fs)
  plt.legend(fontsize='x-large',bbox_to_anchor=(1, 1),loc='best', ncol=1,frameon=False)
  plt.savefig("output/fig/mid2high/{}/conf_{}.pdf".format(ac_tar[0], workload),bbox_inches='tight')


if __name__ == '__main__':
  main()
