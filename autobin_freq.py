# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from compiler.ast import flatten

def autobin_freq(df,feature,label,initial_bin_num,min_class_pct,stop_limit):
    df_bk=df[[feature,label]] # used for final binning
    df=df_bk.copy()           # used for iterative merging of bins
    df.columns=['predictor','target']
    stop_limit_exceeded=False
    bin_num=initial_bin_num
    while(bin_num>=2 and stop_limit_exceeded==False):
      #equal_frequency binning with specified bin_num 
      df.predictor_bin=pd.qcut(df.predictor.rank(method='first'),bin_num)
      # Compute crosstab from binned variable and target variable
      cttab=pd.crosstab(df.predictor_bin,df.target)
      cttab.columns=['good','bad']
      # Compute columns percents for target classes from crosstab frequencies
      cttab['good_ratio']=1.0*cttab.ix[:,'good']/sum(cttab.ix[:,'good'])
      cttab['bad_ratio']=1.0*cttab.ix[:,'bad']/sum(cttab.ix[:,'bad'])
      # Correct column percents in case of 0 frequencies 
      if (min(cttab.good)==0 or min(cttab.bad)==0):
          cttab.good_ratio=1.0*(cttab.good_ratio+0.0001)/sum(cttab.good_ratio+0.0001)
          cttab.bad_ratio=1.0*(cttab.bad_ratio+0.0001)/sum(cttab.bad_ratio+0.0001)
      #print cttab
      cttab['woe100']=100.0*np.log(cttab['good_ratio']/cttab['bad_ratio'])           
      cttab['iv']=(cttab['good_ratio']-cttab['bad_ratio'])*(cttab['woe100'])/100.0
      #Compute final IV
      iv_total=sum(cttab['iv'])
      # Collect total IVs for different binning solutions
      if locals().has_key('iv_collect'):
          iv_collect.append(iv_total)
      else:
          iv_collect=[iv_total]
      # In case IV decreases by more than percentage specified by stop.limit parameter above
      # restore former binning solution (cutpoints) and leave loop
      if len(iv_collect)>1 and stop_limit_exceeded==False:
          iv_decrease=1.0*(iv_collect[len(iv_collect)-2]-iv_collect[len(iv_collect)-1])/iv_collect[len(iv_collect)-2]
#          print iv_decrease
          if iv_decrease>stop_limit:
              stop_limit_exceeded=True
      #print bin_num,iv_total
      bin_num-=1
    bin_num_final=bin_num+1
    iv_total_final=iv_collect[len(iv_collect)-2]
    predictor_freq_final=pd.qcut(df.predictor.rank(method='first'),bin_num_final,labels=range(bin_num_final))
    return predictor_freq_final,iv_total_final,bin_num_final
