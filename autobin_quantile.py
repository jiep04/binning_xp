# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from compiler.ast import flatten


def autobin_quantile(df,feature,label,initial_bin_num,min_class_pct,stop_limit):
    df_bk=df[[feature,label]] # used for final binning
    df=df_bk.copy()           # used for iterative merging of bins
    df.columns=['predictor','target']
    stop_limit_exceeded=False
    bin_num=initial_bin_num
    while(bin_num>=2 and stop_limit_exceeded==False): 
      cutpoints=[min(df.predictor)]
      quantile_value=df.predictor.quantile(map(lambda x:1.0*x/bin_num,\
                                               range(1,bin_num)),\
                                                  interpolation='nearest')
      cutpoints.append(list(quantile_value))
      cutpoints.append(max(df.predictor))
      cutpoints=flatten(cutpoints)
      cutpoints=list(np.unique(cutpoints)) # remove multiple cutpoints with same value
      #print "Initial binning number is: %i" %(len(cutpoints)-1)   
      ## Calculate initial crosstab from binned variable and target variable
      df.predictor_bin=pd.cut(df.predictor,cutpoints, right=True,\
                          labels=range(len(cutpoints)-1), include_lowest=True)
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
              cutpoints_final=cutpoints_backup[:] #copy
              cttab_final=cttab_backup.copy(deep=True)
              stop_limit_exceeded=True
      if not locals().has_key('cutpoints_backup'):
          cutpoints_backup=cutpoints[:]
          cttab_final=cttab.copy(deep=True)
      ## before merging to be able to retrieve solution in case IV decrease is too strong
      cutpoints_backup = cutpoints[:]
      cttab_backup = cttab.copy(deep=True) 
      #print bin_num,iv_total
      ## update bin_num iteratively
      bin_num-=1
    ## Compute final IV
    iv_total_final=sum(cttab_final['iv'])
    ## Output final class
    predictor_quantile_final=pd.cut(df.predictor,cutpoints_final, right=True,\
                            labels=range(len(cutpoints_final)-1), include_lowest=True)
    predictor_quantile_final.columns=[feature]
    return predictor_quantile_final,iv_total_final,len(cutpoints_final)-1
