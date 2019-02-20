#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:15:50 2018

Regression of sex on PCs and age, UK Biobank Round 2.

@author: nbaya
"""
import hail as hl
import pandas as pd

print('\n####################')
print('Running sex ~ age + age^2 + 20 PCs regression')
print('####################')

cov =  hl.import_table('gs://phenotype_31063/ukb31063.gwas_covariates.both_sexes.tsv',
                             key='s', impute=True, types={'s': hl.tstr})
cov = cov.annotate(age = cov.age - cov.aggregate(hl.agg.mean(cov.age))) #center age
cov = cov.annotate(age_squared = cov.age**2)
cov = cov.annotate(age_isFemale = cov.age*cov.isFemale)
cov = cov.annotate(age_squared_isFemale = cov.age_squared*cov.isFemale)

withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
withdrawn_set = set(withdrawn.f0.take(withdrawn.count()))

cov = cov.filter(hl.literal(withdrawn_set).contains(cov['s']),keep=False) #Remove withdrawn samples from covariates

cov1 = ['intercept','age','age_square']
cols1 = [['isFemale','r2_mul','r2_adj'],['beta_{:}'.format(i) for i in cov1],['beta_PC{:}'.format(i) for i in range(1, 21)],
        ['se_{:}'.format(i) for i in cov1],['se_PC{:}'.format(i) for i in range(1, 21)],
        ['tstat_{:}'.format(i) for i in cov1],['tstat_PC{:}'.format(i) for i in range(1, 21)],
        ['pval_{:}'.format(i) for i in cov1],['pval_PC{:}'.format(i) for i in range(1, 21)]]
cols1 = [i for j in cols1 for i in j]

df1 = pd.DataFrame(columns = cols1)

cov_list1 = [ cov['age'], cov['age_squared']]+ [cov['PC{:}'.format(i)] for i in range(1, 21)] 
reg1 = cov.aggregate(hl.agg.linreg(y=cov['isFemale'], x = [1]+cov_list1))
stats1 = [['isFemale'],[reg1.multiple_r_squared,reg1.adjusted_r_squared],reg1.beta,
          reg1.standard_error,reg1.t_stat,reg1.p_value]
stats1 = [i for j in stats1 for i in j] #flatten list

df1.loc[0] = stats1

hl.Table.from_pandas(df1).export('gs://nbaya/sex_linreg/ukb31063.both_sexes.sex_linreg_w_PCs.tsv.bgz',header=True)

print('####################')
print('Regression complete')
print('####################')