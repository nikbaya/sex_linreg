#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:58:10 2018

Phenotypic association of sex for all phenotypes in UK Biobank Round 2.

Three related models of linear regression, with progressively fewer sex-related predictors.

@author: nbaya
"""
import hail as hl
import pandas as pd
import datetime
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phsource', type=str, required=True, help="phenotype source (phesant, icd10, finngen)")
parser.add_argument('--parsplit', type=int, required=True, help="number of batches to split phsource phenotypes into")
parser.add_argument('--paridx', type=int, required=True, help="batch id number")

args = parser.parse_args()

paridx = args.paridx

phsource = args.phsource

print('\n####################')
print('Running phenotypes from ' + phsource)
print('Number of parsplit batches: ' + str(args.parsplit))
print('Batch paridx: ' + str(args.paridx))
print('####################')

if phsource == 'icd10':
    phen_tb_all = hl.import_table('gs://nbaya/linreg/icd10_phenotypes.both_sexes.tsv.bgz',missing='',impute=True)
elif phsource == 'phesant':
    phen_tb_all = hl.import_table('gs://phenotype_31063/ukb31063.phesant_phenotypes.both_sexes.tsv.bgz',missing='',impute=True,types={'"userId"': hl.tstr}).rename({'"userId"': 's'})
elif phsource == 'finngen':
    phen_tb_all = hl.import_table('gs://ukb31063-mega-gwas/phenotype-files/curated-phenotypes/2018-04-06_ukb-finngen-pheno-for-analysis.tsv',missing='',impute=True,types={'eid':hl.tstr}).rename({'eid':'s'})
phen_tb_all = phen_tb_all.annotate(s = hl.str(phen_tb_all.s))
phen_tb_all = phen_tb_all.key_by('s')

phenlist = [x.replace('\"','') for x in phen_tb_all.row_value][0:]

cov =  hl.import_table('gs://phenotype_31063/ukb31063.gwas_covariates.both_sexes.tsv',
                             key='s', impute=True, types={'s': hl.tstr})
cov = cov.annotate(age = cov.age - cov.aggregate(hl.agg.mean(cov.age))) #center age
cov = cov.annotate(age_squared = cov.age**2)
cov = cov.annotate(age_isFemale = cov.age*cov.isFemale)
cov = cov.annotate(age_squared_isFemale = cov.age_squared*cov.isFemale)

withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
withdrawn_set = set(withdrawn.f0.take(withdrawn.count()))

cov = cov.filter(hl.literal(withdrawn_set).contains(cov['s']),keep=False) #Remove withdrawn samples from covariates

cov_samples = set(cov.s.take(cov.count())) #sample IDs from covariates table
phen_tb_all = phen_tb_all.filter(hl.literal(cov_samples).contains(phen_tb_all['s']),keep=True) #Only keep samples from the newly filtered covariates

def get_cols(cov):
    cols = [['phen','r2_mul','r2_adj'],['beta_{:}'.format(i) for i in cov],['beta_PC{:}'.format(i) for i in range(1, 21)],
        ['se_{:}'.format(i) for i in cov],['se_PC{:}'.format(i) for i in range(1, 21)],
        ['tstat_{:}'.format(i) for i in cov],['tstat_PC{:}'.format(i) for i in range(1, 21)],
        ['pval_{:}'.format(i) for i in cov],['pval_PC{:}'.format(i) for i in range(1, 21)]]
    cols = [i for j in cols for i in j]
    return cols

cov1 = ['intercept','isFemale', 'age','age_square','age_isFemale','age_square_isFemale']
cols1 = get_cols(cov1)

cov2 = ['intercept','isFemale', 'age','age_square']
cols2 = get_cols(cov2)

cov3 = ['intercept','age','age_square']
cols3 = get_cols(cov3)


df1 = pd.DataFrame(columns = cols1)
df2 = pd.DataFrame(columns = cols2)
df3 = pd.DataFrame(columns = cols3)

numphens = len(phenlist)

start_idx = int((paridx-1)*numphens/args.parsplit)
stop_idx = int((paridx)*numphens/args.parsplit)
idx = range(start_idx,stop_idx,1) #chunks all phenotypes into parsplit number of chunks, then runs on the paridx-th chunk

for i in idx:
    phen = phenlist[i]
    
    print('############')
    print('Running phenotype '+phen+' (iter '+str(i+1)+')')
    print('iter '+str(idx.index(i)+1)+' of '+str(len(idx))+' for parallel batch '+str(paridx))
    starttime = datetime.datetime.now()
    print('############')
    
    if phsource == 'phesant':
        phen_tb = phen_tb_all.select('"'+phen+'"').join(cov, how='inner') #join phenotype and covariates    
        phen_tb = phen_tb.annotate(phen_str = hl.str(phen_tb['"'+phen+'"']))
        phen_tb = phen_tb.filter(phen_tb.phen_str == '',keep=False)
        if phen_tb['"'+phen+'"'].dtype == hl.dtype('bool'):
            phen_tb = phen_tb.annotate(phen = hl.bool(phen_tb.phen_str.replace('\"',''))).drop('"'+phen+'"')
        else:
            phen_tb = phen_tb.annotate(phen = hl.float64(phen_tb.phen_str.replace('\"',''))).drop('"'+phen+'"')
    elif phsource == 'icd10' or phsource == 'finngen':
        phen_tb = phen_tb_all.select(phen).join(cov, how='inner') #join phenotype and covariates    
        phen_tb = phen_tb.annotate(phen_str = hl.str(phen_tb[phen]))
        phen_tb = phen_tb.filter(phen_tb.phen_str == '',keep=False)
        if phen_tb[phen].dtype == hl.dtype('bool'):
            phen_tb = phen_tb.annotate(phen = hl.bool(phen_tb.phen_str)).drop(phen)
        else:
            phen_tb = phen_tb.annotate(phen = hl.float64(phen_tb.phen_str.replace('\"',''))).drop(phen)
            
    n = phen_tb.count()
    print('\n>>> Sample count for phenotype '+phen+': '+str(n)+' <<<')
    
    if phen_tb.filter(phen_tb.isFemale == 1).count() % n != 0: #if phenotype is not sex-specific (i.e. all female or all male)
        # regression 1: All covariates
        cov_list1 = [ phen_tb['isFemale'], phen_tb['age'], phen_tb['age_squared'], phen_tb['age_isFemale'],
                    phen_tb['age_squared_isFemale'] ]+ [phen_tb['PC{:}'.format(i)] for i in range(1, 21)] 
        reg1 = phen_tb.aggregate(hl.agg.linreg(y=phen_tb.phen, x = [1]+cov_list1))
        stats1 = [[phen],[reg1.multiple_r_squared,reg1.adjusted_r_squared],reg1.beta,
                  reg1.standard_error,reg1.t_stat,reg1.p_value]
        stats1 = [i for j in stats1 for i in j] #flatten list

        df1.loc[i] = stats1

        # regression 2: All covariates except for sex*age and sex*age2
        cov_list2 = [ phen_tb['isFemale'], phen_tb['age'], phen_tb['age_squared'] 
                    ]+[phen_tb['PC{:}'.format(i)] for i in range(1, 21)] 
        reg2 = phen_tb.aggregate(hl.agg.linreg(y=phen_tb.phen, x = [1]+cov_list2))
        stats2 = [[phen],[reg2.multiple_r_squared,reg2.adjusted_r_squared],reg2.beta,
                  reg2.standard_error,reg2.t_stat,reg2.p_value]
        stats2 = [i for j in stats2 for i in j] #flatten list

        df2.loc[i] = stats2
        
    # regression 3: All covariates except for sex, sex*age and sex*age2
    cov_list3 = [ phen_tb['age'], phen_tb['age_squared'] ]+[phen_tb['PC{:}'.format(i)] for i in range(1, 21)] 
    reg3 = phen_tb.aggregate(hl.agg.linreg(y=phen_tb.phen, x = [1]+cov_list3))
    stats3 = [[phen],[reg3.multiple_r_squared,reg3.adjusted_r_squared],reg3.beta,
              reg3.standard_error,reg3.t_stat,reg3.p_value]
    stats3 = [i for j in stats3 for i in j] #flatten list
    
    df3.loc[i] = stats3
    
    stoptime = datetime.datetime.now()

    print('\n############')
    print('Iteration time for '+phen+': '+str(round((stoptime-starttime).seconds/60, 2))+' minutes')
    print('############')
          
hl.Table.from_pandas(df1).export('gs://nbaya/linreg/ukb31063.'+phsource+'_phenotypes.both_sexes.reg1_batch'+str(args.paridx)+'.tsv.bgz',header=True)
hl.Table.from_pandas(df2).export('gs://nbaya/linreg/ukb31063.'+phsource+'_phenotypes.both_sexes.reg2_batch'+str(args.paridx)+'.tsv.bgz',header=True)
hl.Table.from_pandas(df3).export('gs://nbaya/linreg/ukb31063.'+phsource+'_phenotypes.both_sexes.reg3_batch'+str(args.paridx)+'.tsv.bgz',header=True)
