# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:31:09 2019

@authors:
    Dr. Dan Vilenchik
    Mr. Adam Soffer
    
@Getting Started:

Parameters:    
    
    fpath: 
        Enter the file path and name, either absolute path or relative to the run file. the excepted file is a numeric matrix of dimensions n,p. where n is the amount of rows and p is the amount of columns.
        Default value: “data.csv”
    k: 
        Enter the amount of the wanted output dimensions. Ofcourse k has to smaller than p.
        Default value: “20”.
    k_star: 
        Mentioned above, this algorithm uses k* sized seeds. The algorithm’s runtime is exponential in k*.
        Default value: “1”.
    batch:
        Since the algorithm is built to run parallelly, each cpu will handle |batch| amount of seeds. By changing this parameter, task managment overhead can be decreased thus optimizing the runtime of the algorithm. Due notice - the algorithm does not optimize automaticaly and the optimization is up to the user.
        Default value: “data.csv”.
    cpus:
        Since the algorithm is built to run parallelly, cpus limits the amount of cpu’s used.
        Default value: “1”.
    newrun: 
        This version saves the state of the algorithm in case of bad connection, crashes etc… When calling for the algorithm with the same parameters the last saved state is loaded and the run continues from that checkpoint. If you wish to generate a new run, and ignore old checkpoints, set this parameter to 1.
        Default value: “0”.

Output:
    a csv file located at "./out/[date]/" called "sspcaExp_[k]_[k_star].csv".
    this file holds a column called "k_entries" which holds the features that the algorithm chose.
    
    
"""

import os
PROJECT_DIR = os.getcwd()
DATA_DIR = os.getcwd()
os.chdir(PROJECT_DIR)

from time import clock
from time import strftime
from math import factorial
from itertools import combinations
from multiprocessing import Pool
from multiprocessing import cpu_count
from sys import stdout
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.linalg import eigh

def NumericdataToCovariance(path_to_data):

    X = np.array(pd.read_csv(path_to_data))
    n=X.shape[1]
    Sigma = np.matmul(X.T,X)/n   
    
    return Sigma
    

def SSPCA( inp ):   
    final_ev = -1
    final_s = []
    Sigma = inp[1]
    k = inp[2]
    max_trace = inp[3]
    s_list = np.array(inp[0])
    # Final between |s| <= 3 elements
    for s_kernel in s_list:
        
        s = list(s_kernel.copy())
        unused = np.setdiff1d(np.arange(len(Sigma)),s)
        unused = list(np.sort(unused))
        
        while (len(s) < k):
            
            costs = np.abs(Sigma[np.ix_(unused,s)]).sum(axis=1)
            largest = np.argmax(costs)
            candidate = unused[largest]
            unused.remove(candidate)
            s.append(candidate)
        
        kkmat = Sigma[np.ix_(s,s)]
        
        ev = np.max(eigh(kkmat, eigvals_only=True))/max_trace
        
        if(ev > final_ev):
            final_ev = ev.copy()
            final_s = s.copy()
    
    return final_s, final_ev


def inp_gen(inp_sigma, inp_p, inp_k_star, inp_k, inp_max_trace, B=1):
    combs = combinations(list(range(inp_p)),inp_k_star)
    flag = True
    while(flag):
        batch = []
        for _ in range(B):
            comb = next(combs, None)
            batch += [comb]
            
            if(batch[-1] == None):
                batch.remove(None)
                flag = False
            
        yield ( batch, inp_sigma, inp_k, inp_max_trace)  



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--k", help="how many features to keep??", type=int)
    parser.add_argument("--k_star", help="size of choosing??", type=int)
    parser.add_argument("--fpath", help="relative path to numeric data including filename", type=int)
    parser.add_argument("--batch", help="Batch Size??", type=int)
    parser.add_argument("--cpus", help="How many cpu's should we use??, default is min(60,available)", type=int)
    parser.add_argument("--newrun", help="should we start from the beggining?)", type=int)
    

    args = parser.parse_args()
    
    if(args.k is not None):
        k = args.k
    else:
        k = 20
    if(args.k_star is not None):
        k_star = args.k_star
    else:
        k_star = 1
    if(args.fpath is not None):
        fpath = args.fpath
    else:
        fpath = './data.csv'
    if(args.batch is not None):
        B = args.batch
    else:
        B = 100
    if(args.cpus is not None):
        cpus_available = args.cpus
    else:
        cpus_available = 1
    if(args.newrun is not None):
        newrun = bool(args.newrun)
    else:
        newrun = bool(0)
        
    

    print("newRun is {}, loading data accordingly...".format(newrun))    
    
    if( 'sspca_tmp_sigma_{}_{}.csv'.format(k, k_star) not in os.listdir() or newrun ):
        print('Loading Data and generating covariance matrix')

        Sigma = NumericdataToCovariance(fpath)
        pd.DataFrame(Sigma).to_csv('sspca_tmp_sigma_{}_{}.csv'.format(k, k_star), index=False)

        print('Sigma Hat is generated, Data Shape = [{}][{}]'.format(Sigma.shape[0], Sigma.shape[1]))
    else:
        print('Found Sigma Hat')
        Sigma = np.array( pd.read_csv('sspca_tmp_sigma_{}_{}.csv'.format(k, k_star) ) ) 


    print('==========================================\n')
    print('Project Directory is:', PROJECT_DIR)
    print('Data Directory is:', DATA_DIR)


    #%%
    # =============================================================================
    # Part 1 - initialize 
    # =============================================================================
    ev = 0
    k_entries = []
    start_time = clock()
    cp_time = 0
    skip_to_index = 0
    
    print('cpu count = ' , cpu_count())
    print('\n==========================================\n')

    print('\nk = {} | k_star = {} | batch size = {}'.format(k, k_star,B))
   
    print('Checking for Checkpoint File')
    if( 'sspca_tmp_{}_{}.csv'.format(k, k_star) in os.listdir() and not newrun):
        print('found checkpoint')
        tmp_state = pd.read_csv('sspca_tmp_{}_{}.csv'.format(k, k_star))
        skip_to_index = tmp_state.iloc[0,0]
        ev = tmp_state.iloc[0,1]
        k_entries = tmp_state.iloc[:,2]
        cp_time = tmp_state.iloc[0,3]
        Sigma = pd.read_csv('sspca_tmp_sigma_{}_{}.csv'.format(k, k_star) )
        Sigma = np.array(Sigma)
        print('checkpoint loaded, last saved index = {}, best evaluation = {}'.format(skip_to_index, ev) )
    
    p = Sigma.shape[1]
    max_trace = np.sum(np.diag(Sigma))

    print('max_trace (sum diagonal) = ',max_trace)
    
    print('Sigma shape = {}'.format(Sigma.shape))    
    print('\n==========================================\n')
    
    ntas = int(factorial(p)/(factorial(p-k_star)*factorial(k_star)))
    print('exploring {} combinations - {}'.format(ntas, skip_to_index) )
    
    print('Setting up data generator' )
    data_gen = inp_gen(Sigma, p, k_star, k, max_trace, B)        
    for _ in range( int(skip_to_index/B) ):
        next(data_gen)
    print('Data gen is set' )
    
    if(cpus_available is None):
        cpus_available = cpu_count()-2
        cpus_available = np.min([cpus_available,90])
    print('setting up Multi Processing Pool, cpu\'s amount in use = {}'.format(cpus_available) )            
    
    print('\n==========================================\n')
    
    pool = Pool(processes = cpus_available )    
    
    # =============================================================================
    # Part 2 - Fitting starts
    # =============================================================================
    
    
    print('\n          |---------------SSPCA STARTED---------------|')
    i = skip_to_index
    stdout.write('\r ======= {0:3.2f}% done | current max = {1:2.5f} | i = {2}'.format( (100*i)/ntas, ev, i) )
    stdout.flush()
    
    for tmp_s, tmp_ev in pool.imap(SSPCA, data_gen):
        if(tmp_ev > ev):
            ev = tmp_ev
            k_entries = tmp_s
            
        if( i % 100 == 0 ):
            stdout.write('\r ======= {0:3.2f}% done | current max = {1:2.5f} | i = {2}'.format( (100*i)/ntas, ev, i) )
            stdout.flush()
            
            cp_df = pd.DataFrame(
            {
             'i': i,
             'ev': ev,
             'ks': k_entries,
             'time': clock()-start_time + cp_time,
            }).to_csv('sspca_tmp_{}_{}.csv'.format(k, k_star), index=False)
        i += B
        
    pool.close() 
    pool.join() 
    
    stdout.write('\r ======= {0:3.2f}% done | current max = {1:2.5f} | i = {2} | rtime = {3:8.0f} '.format( (100), ev, i, clock()-start_time + cp_time) )
    stdout.flush()
            
    print('\n          |---------------SSPCA FINISHED---------------|')
    
    # =============================================================================
    # Part 3 - Finalize and save results    
    # =============================================================================
    
    se_mp_var_star = ev
    se_mp_runtime = clock() - start_time + cp_time
    se_mp_k_entries = k_entries
    
    print('trace = {0:2.5f}'.format(se_mp_var_star),
          'rtime = {0:8.0f}'.format(se_mp_runtime) )

    print('sspca chose:')
    print( k_entries ) 
    
# =============================================================================
#         SAVE ALL RESULTS
# =============================================================================

    traces = [se_mp_var_star]
    rtimes = [se_mp_runtime]
    ks = [list(se_mp_k_entries)]
    alg_names = pd.Series(['sspca'], name="alg")
    
    toSave = pd.DataFrame(
    {
     'alg':alg_names,
     'traces':traces,
     'rtimes': rtimes,
     'ks': ks
    })
    
    print('\n==========================================\n')
    if('out' not in os.listdir() ):
        os.mkdir( 'out' )
        print('Generating output folder...') 
    
    if( strftime("%y_%m_%d") not in os.listdir('out/') ):
        os.mkdir( 'out/{}'.format(strftime("%y_%m_%d"))  )
        
    toSave.to_csv('out/{}/sspcaExp_kstar{}_k{}.csv'.format(strftime("%y_%m_%d"), k_star, k ), index=False )

    print('\n==========================================\n')
    print('Simulation Finished')
    
    os.remove('sspca_tmp_{}_{}.csv'.format(k, k_star))
    os.remove('sspca_tmp_sigma_{}_{}.csv'.format(k, k_star))
    
