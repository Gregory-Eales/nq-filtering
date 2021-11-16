# import libs
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pymoo

# import modules
from multiprocessing.pool import ThreadPool
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_reference_directions, get_termination
from pymoo.optimize import minimize
import numpy as np

# import utils
from utils import *

# define optimization function
def optimize_nq_filtering(user_pct=0.01, population_size=100, n_offsprings=20, term_gens=10, n_partitions=12, n_threads=5, random_seed=0, save=True, verbose=True):
    
    '''
    this function optimizes the nq filtering algorithm on a set of data
    goal is to find the optimal filtering parameters to be used
    '''
    print_msg('loading data')
    
    data = generate_data(path='./data/partner_users_dataset.csv', user_pct=user_pct)
    
    print_msg('data loaded')
    print_msg('estimating runtime')
    est_time = estimated_run_time(data, simulate_filtering, population_size=population_size, term_gens=term_gens, n_threads=n_threads)
    print_msg('estimated runtime: {} min'.format(est_time))
    
    print_msg('initializing objects')
    # start time of experiment 
    start_time = time.time()
    
    # initialize the pool
    pool = ThreadPool(n_threads)

    # define the problem by passing the starmap interface of the thread pool
    problem = FilteringNQs(
        runner=pool.starmap,
        func_eval=starmap_parallelized_eval,
        data=data
    )
    
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)

    # optimization algorithm to use (may need to play around with this)
    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=n_offsprings,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
        ref_dirs=ref_dirs
    )

    termination = get_termination("n_gen", term_gens)
    
    print_msg('starting optimization')
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=random_seed,
                   save_history=True,
                   verbose=True
                  )
    
    print_msg('finished optimization')
    
    X = res.X
    F = res.F
    
    if save:
        
        print_msg('saving results')
        params = {
            'percent_of_users_used': '{}%'.format(round(100*user_pct, 0)),
            'population_size': population_size,
            'number_of_offspring': n_offsprings,
            'term_on_generation_n': term_gens,
            'n_partitions': n_partitions,
            'number_of_threads': n_threads,
            'random_seed': random_seed,
            'time_taken_minutes': round((time.time() - start_time)/60, 2)
        }
        
        folder_name = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        os.makedirs('./results/{}'.format(folder_name))
        
        save_optimization_results(F, X, params, folder_name=folder_name)
        
    print_msg('done')
    

def main():
    
    #optimize_nq_filtering(user_pct=0.01, population_size=20, n_offsprings=20, term_gens=20, n_threads=12, random_seed=0, save=True, verbose=True)
    
    data = generate_data(path='./data/partner_users_dataset.csv', user_pct=0.01)
    simulate_filtering(data, x=[26.44, 20.87, 13.92, 41.31, 14.89], verbose=True, filter_nqs=True)
    
    
if __name__ == '__main__':
    main()