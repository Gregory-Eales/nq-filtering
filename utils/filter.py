from pymoo.core.problem import ElementwiseProblem, starmap_parallelized_eval
from tqdm import tqdm
import numpy as np
import json


def save_json(data, filename='filename'):
    with open("./results/{}/{}.json".format(folder_name, filename), "w") as outfile:
        json.dump(data, outfile, indent=4)

def simulate_filtering(data, x=[7, 7, 5, 20, 8], verbose=True, filter_nqs=True):
    
    group_ids = [4, 18, 22, 64, 60]

    data_map = {
        'result': 0,
        'cpi': 1,
        'click_timestamp': 2,
        'term_timestamp': 3,
        'status_code_1': 4,
        'status_code_2': 5,
        'group_id': 6,
    }
    
    # define simulation params
    days_to_look = x[0]
    days_to_block = x[1]
    min_nq_pct_to_block = x[2]
    min_nq_to_block = x[3]
    max_term_time = x[4]
    
    # define metrics variables
    total_completes = 0
    total_revenue = 0
    total_clicks = 0
    total_terms = 0
    total_nqs = 0

    max_completes = 0
    max_clicks = 0
    max_revenue = 0
    max_nqs = 0
    
    # predefine sim variables
    result = 0
    click_time = 0
    term_time = 0
    cpi = 0
    
    # window to look at for nqs
    time_window = days_to_look*24*60*60
    
    # window to block users
    block_window = days_to_block*24*60*60
    
    term_counts = {}
    user_filters = {}

    user_group_filters = {}
    block_times = {}
    last_blocks = {}
    
    for user in tqdm(list(data.keys())):    
        
        block_time = {}
        is_filtered = {}
        click_cache = {}
        click_hist = {}

        for group_id in group_ids:
            click_hist[group_id] = []
            click_cache[group_id] = {}
            block_time[group_id] = None
            is_filtered[group_id] = False

        for click in data[user]:
            
            result = click[data_map['result']]
            click_time = click[data_map['click_timestamp']]
            term_time = click[data_map['term_timestamp']]
            cpi = click[data_map['cpi']]
            group_id = click[data_map['group_id']]
            
            status_code_1 = click[data_map['status_code_1']]
            status_code_2 = click[data_map['status_code_2']]
            
            max_clicks += 1

            if result == 2:
                max_revenue += cpi
                max_completes += 1
                
            if (result == 3 or result == 8 or result == 14) and term_time-click_time < max_term_time:
                max_nqs += 1
            
            if is_filtered[group_id]:
                
                if block_time[group_id] + block_window < click_time:
                    
                    to_delete = []
                    for time in click_cache[group_id].keys():
                        if block_time[group_id] > time:
                            to_delete.append(time)
                            
                    # delete clicks outsite of time to look window
                    for time in to_delete:
                        del click_cache[group_id][time]

                    is_filtered[group_id] = False
                    block_time[group_id] = None

                        
                else:
                    
                    if term_time != 0:
                        
                        if user not in user_filters:
                            user_filters[user] = 0
                        

                        if verbose:
                            user_filters[user] += 1
                            
                            # use group id as key
                            if group_id not in term_counts:
                                term_counts[group_id] = {}

                            if status_code_1 not in term_counts[group_id]:
                                 term_counts[group_id][status_code_1] = 0

                            term_counts[group_id][status_code_1] += 1
                    
                    continue
                    
            total_clicks += 1
            
            # record metrics
            if result == 2:
                total_completes += 1
                total_revenue += cpi
                click_hist[group_id].append(1)
            
            else:
                click_hist[group_id].append(0)

            if  term_time != 0:
                total_terms += 1
                
            if (result == 3 or result == 8 or result == 14) and term_time-click_time < max_term_time:
                total_nqs += 1
                click_cache[group_id][click_time] = 1
                
            else:
                click_cache[group_id][click_time] = 0
                
               
            # rules not applied when no term exists
            if result == 0 or term_time == 0:
                continue
            
            
            if filter_nqs:
                # remove clicks from cache when beyond time window
                to_delete = []
                for time in click_cache[group_id].keys():
                    if click_time - time_window > time:
                        to_delete.append(time)
                        
                # delete clicks outsite of time to look window
                for time in to_delete:
                    del click_cache[group_id][time]

                nq_count = sum(click_cache[group_id].values())
                click_count = len(click_cache[group_id])

                nq_percent = round(100*nq_count/click_count)
                
                #ir = round(100*complete_count/click_count)

                if nq_percent > min_nq_pct_to_block and nq_count > min_nq_to_block:
                    is_filtered[group_id] = True
                    block_time[group_id] = click_time

                    # need to check how many parnters are blocked
                    num_blocks = 0
                    for group_id in group_ids:
                        if is_filtered[group_id]:
                            num_blocks += 1

                    if user not in user_group_filters:
                        user_group_filters[user] = num_blocks
                        block_times[user] = click_time

                    if user_group_filters[user] < num_blocks:
                        user_group_filters[user] = num_blocks
                        block_times[user] = click_time


                    # unblock user if they have no more partners left
                    if num_blocks >= len(group_ids):

                        group_irs = {}
                        for group_id in group_ids:
                            group_irs[group_id] = sum(click_hist[group_id][-100:]) / 100

                        group_irs = {k: v for k, v in sorted(group_irs.items(), key=lambda item: item[1])}

                        print(group_irs)
                        best_group_id = list(group_irs.keys()[-1])

                        is_filtered[nest_group_id] = False
                        block_time[best_group_id] = None

    results = {}
                                        
    if verbose:

        print('-'*40)

        print('actual revenue: ${:,}'.format(round(max_revenue, 2)))
        print('actual clicks: {:,}'.format(max_clicks))
        print('actual completes: {:,}'.format(max_completes))
        print('actual ir: {}'.format(round(max_completes/max_clicks, 2)))
        print('actual epc: {}'.format(round(max_revenue/max_clicks, 2)))
        print('actual nqs: {:,}'.format(round(max_nqs, 2)))
        print('actual nq rate: {}%'.format(round(100*max_nqs/max_clicks, 2)))

        print('-'*40)

        print('sim revenue: ${:,}'.format(round(total_revenue, 2)))
        print('sim clicks: {:,}'.format(total_clicks))
        print('sim completes: {:,}'.format(total_completes))
        print('sim ir: {}'.format(round(total_completes/total_clicks, 2)))
        print('sim epc: {}'.format(round(total_revenue/total_clicks, 2)))
        print('sim nqs: {:,}'.format(total_nqs))
        print('sim nq rate: {}%'.format(round(100*total_nqs/total_clicks, 2)))

        print('-'*40)

        print('revenue retained: {}%'.format(round(100*total_revenue/max_revenue, 2)))
        print('nqs retained: {}%'.format(round(100*total_nqs/max_nqs, 2)))
        print('nqs filtered: {:,}'.format(max_nqs-total_nqs))

        print('-'*40)

        user_group_filters = {k: v for k, v in sorted(user_group_filters.items(), key=lambda item: item[1])}
        print(user_group_filters)
        print(block_times)

        '''
        save_json(term_counts, filename='term_counts')
        save_json(
            dict(sorted(user_filters.items(), key=lambda item: item[1])),
            filename='user_filter_counts'
            )
        '''

    # sim stats
    results['revenue'] = round(total_revenue, 2)
    results['clicks'] = total_clicks
    results['completes'] = total_completes
    results['ir'] = round(total_completes/total_clicks, 2)
    results['epc'] = round(total_revenue/total_clicks, 2)
    results['nqs'] = round(total_nqs, 2)
    results['nq_rate'] = round(100*total_nqs/total_clicks, 2)

    # filtering params
    results['days_to_look'] = days_to_look
    results['days_to_block'] = days_to_block
    results['min_nq_pct_to_block'] = min_nq_pct_to_block
    results['min_nq_to_block'] = min_nq_to_block
    results['max_term_time'] = max_term_time
    
    # optimization params
    results['pct_revenue'] = round(total_revenue/max_revenue, 4) 
    results['pct_nq'] = round(total_nqs/max_nqs, 4)
 
    return results


# define problem
class FilteringNQs(ElementwiseProblem):
    
    '''
    days_to_look = x[0]
    days_to_block = x[1]
    min_nq_pct_to_block = x[2]
    min_nq_to_block = x[3]
    max_term_time = x[4]
    '''
    
    def __init__(self, data, **kwargs):
        super().__init__(n_var=5, # number of variables to use for optimization
                         n_obj=3, # number of objectives to optimize for
                         n_constr=1, # number of constraints on the output
                         xl=np.array([1, 1, 25, 25, 10]), # lower bound for params
                         xu=np.array([30, 30, 100, 100, 60]), # upper bound for params,
                         type_var = np.uint,
                         **kwargs,
                        )
        
        self.data = data
      
    def _evaluate(self, x, out, *args, **kwargs):
        
        results = simulate_filtering(self.data, x=x, verbose=False, filter_nqs=True)
        
        f1 = -1 * results['pct_revenue'] # need to make metrics negative for minimization 
        f2 = -1 * results['ir']
        f3 = results['pct_nq']
        
        out["F"] = [f1, f2, f3]
        out["G"] = [0.75 + f1]






