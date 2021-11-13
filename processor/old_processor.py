# define filter sim
def simulate_filtering(data, x=[7, 7, 5, 20], out={}):
    
    # define simulation a params
    days_to_look = x[0]
    days_to_block = x[1]
    min_click_to_block = x[2]
    min_nq_pct_to_block = x[3]
    
    # define metrics variables
    total_completes = 0
    total_revenue = 0
    total_clicks = 0
    total_terms = 0
    total_nqs = 0
    
    # window to look at for nqs
    time_window = timedelta(days=days_to_look)
    
    # window to block user
    block_window = timedelta(days=days_to_block)
    
    for group_user in tqdm(list(data.keys())):    
        
        block_time = None
        is_filtered = False
        click_cache = {}
        
        for click in data[group_user]:
            
            click_time = datetime.strptime(click['click_timestamp'], DB_DATETIME)
            
            if is_filtered:
                
                if block_time + block_window < click_time:
                    is_filtered = False
                    block_time = None
                    
                else:
                    continue
                    
            total_clicks += 1
            
            # record metrics
            if click['result'] == 2:
                total_completes += 1
                total_revenue += click['cpi_cents']
                
            if click['term_timestamp'] != 0:
                total_terms += 1
                
            if click['result'] == 3:
                total_nqs += 1
                click_cache[click_time] = 1
                
            else:
                click_cache[click_time] = 0
              
            
            # remove clicks from cache when beyond time window
            to_delete = []
            for time in click_cache.keys():
                if click_time - time_window > time:
                    to_delete.append(time)
            
            for time in to_delete:
                del click_cache[time]

            nq_count = sum(click_cache.values())
            click_count = len(click_cache)
            nq_percent = round(100*nq_count/click_count)
            
            """
            if nq_percent > min_nq_pct_to_block and click_count > min_click_to_block:
                is_filtered = True
                block_time = click_time
            """
            
    print('total revenue: ${}'.format(round(total_revenue/100, 2)))
    print('total completes: {}'.format(total_completes))
    print('total clicks: {}'.format(total_clicks))
    print('ir: {}%'.format(round(100*total_completes/total_clicks, 2)))
    print('total nqs: {}'.format(total_nqs))
    print('nq rate: {}%'.format(round(100*total_nqs/total_clicks, 2)))
    
    ir = round(total_completes/total_clicks, 4)
    
    out["F"] = [total_revenue, ir, total_nqs]
    #out["G"] = [g1, g2]
    
    