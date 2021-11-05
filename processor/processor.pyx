def simulate_filtering(data):
    
    #cdef int total_completes = total_revenue = total_clicks = total_terms = total_nqs = 0
    
    cdef int total_completes = 0
    cdef int total_revenue = 0
    cdef int total_clicks = 0
    cdef int total_terms = 0
    cdef int total_nqs = 0
    
    
    for group_user in list(data.keys()):    
        
        
        is_filtered = False
        nq_window = []
        
        for click in data[group_user]:
            
            total_clicks += 1
            
            if click['result'] == 2:
                total_completes += 1
                total_revenue += click['cpi_cents']
            
            if click['result'] == 3:
                total_nqs += 1
                
            if click['term_timestamp'] != 0:
                total_terms += 1

                
    print('total revenue: ${}'.format(round(total_revenue/100, 2)))
    print('total completes: {}'.format(total_completes))
    print('total clicks: {}'.format(total_clicks))
    print('epc: {}%'.format(round(100*total_completes/total_clicks, 2)))
    print('total nqs: {}'.format(total_nqs))
    print('nq rate: {}%'.format(round(100*total_nqs/total_clicks, 2)))