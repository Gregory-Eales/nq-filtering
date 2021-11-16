from random import shuffle
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import time
import os


def generate_data(path='./data/partner_users_dataset.csv', user_pct = 0.2):
    
    df = pd.read_csv(path)
    df = df.fillna(0)
    
    print_msg('loaded {} rows'.format("{:,}".format(df.shape[0])))
    
    user_ids = df['user_id'].unique()
    shuffle(user_ids)
    num_users = len(user_ids)
    
    # only select a percentage of users
    num_users = int(user_pct*num_users)
    
    print_msg('{} users selected'.format(num_users))
    
    df = df[df['group_id'].isin([4, 14, 18, 22, 51, 64, 60])]    
    df = df[df['user_id'].isin(user_ids[0:num_users])]
    
    print_msg('loaded {} rows from selected users'.format("{:,}".format(df.shape[0])))
    
    df = df.sort_values(by=['click_timestamp'])
    df = df.groupby(['user_id'])
    df = df.apply(lambda x: x[['result','cpi','click_timestamp','term_timestamp','status_code_1','status_code_2', 'group_id']].to_dict('records'))
    
    data = df.to_dict()
    del df
    
    print_msg('converting to dictionary')
    for user in tqdm(list(data.keys())):
        data[user] = [list(i.values()) for i in data[user]]
    
    return data

def save_optimization_results(F, X, params, folder_name=''):
    
    with open("./results/{}/{}".format(folder_name, "params.json"), "w") as outfile:
        json.dump(params, outfile, indent=4)
    
    df = pd.DataFrame(
        data=np.concatenate((F, X), axis=1),
        columns=[
            'Negative Rev',
            'Neg IR',
            'NQ Rate',
            'days_to_look',
            'days_to_block',
            'min_click_to_block',
            'min_nq_pct_to_block',
            'min_nq_to_block',
        ]
    )
    
    df.to_csv("./results/{}/{}".format(folder_name, "results.csv"))
    
    
    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 2], -1*F[:, 0], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.xlabel("nqs")
    plt.ylabel("revenue")
    plt.savefig("./results/{}/{}".format(folder_name, 'results.png'))
    
def print_msg(msg):
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("{} | {}".format(local_time, msg))
    
def estimated_run_time(data, func, population_size=100, term_gens=10, n_threads=5):
    '''
    estimated runtime in minutes
    '''
    try:
        t = time.time()
        func(data, x=[7, 7, 5, 20, 8], verbose=False, filter_nqs=True)
        t = (time.time() - t)*population_size*term_gens
        t = round(t/60, 2)
    
    except:
        print_msg('unable to estimate runtime')
    
    return t


def main():
    data = generate_data(user_pct = 0.01)
    for row in data[list(data.keys())[0]]:
        print(row)
    
if __name__ == '__main__':
    main()