import pandas as pd
from random import shuffle


def generate_data(path='./data/nqs-2021-10-03.csv', user_pct = 0.2):
    
    df = pd.read_csv(path)
    df = df.fillna(0)
    
    print(df.shape)
    
    user_ids = df['user_id'].unique()
    shuffle(user_ids)
    num_users = len(user_ids)
    
    # only select a percentage of users
    num_users = int(user_pct*num_users)
        
    df = df[df['user_id'].isin(user_ids[0:num_users])]
    df = df.sort_values(by=['click_timestamp'])
    df = df.groupby(['group_id', 'user_id'])
    df = df.apply(lambda x: x.to_dict('records'))
    
    print(df.shape)
    data = df.to_dict()
    del df
    return data

def main():
    pass
    

if __name__ == '__main__':
    pass