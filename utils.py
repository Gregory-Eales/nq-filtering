import pandas as pd
from random import shuffle
from tqdm import tqdm


def generate_data(path='./data/partner_users_dataset.csv', user_pct = 0.2):
    
    df = pd.read_csv(path)
    df = df.fillna(0)
    
    print('loaded {} rows'.format("{:,}".format(df.shape[0])))
    
    user_ids = df['user_id'].unique()
    shuffle(user_ids)
    num_users = len(user_ids)
    
    # only select a percentage of users
    num_users = int(user_pct*num_users)
    
    print(df.shape)
    df = df[df['group_id'].isin([4, 14, 18, 22, 51, 64, 60])]    
    print(df.shape)
    
    df = df[df['user_id'].isin(user_ids[0:num_users])]
    df = df.sort_values(by=['click_timestamp'])
    df = df.groupby(['user_id'])
    df = df.apply(lambda x: x[['result','cpi','click_timestamp','term_timestamp','status_code_1','status_code_2', 'group_id']].to_dict('records'))
    
    data = df.to_dict()
    del df
    
    for user in tqdm(list(data.keys())):
        data[user] = [list(i.values()) for i in data[user]]
    
    return data


def main():
    data = generate_data(user_pct = 0.01)
    for row in data[list(data.keys())[0]]:
        print(row)
    
if __name__ == '__main__':
    main()