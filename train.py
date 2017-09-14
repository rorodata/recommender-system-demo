import pandas as pd
from recommender import Recommender

def train():
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('/volumes/data/u.user', sep='|', names=u_cols,  encoding='latin-1')

    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('/volumes/data/u.data', sep='\t', names=r_cols,  encoding='latin-1')

    #Reading items file:
    i_cols = ['movie_id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
     'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('/volumes/data/u.item', sep='|', names=i_cols,  encoding='latin-1')

    movies100k_df = pd.merge(pd.merge(ratings, users), items)[['user_id', 'movie_id', 'rating']]

    this_reco=Recommender()
    this_reco.fit(movies100k_df, user_id='user_id', item_id='movie_id', ratings='rating')

    joblib.dump(this_reco, "/volumes/data/recommender-model.pkl")

if __name__ == '__main__':
    train()
