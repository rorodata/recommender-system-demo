import numpy as np
import pandas as pd
import json

class recommender():
    def __init__(self):
        self.item_sim=None
        self.popular_items=None
        self.ratings_df=None
    
    def fit(self, ratings_df, user_id, item_id, ratings):
        assert type(ratings_df) == pd.core.frame.DataFrame
        #assert set([user_id, item_id, rating]) < set(ratings_df.columns)
        self.ratings_df=ratings_df.copy()
        self.ratings_df=self.ratings_df.rename(columns={user_id: 'user_id', item_id: 'item_id', ratings: 'ratings'})
        ratings_pivot=self.ratings_df.pivot(index='user_id', columns='item_id', values='ratings').transpose()
        
        for i in ratings_pivot.index:
            ratings_pivot.loc[i,:].fillna((ratings_pivot.loc[i,:].mean()), inplace=True)
        
        self.item_frequency=self.ratings_df.dropna()['item_id'].value_counts()
        self.ratings_pivot=ratings_pivot.copy() # remove later...not used
        self.item_sim=ratings_pivot.transpose().corr().copy()
        
        self.min_rating=self.ratings_df['ratings'].min()
        self.max_rating=self.ratings_df['ratings'].max()
        self.find_popular_items()
        
    def find_popular_items(self):
        self.popular_items=self.ratings_df.groupby(['item_id'])['ratings'].mean().sort_values(ascending=False)
    
    def score(self,user_id, item_id, Nmax=20):
        assert Nmax > 1
        
        items_rated_by_user=self.ratings_df[self.ratings_df['user_id']==user_id].dropna()

        if items_rated_by_user.empty:
            popular = self.popular_items.index[0] 
            return popular
        
        
        item_sim_ratings=pd.DataFrame(self.item_sim.loc[item_id]).reset_index()
        item_sim_ratings.columns=['item_id', 'sim']
        
        df_temp=items_rated_by_user.merge(item_sim_ratings).sort_values('sim', ascending=False).iloc[0:Nmax]
        #retval= np.average(df_temp['ratings'], weights=df_temp['sim'])
        
        #this compensates for pathelogical cases where negative correltions dominate
        ret_num = (df_temp['ratings'] * df_temp['sim']).sum()
        ret_den = df_temp['sim'].abs().sum()
        retval= ret_num/(1.0*ret_den)
        
        return np.clip(retval, self.min_rating, self.max_rating)
    
    def items_to_search(self, user_id, k=50):
        items_rated_by_user=self.ratings_df[self.ratings_df['user_id']==user_id].dropna()['item_id']
        items_not_rated_by_user=set(self.ratings_df['item_id'])-set(items_rated_by_user)
        data=[self.item_frequency[i] for i in items_not_rated_by_user]
        topk=pd.Series(data=data, index=items_not_rated_by_user).nlargest(k).index
        
        #return list(items_not_rated_by_user)
        return list(topk)
        
    
    def calculate_all_item_suggestions(self, user_id, max_suggestions=30):
        item_search_list=self.items_to_search(user_id, k=max_suggestions)
        scores={}
        for item_id in item_search_list:
            s= self.score(user_id,item_id, 30) #Nmax=30
            scores[item_id]=s
        return pd.Series(scores)
    
    def reco_topk_items_for_user(self, user_id, k=10, ret_json=False):
        """
        inputs:
            user_id - id of user for which recommendations are being requested
            k - number of suggestions to return
        outputs
            item_id, predicted rating  - for top k recommended items
        """
        try:
            retval=self.calculate_all_item_suggestions(user_id).nlargest(k)
            if ret_json:
                return retval.to_json()
            else:
                return retval
        except:
            print('error has occured')
            return -1

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

    this_reco=recommender()
    this_reco.fit(movies100k_df, user_id='user_id', item_id='movie_id', ratings='rating')
    
    joblib.dump(this_reco, "/volumes/data/recommender-model.pkl")

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load("/volumes/data/recommender-model.pkl")
    return _model

def predict(user_id):
    reco_model = load_model()
    try:
        return reco_model.reco_topk_items_for_user(user_id=user_id)
    except:
        return []

if __name__ == '__main__':
    train()
