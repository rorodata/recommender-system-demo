import joblib

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load("/volumes/data/recommender-model.pkl")
    return _model

def predict(user_id):
    reco_model = load_model()
    try:
        retval = reco_model.reco_topk_items_for_user(user_id=user_id).to_dict()
        return [{'item_id':int(k), 'rating':float(v)} for (k,v) in retval.items()]
    except:
        return []
