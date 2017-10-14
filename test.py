"""Script to test the recommender system API
"""
import firefly

api = firefly.Client("https://recommender-system-demo.rorocloud.io/")

result = api.predict(user_id=25)
print(result)
