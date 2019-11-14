from pymongo import MongoClient
from pprint import pprint
import urllib

def get_client():
    mongoURI = (
        "mongodb+srv://"
        + urllib.parse.quote_plus("cv_admin")
        + ":"
        + urllib.parse.quote_plus("databasesAreCool")
        + "@petswap-ac9x2.mongodb.net/test?retryWrites=true&w=majority"
    )
    client = MongoClient(mongoURI, ssl=True)
    return client

def insert_features(features):
    client = get_client()
    db = client.database
    for feature in features:
        insert_one_feature(db, feature)

def insert_one_feature(db, feature_rep):
    feature = {'file': feature_rep[0], 'type': feature_rep[1], 'features': feature_rep[2], 'cluster_id': feature_rep[3]}
    result = db.features.insert_one(feature)
    print("Entry created as " + str(result.inserted_id))

'''
client = get_client()
db = client.database
insert_one_feature(db, ("test", "test", "test"))
'''