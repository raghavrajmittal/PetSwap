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