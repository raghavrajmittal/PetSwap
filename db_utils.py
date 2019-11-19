from pymongo import MongoClient
from pprint import pprint
import urllib
from config import Config


def get_client():
    config = Config()
    mongoURI = (
        "mongodb+srv://"
        + urllib.parse.quote_plus(config.username)
        + ":"
        + urllib.parse.quote_plus(config.password)
        + "@petswap-ac9x2.mongodb.net/test?retryWrites=true&w=majority"
    )
    client = MongoClient(mongoURI, ssl=True)
    return client


def insert_features(features):
    client = get_client()
    db = client.database
    for feature in features:
        try:
            insert_one_feature(db, feature)
        except Exception as e:
            print(e)


def insert_one_feature(db, feature_rep):
    if feature_rep[1] != "cat" and feature_rep[1] != "dog":
        raise Exception("Type should be either cat or dog.")
    else:
        if db.features.find({"file": feature_rep[0]}).count() == 1:
            result = db.features.replace_one(
                {"file": feature_rep[0]},
                {   "file": feature_rep[0],
                    "type": feature_rep[1],
                    "features": feature_rep[2],
                    "cluster_id": feature_rep[3],
                },
            )
            print("Entry updated")
        else:
            feature = {
                "file": feature_rep[0],
                "type": feature_rep[1],
                "features": feature_rep[2],
                "cluster_id": feature_rep[3],
            }
            result = db.features.insert_one(feature)
            print("Entry created as " + str(result.inserted_id))


def find_features(db, cluster_id):
    return list(db.features.find({"cluster_id": cluster_id}))


if __name__ == "__main__":
    client = get_client()
    db = client.database
    insert_features([("1", "cat", "no", "1")])
    print(find_features(db, "1"))
