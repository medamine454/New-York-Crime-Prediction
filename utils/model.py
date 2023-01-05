import xgboost as xgb
import pickle
import lightgbm as lgbm


def crime_name(crime_code):
    """
    :param crime_code: the output of the machine learning model [0, 1, 2]
    :return: the type of the crime predicted converted in letters
    """
    if crime_code == 0:
        return "MISDEMEANOR"
    elif crime_code == 1:
        return "FELONY"
    return "VIOLATION"


class Model:
    def __init__(self, model_path, model_name):
        booster = xgb.Booster()
        if model_name == "xgboost":
            self.model = pickle.load(open(model_path, 'rb'))
        if model_name == "lgbm":
            self.model = lgbm.Booster(model_file=model_path)
