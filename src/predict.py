from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import pandas as pd

def write_prediction(name, model, full, test_X):
    test_Y = model.predict(test_X)
    passenger_id = full[891:].PassengerId
    test = pd.DataFrame({'PassengerId': passenger_id, 'Survived': test_Y})
    # test.shape
    # print(test.head())
    test = test.astype(int)
    test.to_csv(name, index=False)


