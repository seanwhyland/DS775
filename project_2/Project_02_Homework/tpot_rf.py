import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=8675309)

# Average CV score on the training set was:0.4416143787401016
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=20, min_samples_split=20, n_estimators=10)),
    RandomForestRegressor(bootstrap=True, max_features=0.9, min_samples_leaf=10, min_samples_split=11, n_estimators=150)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
