import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=8675309)

# Average CV score on the training set was:0.6550057577178264
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=make_pipeline(
            StackingEstimator(estimator=XGBClassifier(learning_rate=0.001, max_depth=1, min_child_weight=1, n_estimators=50, nthread=1, objective="binary:logistic", reg_alpha=1, reg_lambda=0, subsample=0.05)),
            XGBClassifier(learning_rate=0.001, max_depth=1, min_child_weight=1, n_estimators=50, nthread=1, objective="binary:logistic", reg_alpha=5, reg_lambda=0, subsample=0.05)
        )),
        FunctionTransformer(copy)
    ),
    XGBClassifier(learning_rate=0.001, max_depth=1, min_child_weight=1, n_estimators=50, nthread=1, objective="binary:logistic", reg_alpha=1, reg_lambda=0, subsample=0.05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
