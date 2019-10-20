import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=8675309)

# Average CV score on the training set was:0.6374984992279318
exported_pipeline = XGBClassifier(learning_rate=0.001, max_depth=1, min_child_weight=20, n_estimators=150, nthread=1, objective="binary:logistic", reg_alpha=1, reg_lambda=5, subsample=1.0)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
