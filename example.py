
import logging

import numpy as np
import pandas as pd

from nicdam import NICDAM, EnsembleMember


logging.basicConfig(level=logging.INFO)


n = 10000
data_batch_size = 1


class DummyRegressor:
    def __init__(self):
        self.estimator_type = 'batch'
        self.model = None

    def fit_model(self, data, labels):
        pass
        if self.estimator_type == 'batch':
            self.n_data_rows_model_trained_on = data.shape[0]
        elif self.estimator_type == 'incremental':
            self.n_data_rows_model_trained_on += data.shape[0]

    def predict(self, data):
        return data * 2

    def instantiate_estimator_model(self):
        self.model = 'DummyRegressorModel'

    def preprocess_data(self, data):
        return data

    def score(self, true_labels, pred_labels):
        pass
        return 0.9


stable_member = EnsembleMember(
    estimator=DummyRegressor(),
    cols_data=['col1', 'col2'],  # feature columns
    col_label=['target'],  # target column
    init_weight=1.0,
    instantiation_t=0)

nicdam = NICDAM(
    data_batch_size=data_batch_size,
    stable_member=stable_member,
    min_weight_stable_member=0.4,
    max_trained_on_data_rows=1000,
    maturity_threshold_reactive_members=500,
    max_n_reactive_members=3,
    accretion_rate_weights_reactive_members=0.01)

X_pre = pd.DataFrame(np.random.rand(1000, 2), columns=['col1', 'col2'])
X_pre['target'] = X_pre['col1'] * 2 + X_pre['col2'] * 3

nicdam.fit_stable_member(data=X_pre.loc[:, ['col1', 'col2']], labels=X_pre.loc[:, ['target']])

np.random.seed(0)
X = pd.DataFrame(np.random.rand(n, 2), columns=['col1', 'col2'])
X['target'] = X['col1'] * 2 + X['col2'] * 3

concept_drifts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]  # artificial concept drifts


for i in range(1, n):
    # prediction = ...
    # error = ...

    if i in concept_drifts:
        print(f'Concept drift at {i}')
        nicdam.add_reactive_member(
            estimator=DummyRegressor(),
            instantiation_t=i,
            rotate_members='challenge stable or drop oldest')

    nicdam.fit_incremental(
        addendum_data=X.iloc[i:i + data_batch_size].loc[:, ['col1', 'col2']],
        addendum_labels=X.iloc[i:i + data_batch_size].loc[:, ['target']])

print('Stream terminated.')
