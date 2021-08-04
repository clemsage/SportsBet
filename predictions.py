from sklearn import preprocessing, compose
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import json
import os
import warnings


def dataset_preprocessing(dataset, label_encoder=None, feature_preprocessor=None):
    Y = dataset['result']
    X = dataset.drop('result', axis='columns')

    # Transform categorical labels (H, D, A) into numerical values
    if label_encoder is None:
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(Y)
    Y = label_encoder.transform(Y)

    if feature_preprocessor is None:
        # print(X.info())
        feature_preprocessor = compose.ColumnTransformer(transformers=[
            ('cat', preprocessing.OneHotEncoder(), compose.make_column_selector(dtype_include="object")),
            ('num', preprocessing.StandardScaler(), compose.make_column_selector(dtype_exclude="object"))
        ])
        feature_preprocessor.fit(X)
    X = feature_preprocessor.transform(X)

    return X, Y, label_encoder, feature_preprocessor


class HomeTeamWins:
    def __init__(self):
        self.label_encoder = None

    def infer(self, dataset):
        Y_pred = self.label_encoder.transform(['H'] * len(dataset))
        return Y_pred


class BestRankedTeamWins:
    def __init__(self):
        self.label_encoder = None

    def infer(self, dataset):
        Y_pred = pd.Series(['A'] * len(dataset), index=dataset.index)
        home_team_has_best_ranking = dataset['HomeRanking'] < dataset['AwayRanking']
        Y_pred[home_team_has_best_ranking] = 'H'
        Y_pred = self.label_encoder.transform(Y_pred)
        return Y_pred


class ResultsPredictor(object):
    def __init__(self, league, args):
        # TODO: Model selection and hyperparameter tuning
        if args.config_name is None:
            args.config_name = os.path.join('configs', '%s.json' % args.model_name)
        with open(args.config_name, 'r') as f:
            config = json.load(f)
        self.model = eval(args.model_name)(**config)
        print('\nModel for predicting the game results: %s' % self.model.__class__.__name__)
        self.league = league
        self.training_dataset, self.test_dataset = self.split_train_test_sets()
        self.label_encoder = None
        self.feature_preprocessor = None
        self.baselines = [HomeTeamWins(), BestRankedTeamWins()]

    def split_train_test_sets(self):
        """
        Train on the N-1 first seasons and evaluate on the last season
        """
        training_dataset = []
        for i in range(max(1, len(self.league.seasons) - 1)):
            training_dataset.append(self.league.datasets[self.league.seasons[i].name])
        training_dataset = pd.concat(training_dataset)
        test_dataset = self.league.datasets[self.league.seasons[-1].name]
        if len(self.league.datasets) == 1:
            warnings.warn('The test season is also the training season as only one season is provided')
        return training_dataset, test_dataset

    def train(self):
        X, Y, self.label_encoder, self.feature_preprocessor = dataset_preprocessing(self.training_dataset)
        print('Feature set size: %d' % X.shape[1])
        print('\nTraining of the model on %d samples...' % len(self.training_dataset))
        self.model.fit(X, Y)
        Y_pred = self.model.predict(X)
        print('Training accuracy of the model: %.3f' % accuracy_score(Y, Y_pred))

    def eval(self, with_heuristics=True):
        print('Evaluation of the model on %d samples...' % len(self.test_dataset))
        X, Y, _, _ = dataset_preprocessing(self.test_dataset, self.label_encoder, self.feature_preprocessor)
        Y_pred = self.model.predict(X)
        print('Test accuracy of the model: %.3f' % accuracy_score(Y, Y_pred))

        if with_heuristics:
            for baseline in self.baselines:
                baseline.label_encoder = self.label_encoder
                Y_pred = baseline.infer(self.test_dataset)
                print('Test accuracy of the %s heuristic: %.3f' % (baseline.__class__.__name__,
                                                                   accuracy_score(Y, Y_pred)))

    def infer(self, dataset, with_proba=False):
        X, _, _, _ = dataset_preprocessing(dataset, self.label_encoder, self.feature_preprocessor)
        if with_proba:
            Y_pred = self.model.predict_proba(X)
            Y_pred = pd.DataFrame(Y_pred, columns=self.label_encoder.classes_, index=dataset.index)
            Y_pred['result'] = Y_pred.idxmax(axis=1)
        else:
            Y_pred = self.model.predict(X)
            Y_pred = self.label_encoder.inverse_transform(Y_pred)
            Y_pred = pd.DataFrame(Y_pred, columns=['result'], index=dataset.index)
        return Y_pred
