import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple
from exfiltration_dataset import ExfiltrationDataset
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle
import pprint


class ExfiltrationClassifier(object):

    def __init__(
        self, 
        classifier_type: str, 
        datasets: Dict[str, ExfiltrationDataset], 
        verbose: bool = False
    ):
        """Classifier that trains and evaluates on ExfiltrationDataset objects.

        Parameters
        ----------
        classifier_type : str
            Type of model used. Must be one of the following:
            'logistic regression', 'svm' (support vector machine with the
            Gaussian kernel), 'sgd' (support vector machine optimized with
            stochastic gradient descent), 'naive Bayes', 'decision tree',
            'random forest', 'extra trees', 'adaboost', 'hgb' (Histogram-based
            gradient boosting), 'mlp' (multi-layer perceptron), 'xgb' (XGBoost).
        datasets : dict[str, ExfiltrationDataset]
            Dictionary with keys 'train', 'val' and 'test', and objects of type
            ExfiltrationDataset as the corresponding values.
        verbose : bool, optional
            Verbosity of training, by default False.
        """
        if classifier_type == 'logistic regression':
            self.clf = LogisticRegression(
                penalty='none',
                dual=False,
                C=1,
                class_weight='balanced',
                solver='lbfgs',
                verbose=verbose
            )
        # This works ok, regardless of C, but only for l2 + squared_hinge
        elif classifier_type == 'svm':
            self.clf = SVC(
                C=1,
                kernel='rbf',
                class_weight=None,
                verbose=verbose
            )
        # Works ok, fastest with l2 + no class weights, doesn't converge w/o penalty for balanced weights
        elif classifier_type == 'sgd':
            self.clf = SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=0.0001,
                class_weight=None,
                verbose=verbose
            )
        elif classifier_type == 'naive Bayes':
            self.clf = GaussianNB()
        elif classifier_type == 'decision tree':
            self.clf = DecisionTreeClassifier(
                class_weight='balanced',
                criterion='entropy',
                splitter='best',
                random_state=0,
            )
        elif classifier_type == 'random forest':
            self.clf = RandomForestClassifier(
                n_estimators=5,
                class_weight=None,
                criterion='entropy',
                random_state=0,
                verbose=verbose
            )
        elif classifier_type == 'extra trees':
            self.clf = ExtraTreesClassifier(
                n_estimators=5,
                class_weight=None,
                criterion='gini',
                random_state=0,
                verbose=verbose
            )
        elif classifier_type == 'adaboost':
            self.clf = AdaBoostClassifier(
                n_estimators=20,
                random_state=0
            )
        elif classifier_type == 'hgb':
            self.clf = HistGradientBoostingClassifier(
                random_state=0,
                verbose=verbose
            )
        elif classifier_type == 'mlp':
            self.clf = MLPClassifier(
                hidden_layer_sizes=(20,30,10),
                verbose=verbose
            )
        elif classifier_type == 'xgb':
            self.clf = xgb.XGBClassifier(
                n_estimators = 100,
                objective='binary:logistic',
                tree_method='gpu_hist',
                sampling_method='gradient_based',
                verbosity=1 if verbose else 0,
                subsample=0.1,
                scale_pos_weight=1,
                predictor='gpu_predictor',
                eval_metric='error',
                early_stopping_rounds=10,
                use_label_encoder=False
            )
        else:
            self.clf = None
        self.ds = datasets
    
    def train(self, include_modified=True):
        """Train the classifier.

        Parameters
        ----------
        include_modified : bool, optional
            If False, train only on original examples and don't include the
            modified exfiltration. By default True.
        """
        exfiltration_type = 'both' if include_modified else 'unmodified'
        X, y = self.ds['train'].get_X_y(exfiltration_type)
        if type(self.clf) == xgb.XGBRegressor:
            X_val, y_val = self.ds['val'].get_X_y(exfiltration_type)
            self.clf.fit(
                X, y.astype(int), 
                eval_set=[(X_val, y_val.astype(int))]
            )
        else:
            self.clf.fit(X, y)

    def evaluate(
        self, 
        split: str = 'test', 
        exfiltration_type: str = 'both', 
        return_errors: bool = False
    ) -> Union[dict, Tuple[dict, np.ndarray, np.ndarray]]:
        """Get performance metrics for the trained classifier.

        Parameters
        ----------
        split : str, optional
            The data to evaluate the classifier on. Must be either 'train', 
            'test' or 'val'. By default 'test'.
        exfiltration_type : str, optional
            The type of examples to evaluate on. Must be either 'unmodified' to
            evaluate only on original examples, 'modified' to evaluate only on
            modified exfiltrations, or 'both' to evaluate on all data from the
            specified split. By default 'both'.
        return_errors : bool, optional
            If True, returns the features and the raw requests of the erroneously
            classified example. By default False.

        Returns
        -------
        Union[dict, tuple[dict, np.ndarray, np.ndarray]]
            If return_errors is False, returns just a dictionary with structure
            {'accuracy': float, 'f1-score': {'False': float, 'True': float, 'macro avg': float}}
            If return_errors is True, returns also the feature matrix and raw
            requests for the erroneously classified examples.  
        """
        # Classify test examples
        X, y_true = self.ds[split].get_X_y(exfiltration_type)
        y_pred = self.clf.predict(X)
        
        # Evaluate metrics
        if exfiltration_type == 'modified':
            # We have only attack here
            report = {'accuracy': accuracy_score(y_true, y_pred)}
        else:
            # In this case we have both classes
            report = classification_report(y_true, y_pred, output_dict=True)
            report = {
                'f1-score': {
                    key: report[key]['f1-score'] for key in ['False', 'True', 'macro avg']
                },
                'accuracy': report['accuracy']
            }

        # Get errors
        if return_errors:
            idx_err = np.where(y_pred != y_true)[0]
            X_err = self.ds['train'].scaler.inverse_transform(self.ds[split].X[idx_err], copy=True)
            requests_err = self.ds[split].get_raw_requests(idx_err)
            return report, X_err, requests_err
        else:
            return report


if __name__ == '__main__':
    root = Path('..')
    csv_path_modified = root / 'data' / 'dataset_ws_10_new_v3.csv'
    splits_path = root / 'data' / 'csv_row_splits.pkl'
    csv_path = root / 'data' / 'dataset_ws_10_v3_clean.csv'
    exfilt_dataset_small_path = root / 'data' / 'exfilt_dataset_small_win.pkl'

    feature_type = 'all' 
    n_rows = None
    n_rows_modified = None
    make_new_split = True
    save_new_split_to_disk = False

    # Get objects that assign csv rows across splits
    if make_new_split:
        splitter, splitter_modified = ExfiltrationDataset.create_splits(
            csv_path, csv_path_modified, val_size=0.05, test_size=0.05,
            val_size_modified=0.05, test_size_modified=0.6,
            train_size=0.1, train_size_modified=0.35,
            n_csv_rows=n_rows, n_csv_rows_modified=n_rows_modified,
            random_state=0
        )
        if save_new_split_to_disk:
            ExfiltrationDataset.save_splits(splits_path, splitter, splitter_modified)
    else:
        splitter, splitter_modified = ExfiltrationDataset.load_splits(splits_path)

    # Make datasets
    ds = {}
    for split in ['train', 'test']:
        ds[split] = ExfiltrationDataset(
            feature_type, csv_path, csv_path_modified, 
            split, splitter, splitter_modified, verbose=True
        )
        print(f'{split}: X.shape = {ds[split].X.shape}, y.shape={ds[split].y.shape}')
    with open(root / 'data' / 'exfilt_dataset_small_win.pkl', 'wb') as f:
        pickle.dump(ds, f)

    # Standardization
    ds['train'].standardize()
    ds['test'].standardize(ds['train'].scaler)

    # Instantiate classifier
    clf = ExfiltrationClassifier(
        'logistic regression', ds, verbose=True
    )

    # Train only on unmodified examples?
    clf.train(include_modified=True)

    # Evaluate
    pp = pprint.PrettyPrinter()
    report = clf.evaluate('test', 'unmodified')
    print('Evaluation on original examples:')
    pp.pprint(report)
    report = clf.evaluate('test', 'modified')
    print('Evaluation on modified exfiltrations:')
    pp.pprint(report)
    report = clf.evaluate('test', 'both')
    print('Evaluation on all examples:')
    pp.pprint(report)