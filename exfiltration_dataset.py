import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from pathlib import Path
import os
from typing import Union
import pickle

ROOT = Path('drive/MyDrive/ColabNotebooks/dns_exfiltration')
DATASET_PATH = ROOT / 'data' / 'dataset_ws_10_v3_clean.csv'
NEW_ATTACKS_PATH = ROOT / 'data' / 'dataset_ws_10_new_v3.csv'

class ExfiltrationDataset(object):
    # TODO fix docs
    """Container for the DNS exfiltration dataset.

    Attributes
    ----------
    COLUMN_NAMES : list[str]
        Class attribute with the names the columns in the csv's.
    CSV_FEATURE_INDICES: dict[str, slice]
        Dictionary with keys 'all', 'individual', 'aggregated', containing 
        slices that specify which csv columns should be read to get the
        corresponding feature type. Class attribute.
    IGNORED_DOMAINS : list[str]
        Class attribute with the list of domains known to use legitimate
        exfiltration requests. Class attribute.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler that produces zero-mean and unit-variance features. Initially
        set to None, fit during first call to the `standardize` method.
    dataset_path : str
        Path to the csv with the unmodified examples.
    dataset_path_modified : str
        Path to the csv with modified exfiltration examples.
    csv_rows : numpy.ndarray
        1d array with the indices of rows that were read from the csv with the
        original examples.
    csv_rows_modified : numpy.ndarray
        1d array with the indices of rows that were read from the csv with the
        modified exfiltrations.
    X : numpy.ndarray
        2d array with the features.
    y : numpy.ndarray
        1d array with the labels.
    feature_type : str
        The type of features used ('aggregated', 'individual', or 
        'all').

    Methods
    -------
    create_splits
        Used to randomly sort csv rows into the train, val and test sets. Returns
        two pandas.Series objects - one for the original examples and the other
        for the modified exfiltrations.
    save_splits
        Write splits to disk.
    load_splits
        Read splits from disk.
    standardize
        Fits the scaler attribute and standardizes the features.
    get_X_y
        Returns the predictor matrix X and the label vector y.
    get_raw_requests
        Reads the actual request strings from the csv's.
    """

    # The names of all columns in the csv's (not all are always loaded).
    CSV_COLUMN_NAMES = [
        'user_ip', 'domain', 'timestamp', 'attack', 'request',
        'len', 'subdomains_count', 'w_count', 'w_max', 'entropy', 
        'w_max_ratio', 'w_count_ratio', 'digits_ratio', 'uppercase_ratio',
        'time_avg', 'time_stdev', 'size_avg', 'size_stdev', 'throughput', 
        'unique', 'entropy_avg', 'entropy_stdev'
    ]

    # Index ranges for different types of feature groups
    CSV_FEATURE_INDICES = {
        'all': slice(5, 22),
        'individual': slice(5, 14),
        'aggregated': slice(14, 22)
    }

    # These domains (probably) use exfiltration and are ignored in our research.
    IGNORED_DOMAINS = {'mcafee.com', 'e5.sk'}

    @classmethod
    def _get_csv_split(
        cls, 
        path: str, 
        val_size: Union[int, float], 
        test_size: Union[int, float], 
        train_size: Union[int, float, None] = None, 
        random_state: Union[int, None] = None, 
        n_rows: Union[int, None] = None, 
        stratify: bool = False
    ) -> pd.Series:
        """Split the rows of a csv file into train, val and test sets.

        Parameters
        ----------
        path : str
            Location of the csv file on disk.
        val_size : Union[int, float]
            Size of the validation set: a float in the range [0, 1) signifying
            the fraction of the total number of examples, or an integer
            signifying the absolute number of examples.
        test_size : Union[int, float]
            Size of the test set: a float in the range [0, 1) signifying
            the fraction of the total number of examples, or an integer
            signifying the absolute number of examples.
        train_size : Union[int, float, None], optional
            Size of the train set: a float in the range [0, 1) signifying
            the fraction of the total number of examples, or an integer
            signifying the absolute number of examples. If None, the train set
            is comprised of all examples that didn't go into either the val or 
            the test set. By default None.
        random_state : Union[int, None], optional
            Controls the shuffling applied to the data before applying the split. 
            Pass an int for reproducible output across multiple function calls.  
            By default None.
        n_rows : Union[int, None], optional
            If not None, only the first n_rows csv rows are considered. 
            By default None
        stratify : bool, optional
            If not None, data is split in a stratified fashion, using the values
            of the 'attack' column. By default False.

        Returns
        -------
        pandas.Series
            Categorical series where index designates the csv row and values
            are 'train', 'val', 'test' or NaN (for unused examples).
        """
        # Load the 'attack' column of csv w/ unmodified examples
        y = np.genfromtxt(
            path, dtype=bool, usecols=cls.CSV_COLUMN_NAMES.index('attack'), 
            delimiter=',', max_rows=n_rows
        )

        # Allocate the series that holds the split and initialize all to 'train'
        SplitCategoical = pd.CategoricalDtype(categories=['train', 'val', 'test'])
        split = pd.Series(np.nan, index=range(len(y)), dtype=SplitCategoical)
        idx_unassigned = split.index

        # If specified as fractions of the original set, map to integers
        if isinstance(train_size, float):
            train_size = int(len(y) * train_size)
        if isinstance(val_size, float):
            val_size = int(len(y) * val_size)
        if isinstance(test_size, float):
            test_size = int(len(y) * test_size)

        # Choose indices of test examples
        if test_size > 0:
            idx_unassigned, idx_test = train_test_split(
                idx_unassigned,
                test_size=test_size,
                stratify=y if stratify else None,
                random_state=random_state
            )
            split[idx_test] = 'test'
        
        # Choose indices of validation examples
        if val_size > 0:
            idx_unassigned, idx_val = train_test_split(
                idx_unassigned,
                test_size=val_size,
                stratify=y[idx_unassigned] if stratify else None,
                random_state=random_state
            )
            split[idx_val] = 'val'

        # If train_size wasn't specified, 
        # or if it's bigger than the number of remaining examples, 
        # then all remaining samples go into the train set
        if (train_size is None) or (train_size >= len(idx_unassigned)):
            split[idx_unassigned] = 'train'
        # Otherwise, first check if  the specified train size is positive
        elif train_size > 0:
            idx_unassigned, idx_train = train_test_split(
                idx_unassigned,
                test_size=train_size,
                stratify=y[idx_unassigned] if stratify else None,
                random_state=random_state
            )
            split[idx_train] = 'train'

        return split

    @classmethod
    def create_splits(
        cls, 
        csv_path: str, 
        csv_path_modified: str,
        val_size: Union[int, float], 
        test_size: Union[int, float], 
        val_size_modified: Union[int, float], 
        test_size_modified: Union[int, float],
        train_size: Union[int, float, None] = None, 
        train_size_modified: Union[int, float, None] = None, 
        n_csv_rows: Union[int, None] = None, 
        n_csv_rows_modified: Union[int, None] = None, 
        random_state: Union[int, None] = None
    ):
        """Split the csv rows into train, val and test sets.

        Parameters
        ----------
        csv_path : str
            Location of the csv with the original examples.
        csv_path_modified : str
            Location of the csv with the modified exfiltrations.
        val_size : Union[int, float]
            Determines the number of original examples in the validation set.
            Must be a float in the range [0, 1) signifying
            the fraction of the total number rows in the csv, or an integer
            signifying the absolute number of examples.
        test_size : Union[int, float]
            Determines the number of original examples in the test set.
            Must be a float in the range [0, 1) signifying
            the fraction of the total number rows in the csv, or an integer
            signifying the absolute number of examples.
        val_size_modified : Union[int, float]
            Determines the number of modified exfiltrations in the validation set.
            Must be a float in the range [0, 1) signifying
            the fraction of the total number rows in the csv, or an integer
            signifying the absolute number of examples.
        test_size_modified : Union[int, float]
            Determines the number of modified exfiltrations in the test set.
            Must be a float in the range [0, 1) signifying
            the fraction of the total number rows in the csv, or an integer
            signifying the absolute number of examples.
        train_size : Union[int, float, None], optional
            Determines the number of original examples in the train set.
            Must be a float in the range [0, 1) signifying
            the fraction of the total number rows in the csv, or an integer
            signifying the absolute number of examples. If None, the train set
            is comprised of all examples that didn't go into either the val or 
            the test set. By default None.
        train_size_modified : Union[int, float, None], optional
            Determines the number of original examples in the train set.
            Must be a float in the range [0, 1) signifying
            the fraction of the total number rows in the csv, or an integer
            signifying the absolute number of examples. If None, the train set
            is comprised of all examples that didn't go into either the val or 
            the test set. By default None.
        n_csv_rows : Union[int, None], optional
            If not None, only the first n_rows rows in the csv with the original
            examples are considered. By default None.
        n_csv_rows_modified : Union[int, None], optional
            If not None, only the first n_rows rows in the csv with the modified
            exfiltrations are considered. By default None.
        random_state : Union[int, None], optional
            Controls the shuffling applied to the data before applying the split. 
            Pass an int for reproducible output across multiple function calls.  
            By default None.

        Returns
        -------
        tuple[pandas.Series, pandas.Series]
            Categorical series where index designates the csv row, and values
            are 'train', 'val', 'test' or NaN (for unused examples). The first
            series splits the csv with the original examples, the second series
            splits the csv with the modified exfiltrations.
        """
        csv_row_split = cls._get_csv_split(
            csv_path, val_size, test_size, train_size, random_state, 
            n_csv_rows, stratify=True
        )
        csv_row_split_modified = cls._get_csv_split(
            csv_path_modified, val_size_modified, test_size_modified,
            train_size_modified, random_state, n_csv_rows_modified, stratify=False
        )
        return csv_row_split, csv_row_split_modified

    @staticmethod
    def save_splits(
        path: str, 
        csv_row_split: pd.Series, 
        csv_row_split_modified: pd.Series
    ):
        """Save to path the pandas Series that split the csv's.
        """

        with open(path, 'wb') as f:
            pickle.dump((csv_row_split, csv_row_split_modified), f)

    @staticmethod
    def load_splits(path: str) -> tuple[pd.Series, pd.Series]:
        """Load from path the two pandas Series that split the csv's.
        """
        with open(path, 'rb') as f:
            csv_row_split, csv_row_split_modified = pickle.load(f)
        return csv_row_split, csv_row_split_modified

    @staticmethod
    def _fill_from_csv(
        dst: Union[np.ndarray, tuple[np.ndarray, np.ndarray]], 
        path: str, 
        rows: np.ndarray, 
        columns: Union[np.ndarray, slice], 
        verbose: bool = False
    ):
        """Fill arrays by reading data from a csv.

        Parameters
        ----------
        dst : Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
            Reference to the arrays where the data should be written. If a 
            tuple of numpy arrays is passed, then the first arrays is assumed
            to be the matrix of features, and the second array is assumed to 
            be the vector of labels.
        path : str
            Location of the csv file on disk.
        rows : np.ndarray
            The csv rows to be read.
        columns : Union[np.ndarray, tuple[np.ndarray, int]]
            The csv columns to be read. If dst is a tuple, then the second
            element specifies the index of the column that holds the labels.
        verbose : bool, optional
            Verbosity of execution, by default False.
        """
        # Read rows and columns from csv given by path and fill into dst
        rows_ = np.sort(rows)
        with open(path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            # Head indexes rows_ and dst
            head = 0
            for i, line in enumerate(csv_reader):
                # Read and overwrite until you reach rows_[head]
                if i < rows_[head]:
                    continue
                # When current row i reaches rows_[head], get the data
                if type(dst) is tuple and type(columns) is tuple:
                    # In this case, I'm assuming we are filling X and y
                    dst[0][head] = line[columns[0]]
                    dst[1][head] = (line[columns[1]] == 'True')
                else:
                    dst[head] = line[columns]
                if (head % 10000 == 0) and verbose:
                    print(f'\rReading row {head:8} of {len(rows_):8}, {100*head/len(rows_):5.2f}% completed', end='')
                head += 1
                if head == len(rows_):
                    break
        if verbose:
            print('')

    def __init__(
        self, 
        feature_type: str, 
        dataset_path: str, 
        dataset_path_modified: str, 
        split: str, 
        csv_splitter: pd.Series, 
        csv_splitter_modified: pd.Series,
        verbose: bool = False
    ):
        """Read the csv's and instantiate ExfiltrationDataset.

        Parameters
        ----------
        feature_type : str
            Must be 'aggregated', 'individual', or 'all'.
        dataset_path : str
            Path to the original DNS exfiltration dataset, containing examples
            prior to exfiltrator modification.
        dataset_path_modified : str
            Path to the csv with the modified exfiltration examples.
        split: str
            'train', 'val' or 'test'
        csv_splitter: pandas.Series
            Specifies the distribution of rows of the csv with the original
            examples over the train, val and test sets. Index holds the csv rows, 
            and values are 'train', 'val', 'test' or NaN (for unused examples).
        csv_splitter_modified: pandas.Series
            Specifies the distribution of rows of the csv with the modified
            exfiltrations over the train, val and test sets. Index holds the csv 
            rows, and values are 'train', 'val', 'test' or NaN (for unused examples).
        verbose: bool, optional
            Verbosity of execution, defaults to False.
        """
        
        # Assert that both dataset.csv and dataset_modified.csv exist in the
        # specified directory

        assert os.path.exists(dataset_path) and os.path.exists(dataset_path_modified), \
            "Required files not available at the specified path!"

        # Figure out which csv rows to read
        if split == 'all':
            csv_rows = csv_splitter.index.values
            csv_rows_modified = csv_splitter_modified.index.values    
        else:
            csv_rows = csv_splitter[csv_splitter == split].index.values
            csv_rows_modified = csv_splitter_modified[csv_splitter_modified == split].index.values

        # Read features X and labels y from csv
        n_examples = len(csv_rows) + len(csv_rows_modified)
        # Read only labels for original examples - we know modified examples
        # are all exfiltrations and have 'attack' set to 'True'
        X_cols = self.CSV_FEATURE_INDICES[feature_type]
        y_col = self.CSV_COLUMN_NAMES.index('attack')
        # Allocate
        X = np.empty(
            shape=(len(csv_rows) + len(csv_rows_modified), X_cols.stop - X_cols.start), 
            dtype=np.float32
        )
        y = np.full((n_examples,), True)
        if verbose:
            print('Reading original examples:')
        self._fill_from_csv((X, y), dataset_path, csv_rows, (X_cols, y_col), verbose)
        if verbose:
            print('Reading features of modified examples:')
        self._fill_from_csv(X[len(csv_rows):], dataset_path_modified, csv_rows_modified, X_cols)

        self.scaler = None
        self.feature_type = feature_type
        self.dataset_path = dataset_path
        self.dataset_path_modified = dataset_path_modified
        self.csv_rows = csv_rows
        self.csv_rows_modified = csv_rows_modified
        self.X = X
        self.y = y


    def standardize(self, scaler: Union[StandardScaler, None] = None):
        """Fit and apply a standradizer to the numerical features.

        When this method is invoked for the first time with the scaler parameter 
        set to None, a standardizer is fit and applied on X.  This results in numerical features with roughly zero mean and 
        unit variance. The resulting standardizer is saved as the `scaler` attribute.
        If there are no numerical features (for example, if only
        raw request strings were loaded), invoking this method has no effect.

        Parameters
        ----------
        scaler : sklearn.preprocessing.StandardScaler or None, optional
            If a standardizer object is passed, then it is used to standardize
            the data. Otherwise, if the `scaler` attribute is None, a new scaler 
            is instantiated, fitted and saved as the `scaler` attribute. Defaults
            to None.
        """
        if scaler is not None:
            scaler_ = scaler
        else:
            if self.scaler is None:
                self.scaler = StandardScaler(copy=False).fit(self.X)
            scaler_ = self.scaler
        self.X = scaler_.transform(self.X)


    def get_raw_requests(self, idx):
        """Returns raw requests strings for the examples specified by `idx`.

        Parameters
        ----------
        idx : pandas.Index
            The indices of the examples for which the raw requests should be
            returned.

        Returns
        -------
        pandas.Series
            The raw request strings that correspond to the specified indices.
        """

        # Allocate
        req = np.empty(len(idx), dtype=object)

        # Reading from csv's returns sorted row indices
        idx_s = np.sort(idx)

        # Indices into X and y that correspond to unmodified examples
        idx_ = idx_s[idx_s < len(self.csv_rows)]
        # Corresponding indices into csv rows
        rows = self.csv_rows[idx_]
        # Just need the one column
        column = self.CSV_COLUMN_NAMES.index('request')
        # Lookup requests from csv
        self._fill_from_csv(req, self.dataset_path, rows, column)

        # The same for modified examples, just remember to offset the
        # index into X WRT index into csv
        n_filled = len(idx_)
        idx_ = idx_s[idx_s >= len(self.csv_rows)] - len(self.csv_rows)
        rows = self.csv_rows_modified[idx_] 
        self._fill_from_csv(req[n_filled:], self.dataset_path_modified, rows, column)

        # Return series with the original index ordering
        req = pd.Series(data=req, index=idx_s)
        return req[idx]

    def get_X_y(
        self, 
        exfiltration_type:str = 'both'
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the feature matrix X and the label vector y for the specified 
        type of examples.

        Parameters
        ----------
        exfiltration_type : str
            Specifies the type of examples that should be returned: 
            'modified' returns just the modified exfiltrations,
            'unmodified' returns just the original examples,
            'both' returns all examples.

        Returns
        -------
        X, y
            The feature matrix and the label vector.

        Raises
        ------
        ValueError
            If exfiltration_type is not either 'both', 'modified' nor 
            'unmodified'.
        """
        n_unmodified = len(self.csv_rows)
        if exfiltration_type == 'both':
            # Return all examples
            return self.X, self.y
        elif exfiltration_type == 'modified':
            # Return only modified examples
            return self.X[n_unmodified:], self.y[n_unmodified:]
        elif exfiltration_type == 'unmodified':
            # Return only original examples
            return self.X[:n_unmodified], self.y[:n_unmodified]
        else:
            raise ValueError("'exfiltration_type' must be 'modified', 'unmodified' or 'both'.")


# %%

if __name__ == '__main__':
    root = Path('..')
    csv_path_modified = root / 'data' / 'dataset_ws_10_new_v3.csv'
    splits_path = root / 'data' / 'csv_row_splits.pkl'
    csv_path = root / 'data' / 'dataset_ws_10_v3_clean.csv'
    csv_url = "https://drive.google.com/open?id=1_tlzJGY1XVT0yiYMrfJfzs3vKz3cajNG&authuser=milatadic777%40gmail.com&usp=drive_fs"
    csv_modified_url = "https://drive.google.com/open?id=1jF2y6wmjZre15hLFrAkUXl9q2Rr5O5J_&authuser=milatadic777%40gmail.com&usp=drive_fs"
    feature_type = 'all' 
    n_rows = 10000
    n_rows_modified = 1000

    test_splitting = True
    test_standardization = False
    test_raw_request_getter = False
    
    # Test splitting
    if test_splitting:
        split, split_modified = ExfiltrationDataset.create_splits(
            csv_path, csv_path_modified, 
            val_size=0.1, test_size=0.1, 
            val_size_modified=0, test_size_modified=0.6,
            train_size=0.1, train_size_modified=0.4,
            n_csv_rows = n_rows, n_csv_rows_modified=n_rows_modified,
            random_state=0
        )
        print(f'csv with unmodified examples')
        print(split.value_counts())
        print(f'csv with modified exfiltrations')
        print(split_modified.value_counts())
        ExfiltrationDataset.save_splits(splits_path, split, split_modified)

    # Make a dataset
    splitter, splitter_modified = ExfiltrationDataset.load_splits(splits_path)
    ds = ExfiltrationDataset(
        feature_type, csv_path, csv_path_modified, 
        'val', splitter, splitter_modified, verbose=True
    )
    print(f'X.shape = {ds.X.shape}, y.shape={ds.y.shape}')
    
    # Test standardization
    if test_standardization:
        print(f'Before standardization: means = {np.mean(ds.X, axis=0)}')
        ds.standardize()
        print(f'After standardization: means = {np.mean(ds.X, axis=0)}')

    # Test reading raw requests from disk
    if test_raw_request_getter:
        idx = pd.Index([0, n_rows, 2, 3, n_rows+2])
        requests = ds.get_raw_requests(idx)
        print(requests)

    pass
