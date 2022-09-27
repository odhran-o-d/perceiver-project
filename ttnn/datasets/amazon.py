import numpy as np
import pandas as pd
from catboost.datasets import amazon

from ttnn.datasets.base import BaseDataset


class AmazonDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            is_imputation=False,
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """Amazon.com - Employee Access Challenge
        https://www.kaggle.com/c/amazon-employee-access-challenge/data

        Used in CatBoost paper (we use their API for data loading).

        32769 rows, 9 features, 1 binary target column (ACTION)

        Binary classification.

        ACTION	ACTION is 1 if the resource was approved, 0 if the resource was not
        RESOURCE	An ID for each resource
        MGR_ID	The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time
        ROLE_ROLLUP_1	Company role grouping category id 1 (e.g. US Engineering)
        ROLE_ROLLUP_2	Company role grouping category id 2 (e.g. US Retail)
        ROLE_DEPTNAME	Company role department description (e.g. Retail)
        ROLE_TITLE	Company role business title description (e.g. Senior Engineering Retail Manager)
        ROLE_FAMILY_DESC	Company role family extended description (e.g. Retail Manager, Software Engineering)
        ROLE_FAMILY	Company role family description (e.g. Retail Manager)
        ROLE_CODE	Company role code; this code is unique to each role (e.g. Manager)

        Target in first column.
        """

        # Can't use the test rows, because there is no ground truth action
        self.data_table, _ = amazon()

        if isinstance(self.data_table, np.ndarray):
            pass
        elif isinstance(self.data_table, pd.DataFrame):
            self.data_table = self.data_table.to_numpy()

        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]

        self.num_target_cols = []
        self.cat_target_cols = [0]

        self.num_features = []
        self.cat_features = list(range(self.D))

        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.data_table, self.missing_matrix = self.impute_missing_entries(
            cat_features=self.cat_features, data_table=self.data_table,
            missing_matrix=self.missing_matrix)

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = []
