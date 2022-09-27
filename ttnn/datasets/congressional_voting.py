import pandas as pd
from ttnn.datasets.base import BaseDataset


class CongressionalVotingDataset(BaseDataset):
    def __init__(self, c):
        super(CongressionalVotingDataset, self).__init__(
            is_imputation=False, fixed_test_set_index=None)
        self.c = c

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix) = prepare_congressional_voting_dataset(
            self.c)

        # self.target_col =
        self.is_data_loaded = True

def prepare_congressional_voting_dataset(path):
    df = pd.read_csv(path, header=None)
    print(df)
    """
    TODO deal with this dataset: has the values to impute
    listed as ? marks
    """
    # df.columns = c4_df.columns.astype(str)
    # cat_features = [col for col in list(c4_df.columns)]
    # preprocessor = ColumnTransformer
    #     transformers=[
    #         ('cat', categorical_transformer, cat_features)])
    # return c4_df, preprocessor, cat_features
