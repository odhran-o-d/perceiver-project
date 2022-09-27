import numpy as np
import pandas as pd

from ttnn.datasets.base import BaseDataset


def get_ecoli_dataset():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "ecoli/ecoli.data")
    data = pd.read_csv(url, header=None, sep='\s+')
    features = data.iloc[:, 1:-1]
    features = pd.get_dummies(features)
    labels = data.iloc[:, -1]
    labels = labels.astype('category').cat.codes
    return features, labels


class EcoliDataset(BaseDataset):
    def __init__(self, c):
        super(EcoliDataset, self).__init__(
            is_imputation=False,
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """E coli dataset.

        5. Number of Instances:  336 for the E.coli dataset and


        6. Number of Attributes.
                 for E.coli dataset:  8 ( 7 predictive, 1 name )


        7. Attribute Information.

          1.  Sequence Name: Accession number for the SWISS-PROT database
          2.  mcg: McGeoch's method for signal sequence recognition.
          3.  gvh: von Heijne's method for signal sequence recognition.
          4.  lip: von Heijne's Signal Peptidase II consensus sequence score.
                   Binary attribute.
          5.  chg: Presence of charge on N-terminus of predicted lipoproteins.
               Binary attribute.
          6.  aac: score of discriminant analysis of the amino acid content of
               outer membrane and periplasmic proteins.
          7. alm1: score of the ALOM membrane spanning region prediction program.
          8. alm2: score of ALOM program after excluding putative cleavable signal
               regions from the sequence.

        8. Missing Attribute Values: None.

        9. Class Distribution. The class is the localization site. Please see Nakai &
                       Kanehisa referenced above for more details.

          cp  (cytoplasm)                                    143
          im  (inner membrane without signal sequence)        77
          pp  (perisplasm)                                    52
          imU (inner membrane, uncleavable signal sequence)   35
          om  (outer membrane)                                20
          omL (outer membrane lipoprotein)                     5
          imL (inner membrane lipoprotein)                     2
          imS (inner membrane, cleavable signal sequence)      2
        """
        X, y = get_ecoli_dataset()
        X = X.to_numpy()
        y = np.expand_dims(y.to_numpy(), -1)
        self.data_table = np.concatenate([X, y], axis=-1)
        print('... done.')

        # For now, drop the three least populated classes to avoid errors.
        # Classes: 2, 3, 6
        excluded_classes = [2, 3, 6]
        for class_label in excluded_classes:
            valid_rows = self.data_table[:, -1] != class_label
            self.data_table = self.data_table[valid_rows]

        self.N, self.D = self.data_table.shape

        # Multi-class classification
        self.num_target_cols = []
        self.cat_target_cols = [7]

        self.cat_features = [7]
        self.num_features = list(range(0, self.D - 1))
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)

        self.is_data_loaded = True
