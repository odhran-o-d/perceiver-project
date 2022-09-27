import numpy as np

from ttnn.datasets.base import BaseDataset
from ttnn.datasets.ogbn_utils import load_ogb_fixed_split_dataset


class OGBNArxivDataset(BaseDataset):
    """
    Via https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv.

    [1] Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong,
        and Anshul Kanakia. Microsoft academic graph: When experts are not
        enough. Quantitative Science Studies, 1(1):396–413, 2020.
    [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean.
        Distributed representationsof words and phrases and their
        compositionality. In Advances in Neural Information Processing Systems
        (NeurIPS), pp. 3111–3119, 2013.

    40-class classification (40 subject areas of arXiv CS papers).
    Last column is categorical target column,
    first 128 columns are continuous skip-gram features.
    169,343 rows in total.
    Fixed train, val, test set corresponding to 2017, 2018, 2019 publications.

    NOTE than in TTNN, our intention is to attempt to learn the paper-paper
    relationships pertinent to the subject area prediction task directly from
    the node features, i.e. we do not use the edges, which in this case
    are citations linking the papers.

    See detailed explanation below:

        The ogbn-arxiv dataset is a directed graph, representing the citation
    network between all Computer Science (CS) arXiv papers indexed by MAG [1].

    Each node is an arXiv paper and each directed edge indicates that one
    paper cites another one. Each paper comes with a 128-dimensional feature
    vector obtained by averaging the embeddings of words in its title and
    abstract. The embeddings of individual words are computed by running the
    skip-gram model [2] over the MAG corpus. We also provide the mapping
    from MAG paper IDs into the raw texts of titles and abstracts here. In
    addition, all papers are also associated with the year that the
    corresponding paper was published.

    Prediction task: The task is to predict the 40 subject areas of arXiv CS
    papers, e.g., cs.AI, cs.LG, and cs.OS, which are manually determined
    (i.e., labeled) by the paper’s authors and arXiv moderators. With the
    volume of scientific publications doubling every 12 years over the past
    century, it is practically important to automatically classify each
    publication’s areas and topics. Formally, the task is to predict the
    primary categories of the arXiv papers, which is formulated as a
    40-class classification problem.

    Dataset splitting: We consider a realistic data split based on the
    publication dates of the papers. The general setting is that the ML
    models are trained on existing papers and then used to predict the
    subject areas of newly-published papers, which supports the direct
    application of them into real-world scenarios, such as helping the
    arXiv moderators.

    Specifically, we propose to train on papers published until 2017,
    validate on those published in 2018, and test on those published
    since 2019.
    """
    def __init__(self, c):
        super(OGBNArxivDataset, self).__init__(
            is_imputation=False,
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix, self.fixed_split_indices) = (
            self.load_ogbn_arxiv_dataset())

        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]
        self.is_data_loaded = True
        self.tmp_file_names = []  # TODO: actually delete tmp files

    def load_ogbn_arxiv_dataset(self):
        graph, label, fixed_split_indices = load_ogb_fixed_split_dataset(
            name='ogbn-arxiv', data_path=self.c.data_path + '/ogbn-arxiv')
        data_table = np.hstack((graph['node_feat'], label))
        N, D = data_table.shape
        cat_features = [D - 1]
        num_features = list(range(D - 1))
        missing_matrix = np.zeros((N, D), dtype=np.bool_)
        return (
            data_table, N, D, cat_features, num_features, missing_matrix,
            fixed_split_indices)

