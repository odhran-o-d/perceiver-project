import torch
import torch.nn as nn
from einops import repeat


class CategoricalEmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """
    def __init__(
            self, dim_embedding, time_code, dim_feature=2):
        super().__init__()

        self.time_code = time_code
        self.total_features = len(self.time_code[0]) + len(self.time_code[1])

        self.in_embedding = nn.ModuleList([
            nn.Embedding(dim_feature, dim_embedding)
            for _ in range(self.total_features)])

    def forward(self, X):
        X = [self.in_embedding[t](torch.unsqueeze(X[:, t], 1))
             for t in self.time_code[0]]
        for time_subset in self.time_code[1:]:
            X += [self.in_embedding[self.time_code[1][i]](torch.unsqueeze(X[:, t], 1))
                  for i, t in enumerate(time_subset)]
        return torch.stack(X, 1)


# class ContinuousEmbeddingGenerator(torch.nn.Module):
#     """
#     Classical embeddings generator
#     """
#     def __init__(
#             self, dim_embedding, num_features, dim_feature=7):
#         super().__init__()

#         self.time_code = time_code
#         self.total_features = len(self.time_code[0]) + len(self.time_code[1])

#         self.in_embedding = nn.ModuleList([
#             nn.Linear(dim_feature, dim_embedding)
#             for _ in range(self.total_features)])

#     def forward(self, X):
#         X = [self.in_embedding[t](torch.unsqueeze(X[:, t], 1))
#              for t in self.time_code[0]]
#         for time_subset in self.time_code[1:]:
#             X += [self.in_embedding[self.time_code[1][i]](torch.unsqueeze(X[:, t], 1))
#                   for i, t in enumerate(time_subset)]
#         return torch.stack(X, 1)


class Feature_embedding(torch.nn.Module):
    '''supports binary encoded data provided by
    the FIDDLE preprocessing system'''

    def __init__(
            self, dim_embedding, time_code, device):
        super().__init__()

        self.time_code = time_code
        total_features = len(self.time_code[0]) + len(self.time_code[1])
        self.time_series_start = self.time_code[1][0]
        self.feature_indices = torch.arange(
            total_features, device=device, dtype='long')
        self.feature_index_embedding = nn.Embedding(
            total_features, dim_embedding)

    def forward(self, X):
        feature_index_embeddings = self.feature_index_embedding(self.feature_indices)
        feature_index_embeddings = torch.cat((
            feature_index_embeddings[0:self.time_series_start],
            feature_index_embeddings[self.time_series_start:].repeat(
                len(self.time_code) - 1, 1)))

        feature_index_embeddings = torch.unsqueeze(
            feature_index_embeddings, 0)
        feature_index_embeddings = feature_index_embeddings.repeat(
            X.size(0), 1, 1)

        return X + feature_index_embeddings


class Time_embedding(torch.nn.Module):
    '''supports binary encoded data provided by
    the FIDDLE preprocessing system'''

    def __init__(
            self, dim_embedding, time_code, device):
        super().__init__()
        self.time_code = time_code
        self.time_indices = torch.arange(
            len(self.time_code), device=device, dtype='long')              
        self.time_index_embedding = nn.Embedding(
            len(self.time_code), dim_embedding)

    def forward(self, X):
        time_index_embeddings = self.time_index_embedding(self.time_indices)
        time_index_embeddings = torch.cat(([
            repeat(time_index_embeddings[i], 'd -> n d', n=len(t))
            for i, t in enumerate(self.time_code)]))

        time_index_embeddings = torch.unsqueeze(
            time_index_embeddings, 0)
        time_index_embeddings = time_index_embeddings.repeat(
            X.size(0), 1, 1)

        return X + time_index_embeddings
