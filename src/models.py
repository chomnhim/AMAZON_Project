import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from time import time
import gc


class UserBasedCF:
    def __init__(self, top_k=50):
        self.k = top_k
        self.similarity_matrix = None
        self.user_means = None
        self.train_csr = None
        self.global_mean = None
        
    def fit(self, train_sparse, chunk_size=500):
        start_time = time()
        print(f"\n=== Training User-Based CF (k={self.k}) ===")
        
        n_users = train_sparse.shape[0]
        self.train_csr = train_sparse.tocsr().astype(np.float32)
        self.global_mean = train_sparse.data.mean()
        
        # Compute user means
        self.user_means = np.full(n_users, self.global_mean, dtype=np.float32)
        for u in range(n_users):
            start, end = self.train_csr.indptr[u], self.train_csr.indptr[u+1]
            if end > start:
                self.user_means[u] = self.train_csr.data[start:end].mean()
        
        # Center ratings
        train_centered = self.train_csr.copy()
        for u in range(n_users):
            start, end = train_centered.indptr[u], train_centered.indptr[u+1]
            if end > start:
                train_centered.data[start:end] -= self.user_means[u]
        
        # Normalize
        norms = np.sqrt(np.array(train_centered.multiply(train_centered).sum(axis=1)).flatten())
        norms[norms == 0] = 1
        
        train_norm = train_centered.copy()
        for u in range(n_users):
            start, end = train_norm.indptr[u], train_norm.indptr[u+1]
            if end > start:
                train_norm.data[start:end] /= norms[u]
        
        # Compute top-k similarities
        self.top_k_indices = np.zeros((n_users, self.k), dtype=np.int32)
        self.top_k_values = np.zeros((n_users, self.k), dtype=np.float32)
        
        for i in range(0, n_users, chunk_size):
            end = min(i + chunk_size, n_users)
            chunk_sim = (train_norm[i:end] @ train_norm.T).toarray()
            
            for j, row_idx in enumerate(range(i, end)):
                sims = chunk_sim[j].copy()
                sims[row_idx] = -np.inf
                
                top_idx = np.argpartition(sims, -self.k)[-self.k:]
                top_idx = top_idx[np.argsort(sims[top_idx])][::-1]
                
                self.top_k_indices[row_idx] = top_idx
                self.top_k_values[row_idx] = np.maximum(sims[top_idx], 0)
        
        elapsed = time() - start_time
        print(f"✓ Training complete in {elapsed:.2f}s")
        
    def predict_all(self):
        n_users, n_items = self.train_csr.shape
        predictions = np.full((n_users, n_items), self.global_mean, dtype=np.float32)
        
        for u in range(n_users):
            neighbors = self.top_k_indices[u]
            sims = self.top_k_values[u]
            
            valid = sims > 0.01
            neighbors, sims = neighbors[valid], sims[valid]
            
            if len(neighbors) == 0:
                predictions[u, :] = self.user_means[u]
                continue
            
            neighbor_ratings = self.train_csr[neighbors].toarray()
            neighbor_means = self.user_means[neighbors]
            neighbor_centered = neighbor_ratings - neighbor_means[:, np.newaxis]
            neighbor_centered[neighbor_ratings == 0] = 0
            
            rated_mask = neighbor_ratings > 0
            
            weighted_sum = (neighbor_centered * sims[:, np.newaxis] * rated_mask).sum(axis=0)
            weight_sum = (sims[:, np.newaxis] * rated_mask).sum(axis=0)
            
            predictions[u] = np.where(weight_sum > 0,
                                      self.user_means[u] + weighted_sum / weight_sum,
                                      self.user_means[u])
        
        predictions = np.clip(predictions, 1, 5)
        return predictions


class ItemBasedCF:
    def __init__(self, top_k=50, min_support=2):
        self.k = top_k
        self.min_support = min_support
        self.sim_matrix = None
        self.train_csr = None
        self.item_means = None
        self.global_mean = None
        
    def fit(self, train_sparse, chunk_size=500):
        start_time = time()
        print(f"\n=== Training Item-Based CF (k={self.k}) ===")
        
        self.train_csr = train_sparse.tocsr()
        self.global_mean = train_sparse.data.mean()
        
        n_items = train_sparse.shape[1]
        item_user = train_sparse.T.tocsr().astype(np.float32)
        
        # Compute item means
        self.item_means = np.full(n_items, self.global_mean, dtype=np.float32)
        for i in range(n_items):
            start, end = item_user.indptr[i], item_user.indptr[i+1]
            if end > start:
                self.item_means[i] = item_user.data[start:end].mean()
        
        # Center ratings
        item_centered = item_user.copy()
        for i in range(n_items):
            start, end = item_centered.indptr[i], item_centered.indptr[i+1]
            if end > start:
                item_centered.data[start:end] -= self.item_means[i]
        
        # Normalize
        norms = np.sqrt(np.array(item_centered.multiply(item_centered).sum(axis=1)).flatten())
        norms[norms == 0] = 1e-10
        
        # Compute similarities
        rows, cols, data = [], [], []
        
        for start_idx in range(0, n_items, chunk_size):
            end_idx = min(start_idx + chunk_size, n_items)
            
            chunk = item_centered[start_idx:end_idx]
            sim_chunk = (chunk @ item_centered.T).toarray()
            
            chunk_bin = (item_user[start_idx:end_idx] > 0).astype(np.float32)
            all_bin = (item_user > 0).astype(np.float32)
            support = (chunk_bin @ all_bin.T).toarray()
            
            for i in range(sim_chunk.shape[0]):
                gidx = start_idx + i
                row_sims = sim_chunk[i] / (norms[gidx] * norms + 1e-10)
                row_sims[gidx] = 0
                
                valid = (support[i] >= self.min_support) & (row_sims > 0)
                row_sims *= valid
                
                if np.sum(row_sims > 0) > self.k:
                    top_idx = np.argpartition(row_sims, -self.k)[-self.k:]
                    top_idx = top_idx[row_sims[top_idx] > 0]
                else:
                    top_idx = np.where(row_sims > 0)[0]
                
                if len(top_idx) > 0:
                    rows.extend([gidx] * len(top_idx))
                    cols.extend(top_idx)
                    data.extend(row_sims[top_idx])
        
        self.sim_matrix = csr_matrix((data, (rows, cols)), shape=(n_items, n_items))
        
        elapsed = time() - start_time
        print(f"✓ Training complete in {elapsed:.2f}s")
        print(f"  Similarity matrix: {self.sim_matrix.nnz:,} entries")
    
    def predict_all(self):
        n_users, n_items = self.train_csr.shape
        predictions = np.full((n_users, n_items), self.global_mean, dtype=np.float32)
        
        for u in range(n_users):
            user_ratings = self.train_csr[u]
            if user_ratings.nnz == 0:
                continue
            
            weighted = (user_ratings @ self.sim_matrix).toarray().flatten()
            
            user_binary = (user_ratings > 0).astype(np.float32)
            sim_abs = self.sim_matrix.copy()
            sim_abs.data = np.abs(sim_abs.data)
            sum_sim = (user_binary @ sim_abs).toarray().flatten()
            
            sum_sim[sum_sim == 0] = 1e-10
            predictions[u] = weighted / sum_sim
            
            no_neighbors = sum_sim < 1e-9
            predictions[u, no_neighbors] = self.item_means[no_neighbors]
        
        predictions = np.clip(predictions, 1, 5)
        return predictions


class MatrixFactorizationALS:
    def __init__(self, n_factors=50, reg_param=0.01, n_epochs=20,
                 early_stopping=True, patience=3, verbose=True):
        self.k = n_factors
        self.reg = reg_param
        self.epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.P = None
        self.Q = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0
        
    def fit(self, train_sparse, val_sparse=None):
        start_time = time()
        print(f"\n=== Training ALS (k={self.k}, reg={self.reg}) ===")
        
        n_users, n_items = train_sparse.shape
        self.global_mean = train_sparse.data.mean()
        
        # Initialize parameters
        scale = 0.01
        self.P = np.random.normal(0, scale, (n_users, self.k)).astype(np.float32)
        self.Q = np.random.normal(0, scale, (n_items, self.k)).astype(np.float32)
        self.user_bias = np.zeros(n_users, dtype=np.float32)
        self.item_bias = np.zeros(n_items, dtype=np.float32)
        
        train_csr = train_sparse.tocsr()
        train_csc = train_sparse.tocsc()
        lambda_I = self.reg * np.eye(self.k, dtype=np.float32)
        
        best_val_rmse = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Update users
            for u in range(n_users):
                start, end = train_csr.indptr[u], train_csr.indptr[u+1]
                if start == end:
                    continue
                
                items = train_csr.indices[start:end]
                ratings = train_csr.data[start:end].astype(np.float32)
                
                r_adj = ratings - self.global_mean - self.user_bias[u] - self.item_bias[items]
                
                Q_i = self.Q[items]
                A = Q_i.T @ Q_i + lambda_I
                V = Q_i.T @ r_adj
                self.P[u] = np.linalg.solve(A, V)
                
                pred = self.P[u] @ Q_i.T
                self.user_bias[u] = np.mean(ratings - self.global_mean - self.item_bias[items] - pred)
            
            # Update items
            for i in range(n_items):
                start, end = train_csc.indptr[i], train_csc.indptr[i+1]
                if start == end:
                    continue
                
                users = train_csc.indices[start:end]
                ratings = train_csc.data[start:end].astype(np.float32)
                
                r_adj = ratings - self.global_mean - self.user_bias[users] - self.item_bias[i]
                
                P_u = self.P[users]
                A = P_u.T @ P_u + lambda_I
                V = P_u.T @ r_adj
                self.Q[i] = np.linalg.solve(A, V)
                
                pred = P_u @ self.Q[i]
                self.item_bias[i] = np.mean(ratings - self.global_mean - self.user_bias[users] - pred)
            
            # Validation check
            if self.verbose:
                train_rmse = self._compute_rmse(train_sparse, sample_size=5000)
                
                if val_sparse is not None and self.early_stopping:
                    val_rmse = self._compute_rmse(val_sparse)
                    print(f"  Epoch {epoch+1}/{self.epochs}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")
                    
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f"  Early stopping at epoch {epoch+1}")
                            break
                else:
                    print(f"  Epoch {epoch+1}/{self.epochs}: Train RMSE={train_rmse:.4f}")
        
        elapsed = time() - start_time
        print(f"✓ Training complete in {elapsed:.2f}s")
    
    def _compute_rmse(self, sparse_matrix, sample_size=None):
        coo = sparse_matrix.tocoo()
        
        if sample_size and len(coo.data) > sample_size:
            idx = np.random.choice(len(coo.data), sample_size, replace=False)
            preds = self.predict_pairs(coo.row[idx], coo.col[idx])
            return np.sqrt(np.mean((preds - coo.data[idx]) ** 2))
        else:
            preds = self.predict_pairs(coo.row, coo.col)
            return np.sqrt(np.mean((preds - coo.data) ** 2))
    
    def predict_pairs(self, users, items):
        preds = (self.global_mean +
                self.user_bias[users] +
                self.item_bias[items] +
                np.sum(self.P[users] * self.Q[items], axis=1))
        return np.clip(preds, 1, 5)
    
    def predict_all(self):
        pred = (self.global_mean +
               self.user_bias[:, np.newaxis] +
               self.item_bias[np.newaxis, :] +
               self.P @ self.Q.T)
        return np.clip(pred, 1, 5)


def evaluate_model(predictions, val_sparse, test_sparse, train_sparse,
                   model_name, dataset='validation', sample_users=2000):
    target_sparse = val_sparse if dataset == 'validation' else test_sparse
    
    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name} ({dataset.upper()})")
    print(f"{'='*50}")
    
    # RMSE and MAE
    test_coo = target_sparse.tocoo()
    preds = predictions[test_coo.row, test_coo.col]
    rmse = np.sqrt(np.mean((preds - test_coo.data) ** 2))
    mae = np.mean(np.abs(preds - test_coo.data))
    
    # Precision, Recall, F1 @ K
    p5, r5 = compute_precision_recall_at_k(
        predictions, target_sparse, train_sparse, k=5, sample_users=sample_users)
    p10, r10 = compute_precision_recall_at_k(
        predictions, target_sparse, train_sparse, k=10, sample_users=sample_users)
    
    f1_5 = 2 * p5 * r5 / (p5 + r5 + 1e-10)
    f1_10 = 2 * p10 * r10 / (p10 + r10 + 1e-10)
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  P@5:  {p5:.4f}, R@5: {r5:.4f}, F1@5: {f1_5:.4f}")
    print(f"  P@10: {p10:.4f}, R@10: {r10:.4f}, F1@10: {f1_10:.4f}")
    
    return {
        'model': model_name,
        'dataset': dataset,
        'rmse': rmse, 'mae': mae,
        'precision@5': p5, 'recall@5': r5, 'f1@5': f1_5,
        'precision@10': p10, 'recall@10': r10, 'f1@10': f1_10
    }


def compute_precision_recall_at_k(predictions, test_sparse, train_sparse,
                                  k=10, threshold=3.5, sample_users=2000):
    n_users = predictions.shape[0]
    precisions, recalls = [], []
    
    test_csr = test_sparse.tocsr()
    train_csr = train_sparse.tocsr()
    
    user_indices = np.where(np.array(test_csr.sum(axis=1)).flatten() > 0)[0]
    sample_users = min(sample_users, len(user_indices))
    user_indices = np.random.choice(user_indices, sample_users, replace=False)
    
    for user_idx in user_indices:
        test_row = test_csr[user_idx]
        if test_row.nnz == 0:
            continue
        
        actual_relevant = test_row.indices[test_row.data >= threshold]
        if len(actual_relevant) == 0:
            continue
        
        user_preds = predictions[user_idx].copy()
        train_items = train_csr[user_idx].indices
        user_preds[train_items] = -np.inf
        
        top_k_items = np.argsort(user_preds)[-k:][::-1]
        top_k_items = top_k_items[user_preds[top_k_items] > 0]
        
        if len(top_k_items) == 0:
            continue
        
        hits = len(np.intersect1d(top_k_items, actual_relevant))
        precisions.append(hits / len(top_k_items))
        recalls.append(hits / len(actual_relevant))
    
    return (np.mean(precisions) if precisions else 0,
            np.mean(recalls) if recalls else 0)


def recommend_for_user(user_id, predictions, train_sparse, idx_to_product,
                       user_to_idx, top_n=10):
    if user_id not in user_to_idx:
        print(f"User ID '{user_id}' not found")
        return []
    
    user_idx = user_to_idx[user_id]
    user_preds = predictions[user_idx].copy()
    
    # Filter out already rated items
    train_csr = train_sparse.tocsr()
    rated_items = train_csr[user_idx].indices
    user_preds[rated_items] = -np.inf
    
    # Get top-N items
    top_indices = np.argsort(user_preds)[-top_n:][::-1]
    
    recommendations = []
    for item_idx in top_indices:
        product_id = idx_to_product[item_idx]
        pred_rating = user_preds[item_idx]
        recommendations.append((product_id, pred_rating))
    
    return recommendations