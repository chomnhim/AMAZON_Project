import numpy as np
from collections import Counter
from datetime import datetime
from scipy.sparse import csr_matrix, save_npz, load_npz
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath, max_rows=None):
    try:
        data = np.genfromtxt(
            filepath, 
            delimiter=',', 
            skip_header=1,
            dtype=[('user_id', 'U50'), ('product_id', 'U50'), 
                   ('rating', 'f8'), ('timestamp', 'i8')],
            encoding='utf-8',
            max_rows=max_rows
        )
        print(f"✓ Loaded {len(data):,} ratings successfully")
        return data
    except Exception as e:
        print(f"Error loading with genfromtxt: {e}")
        print("Trying alternative method...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]
        
        data_list = []
        limit = max_rows if max_rows else len(lines)
        
        for line in lines[:limit]:
            parts = line.strip().split(',')
            if len(parts) == 4:
                try:
                    data_list.append((
                        parts[0],
                        parts[1],
                        float(parts[2]),
                        int(parts[3])
                    ))
                except:
                    continue
        
        data = np.array(data_list, dtype=[('user_id', 'U50'), ('product_id', 'U50'), 
                                          ('rating', 'f8'), ('timestamp', 'i8')])
        print(f"✓ Loaded {len(data):,} ratings using alternative method")
        return data


def clean_data(data):
    user_ids = np.array([row[0] for row in data])
    product_ids = np.array([row[1] for row in data])
    ratings = np.array([row[2] for row in data])
    timestamps = np.array([row[3] for row in data])
    
    original_size = len(ratings)
    print(f"\n=== DATA CLEANING ===")
    print(f"Original size: {original_size:,}")
    
    # Remove invalid ratings
    invalid_mask = np.logical_or(ratings < 1, ratings > 5)
    n_invalid = np.sum(invalid_mask)
    
    if n_invalid > 0:
        print(f"  Removing {n_invalid} invalid ratings...")
        valid_mask = ~invalid_mask
        user_ids = user_ids[valid_mask]
        product_ids = product_ids[valid_mask]
        ratings = ratings[valid_mask]
        timestamps = timestamps[valid_mask]
    
    # Remove duplicates (keep latest)
    print(f"  Checking for duplicates...")
    sorted_indices = np.argsort(timestamps)[::-1]
    user_ids = user_ids[sorted_indices]
    product_ids = product_ids[sorted_indices]
    ratings = ratings[sorted_indices]
    timestamps = timestamps[sorted_indices]
    
    seen_pairs = set()
    keep_indices = []
    
    for i in range(len(user_ids)):
        pair = (user_ids[i], product_ids[i])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            keep_indices.append(i)
    
    keep_indices = np.array(keep_indices)
    user_ids = user_ids[keep_indices]
    product_ids = product_ids[keep_indices]
    ratings = ratings[keep_indices]
    timestamps = timestamps[keep_indices]
    
    print(f"  After cleaning: {len(ratings):,}")
    print(f"  Removed: {original_size - len(ratings):,} ({(original_size - len(ratings))/original_size*100:.2f}%)")
    
    return user_ids, product_ids, ratings, timestamps


def filter_data(user_ids, product_ids, ratings, timestamps, 
                min_user_ratings=10, min_product_ratings=10):
    print(f"\n=== DATA FILTERING ===")
    print(f"Criteria: min_user_ratings={min_user_ratings}, min_product_ratings={min_product_ratings}")
    
    iteration = 0
    prev_size = len(ratings)
    
    while True:
        iteration += 1
        
        user_counts = Counter(user_ids)
        product_counts = Counter(product_ids)
        
        valid_users = {user for user, count in user_counts.items() 
                       if count >= min_user_ratings}
        user_mask = np.array([uid in valid_users for uid in user_ids])
        
        valid_products = {prod for prod, count in product_counts.items() 
                         if count >= min_product_ratings}
        product_mask = np.array([pid in valid_products for pid in product_ids])
        
        valid_mask = user_mask & product_mask
        
        user_ids = user_ids[valid_mask]
        product_ids = product_ids[valid_mask]
        ratings = ratings[valid_mask]
        timestamps = timestamps[valid_mask]
        
        n_users = len(np.unique(user_ids))
        n_products = len(np.unique(product_ids))
        
        print(f"  Iteration {iteration}: {len(ratings):,} ratings, {n_users:,} users, {n_products:,} products")
        
        if len(ratings) == prev_size or iteration > 10:
            break
        prev_size = len(ratings)
    
    print(f"✓ Filtering complete")
    return user_ids, product_ids, ratings, timestamps


def create_mappings(user_ids, product_ids):
    unique_users = np.unique(user_ids)
    unique_products = np.unique(product_ids)
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    product_to_idx = {prod: idx for idx, prod in enumerate(unique_products)}
    
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_product = {idx: prod for prod, idx in product_to_idx.items()}
    
    user_indices = np.array([user_to_idx[uid] for uid in user_ids])
    product_indices = np.array([product_to_idx[pid] for pid in product_ids])
    
    print(f"\n✓ Created mappings:")
    print(f"  Users: 0 to {len(unique_users)-1}")
    print(f"  Products: 0 to {len(unique_products)-1}")
    
    return user_to_idx, product_to_idx, idx_to_user, idx_to_product, user_indices, product_indices


def create_rating_matrix(user_indices, product_indices, ratings):
    n_users = len(np.unique(user_indices))
    n_products = len(np.unique(product_indices))
    
    rating_matrix = np.zeros((n_users, n_products), dtype=np.float32)
    rating_matrix[user_indices, product_indices] = ratings
    
    non_zero = np.count_nonzero(rating_matrix)
    sparsity = 1 - (non_zero / (n_users * n_products))
    
    print(f"\n✓ Created rating matrix:")
    print(f"  Shape: {rating_matrix.shape}")
    print(f"  Non-zero entries: {non_zero:,}")
    print(f"  Sparsity: {sparsity*100:.6f}%")
    
    return rating_matrix


def split_data(user_indices, product_indices, ratings, 
               train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)
    
    n_users = len(np.unique(user_indices))
    n_products = len(np.unique(product_indices))
    
    print(f"\n=== DATA SPLITTING ===")
    print(f"Ratios: Train {train_ratio*100:.0f}% | Val {val_ratio*100:.0f}% | Test {test_ratio*100:.0f}%")
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for u_idx in range(n_users):
        user_ratings_idx = np.where(user_indices == u_idx)[0]
        n_user_ratings = len(user_ratings_idx)
        
        if n_user_ratings >= 3:
            shuffled = np.random.permutation(user_ratings_idx)
            
            n_test = max(1, int(n_user_ratings * test_ratio))
            n_val = max(1, int(n_user_ratings * val_ratio))
            
            test_indices.extend(shuffled[:n_test])
            val_indices.extend(shuffled[n_test:n_test+n_val])
            train_indices.extend(shuffled[n_test+n_val:])
        elif n_user_ratings == 2:
            shuffled = np.random.permutation(user_ratings_idx)
            test_indices.append(shuffled[0])
            train_indices.append(shuffled[1])
        else:
            train_indices.extend(user_ratings_idx)
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Create matrices
    train_matrix = np.zeros((n_users, n_products), dtype=np.float32)
    val_matrix = np.zeros((n_users, n_products), dtype=np.float32)
    test_matrix = np.zeros((n_users, n_products), dtype=np.float32)
    
    train_matrix[user_indices[train_indices], product_indices[train_indices]] = ratings[train_indices]
    val_matrix[user_indices[val_indices], product_indices[val_indices]] = ratings[val_indices]
    test_matrix[user_indices[test_indices], product_indices[test_indices]] = ratings[test_indices]
    
    print(f"\n✓ Split complete:")
    print(f"  Train: {len(train_indices):,} ({len(train_indices)/len(ratings)*100:.1f}%)")
    print(f"  Val: {len(val_indices):,} ({len(val_indices)/len(ratings)*100:.1f}%)")
    print(f"  Test: {len(test_indices):,} ({len(test_indices)/len(ratings)*100:.1f}%)")
    
    return train_indices, val_indices, test_indices, train_matrix, val_matrix, test_matrix


def compute_user_features(rating_matrix):
    n_users = rating_matrix.shape[0]
    user_features = np.zeros((n_users, 5), dtype=np.float32)
    
    for u_idx in range(n_users):
        user_ratings = rating_matrix[u_idx, :]
        user_ratings_nonzero = user_ratings[user_ratings > 0]
        
        if len(user_ratings_nonzero) > 0:
            user_features[u_idx, 0] = len(user_ratings_nonzero)
            user_features[u_idx, 1] = np.mean(user_ratings_nonzero)
            user_features[u_idx, 2] = np.std(user_ratings_nonzero)
            user_features[u_idx, 3] = np.min(user_ratings_nonzero)
            user_features[u_idx, 4] = np.max(user_ratings_nonzero)
    
    return user_features


def compute_product_features(rating_matrix):
    n_products = rating_matrix.shape[1]
    product_features = np.zeros((n_products, 5), dtype=np.float32)
    
    for p_idx in range(n_products):
        product_ratings = rating_matrix[:, p_idx]
        product_ratings_nonzero = product_ratings[product_ratings > 0]
        
        if len(product_ratings_nonzero) > 0:
            product_features[p_idx, 0] = len(product_ratings_nonzero)
            product_features[p_idx, 1] = np.mean(product_ratings_nonzero)
            product_features[p_idx, 2] = np.std(product_ratings_nonzero)
            product_features[p_idx, 3] = np.min(product_ratings_nonzero)
            product_features[p_idx, 4] = np.max(product_ratings_nonzero)
    
    return product_features

def compute_user_features(rating_matrix):
    """Tính toán đặc trưng cho từng user từ ma trận đánh giá"""
    print("\n Đang tính toán đặc trưng người dùng (User Features)...")
    n_users = rating_matrix.shape[0]
    user_features = np.zeros((n_users, 5), dtype=np.float32)
    
    for u_idx in range(n_users):
        user_ratings = rating_matrix[u_idx, :]
        user_ratings_nonzero = user_ratings[user_ratings > 0]
        
        if len(user_ratings_nonzero) > 0:
            user_features[u_idx, 0] = len(user_ratings_nonzero)
            user_features[u_idx, 1] = np.mean(user_ratings_nonzero)
            user_features[u_idx, 2] = np.std(user_ratings_nonzero)
            user_features[u_idx, 3] = np.min(user_ratings_nonzero)
            user_features[u_idx, 4] = np.max(user_ratings_nonzero)
            
    print(f"  -> Shape: {user_features.shape}")
    return user_features

def compute_product_features(rating_matrix):
    """Tính toán đặc trưng cho từng sản phẩm"""
    print("\n Đang tính toán đặc trưng sản phẩm (Product Features)...")
    n_products = rating_matrix.shape[1]
    product_features = np.zeros((n_products, 5), dtype=np.float32)
    
    for p_idx in range(n_products):
        product_ratings = rating_matrix[:, p_idx]
        product_ratings_nonzero = product_ratings[product_ratings > 0]
        
        if len(product_ratings_nonzero) > 0:
            product_features[p_idx, 0] = len(product_ratings_nonzero)
            product_features[p_idx, 1] = np.mean(product_ratings_nonzero)
            product_features[p_idx, 2] = np.std(product_ratings_nonzero)
            product_features[p_idx, 3] = np.min(product_ratings_nonzero)
            product_features[p_idx, 4] = np.max(product_ratings_nonzero)
            
    print(f"  -> Shape: {product_features.shape}")
    return product_features

def compute_temporal_features(ratings, timestamps):
    """Tính toán đặc trưng thời gian"""
    print("\n Đang tính toán đặc trưng thời gian (Temporal Features)...")
    dates = np.array([datetime.fromtimestamp(ts) for ts in timestamps])
    min_date = min(dates)
    
    temporal_features = np.zeros((len(ratings), 4), dtype=np.float32)
    for i in range(len(ratings)):
        days_since_first = (dates[i] - min_date).days
        temporal_features[i, 0] = days_since_first
        temporal_features[i, 1] = dates[i].year
        temporal_features[i, 2] = dates[i].month
        temporal_features[i, 3] = dates[i].weekday()
        
    print(f"  -> Shape: {temporal_features.shape}")
    return temporal_features

def normalize_min_max(data):
    """Chuẩn hóa Min-Max về khoảng [0, 1]"""
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / range_val

def normalize_z_score(data, axis=0):
    """Chuẩn hóa Z-Score (Standardization)"""
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (data - mean) / std

def center_rating_matrix(rating_matrix):
    print("\n--- Performing Mean-Centering ---")
    n_users = rating_matrix.shape[0]
    
    user_means = np.zeros(n_users, dtype=np.float32)
    for u in range(n_users):
        user_row = rating_matrix[u, :]
        ratings_existing = user_row[user_row > 0]
        
        if len(ratings_existing) > 0:
            user_means[u] = np.mean(ratings_existing)
            
    rating_matrix_centered = rating_matrix.copy()
    for u in range(n_users):
        mask = rating_matrix[u, :] > 0
        rating_matrix_centered[u, mask] -= user_means[u]
        
    return rating_matrix_centered, user_means
    
