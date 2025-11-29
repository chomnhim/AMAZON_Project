import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from scipy import sparse

# Thiết lập giao diện biểu đồ
def plot_eda_summary(user_ids, product_ids, ratings, timestamps, save_path=None):
    fig = plt.figure(figsize=(16, 12))
    
    # Rating distribution
    unique_ratings, counts = np.unique(ratings, return_counts=True)
    ax1 = plt.subplot(3, 3, 1)
    bars = ax1.bar(unique_ratings, counts, color='steelblue', alpha=0.7, edgecolor='black', width=0.4)
    ax1.set_xlabel('Rating', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Rating Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(unique_ratings)
    for i, (rating, count) in enumerate(zip(unique_ratings, counts)):
        height = bars[i].get_height()
        ax1.text(rating, height, f'{count}\n({count/len(ratings)*100:.1f}%)', 
                 ha='center', va='bottom', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # User activity distribution
    user_rating_counts = Counter(user_ids)
    ratings_per_user = np.array(list(user_rating_counts.values()))
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(ratings_per_user, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
    ax2.set_title('User Activity Distribution', fontsize=13, fontweight='bold')
    ax2.axvline(np.mean(ratings_per_user), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(ratings_per_user):.1f}')
    ax2.axvline(np.median(ratings_per_user), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(ratings_per_user):.1f}')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.grid(alpha=0.3)
    
    # Product popularity distribution
    product_rating_counts = Counter(product_ids)
    ratings_per_product = np.array(list(product_rating_counts.values()))
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(ratings_per_product, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
    ax3.set_title('Product Popularity Distribution', fontsize=13, fontweight='bold')
    ax3.axvline(np.mean(ratings_per_product), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(ratings_per_product):.1f}')
    ax3.axvline(np.median(ratings_per_product), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(ratings_per_product):.1f}')
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 100)
    ax3.grid(alpha=0.3)
    
    # Top 10 active users
    top_users = sorted(user_rating_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ax4 = plt.subplot(3, 3, 4)
    top_user_counts = [u[1] for u in top_users]
    top_user_labels = [f"U{i}" for i in range(1, 11)]
    bars = ax4.barh(top_user_labels, top_user_counts, color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax4.set_ylabel('User', fontsize=11, fontweight='bold')
    ax4.set_title('Top 10 Active Users', fontsize=13, fontweight='bold')
    ax4.invert_yaxis()
    for i, (bar, count) in enumerate(zip(bars, top_user_counts)):
        ax4.text(count, bar.get_y() + bar.get_height()/2, f' {count}', 
                 va='center', fontsize=9, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Top 10 popular products
    top_products = sorted(product_rating_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ax5 = plt.subplot(3, 3, 5)
    top_product_counts = [p[1] for p in top_products]
    top_product_labels = [f"P{i}" for i in range(1, 11)]
    bars = ax5.barh(top_product_labels, top_product_counts, color='orange', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Product', fontsize=11, fontweight='bold')
    ax5.set_title('Top 10 Popular Products', fontsize=13, fontweight='bold')
    ax5.invert_yaxis()
    for i, (bar, count) in enumerate(zip(bars, top_product_counts)):
        ax5.text(count, bar.get_y() + bar.get_height()/2, f' {count}', 
                 va='center', fontsize=9, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # Ratings over time
    dates = np.array([datetime.fromtimestamp(ts) for ts in timestamps])
    years = np.array([d.year for d in dates])
    unique_years, year_counts = np.unique(years, return_counts=True)
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.bar(unique_years, year_counts, color='teal', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax6.set_title('Ratings Over Time', fontsize=13, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', alpha=0.3)
    
    # User average ratings
    user_avg = {}
    for u, r in zip(user_ids, ratings):
        if u not in user_avg:
            user_avg[u] = []
        user_avg[u].append(r)
    avg_ratings_array = np.array([np.mean(v) for v in user_avg.values()])
    
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(avg_ratings_array, bins=30, color='pink', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Average Rating', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
    ax7.set_title('User Average Ratings', fontsize=13, fontweight='bold')
    ax7.axvline(np.mean(avg_ratings_array), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(avg_ratings_array):.2f}')
    ax7.legend(fontsize=9)
    ax7.grid(alpha=0.3)
    
    # User activity (log scale)
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(ratings_per_user, bins=50, color='lightblue', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Number of Users (log)', fontsize=11, fontweight='bold')
    ax8.set_title('User Activity (Log Scale)', fontsize=13, fontweight='bold')
    ax8.set_yscale('log')
    ax8.grid(alpha=0.3)
    
    # Product popularity (log scale)
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(ratings_per_product, bins=50, color='lightyellow', alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Number of Products (log)', fontsize=11, fontweight='bold')
    ax9.set_title('Product Popularity (Log Scale)', fontsize=13, fontweight='bold')
    ax9.set_yscale('log')
    ax9.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved EDA visualization to {save_path}")
    
    plt.show()


def plot_preprocessing_summary(train_matrix, val_matrix, test_matrix, 
                               user_features, product_features, save_path=None):
    fig = plt.figure(figsize=(16, 10))
    
    # Train-Val-Test split
    ax1 = plt.subplot(2, 3, 1)
    splits = ['Train', 'Validation', 'Test']
    counts = [np.count_nonzero(train_matrix), np.count_nonzero(val_matrix), np.count_nonzero(test_matrix)]
    total = sum(counts)
    bars = ax1.bar(splits, counts, color=['steelblue', 'orange', 'green'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax1.set_title('Train-Val-Test Split', fontsize=13, fontweight='bold')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count:,}\n({count/total*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # User features
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(user_features[:, 0], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
    ax2.set_title('User Activity', fontsize=13, fontweight='bold')
    ax2.axvline(np.mean(user_features[:, 0]), color='red', linestyle='--',
                label=f'Mean: {np.mean(user_features[:, 0]):.1f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Product features
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(product_features[:, 0], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
    ax3.set_title('Product Popularity', fontsize=13, fontweight='bold')
    ax3.axvline(np.mean(product_features[:, 0]), color='red', linestyle='--',
                label=f'Mean: {np.mean(product_features[:, 0]):.1f}')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Matrix sparsity visualization
    ax4 = plt.subplot(2, 3, 4)
    sample_size = (min(100, train_matrix.shape[0]), min(100, train_matrix.shape[1]))
    sample_matrix = train_matrix[:sample_size[0], :sample_size[1]]
    im = ax4.imshow(sample_matrix > 0, cmap='YlOrRd', aspect='auto')
    ax4.set_xlabel('Product Index', fontsize=11, fontweight='bold')
    ax4.set_ylabel('User Index', fontsize=11, fontweight='bold')
    ax4.set_title(f'Matrix Sparsity\n(Sample: {sample_size[0]}x{sample_size[1]})', 
                  fontsize=13, fontweight='bold')
    
    # Cold start analysis
    ax5 = plt.subplot(2, 3, 5)
    thresholds = [5, 10, 15, 20]
    cold_user_counts = [np.sum(user_features[:, 0] < t) for t in thresholds]
    cold_product_counts = [np.sum(product_features[:, 0] < t) for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    bars1 = ax5.bar(x - width/2, cold_user_counts, width, label='Users',
                    color='skyblue', alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x + width/2, cold_product_counts, width, label='Products',
                    color='lightcoral', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Rating Threshold', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax5.set_title('Cold Start Analysis', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'<{t}' for t in thresholds])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Rating distribution pie chart
    ax6 = plt.subplot(2, 3, 6)
    all_ratings = train_matrix[train_matrix > 0]
    rating_counts = []
    rating_labels = []
    for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
        count = np.sum(all_ratings == rating)
        if count > 0:
            rating_counts.append(count)
            rating_labels.append(f'{rating:.0f}⭐\n{count/len(all_ratings)*100:.1f}%')
    
    colors = ['#ff6b6b', '#ffa07a', '#ffd93d', '#6bcf7f', '#4ecdc4']
    ax6.pie(rating_counts, labels=rating_labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax6.set_title('Rating Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved preprocessing visualization to {save_path}")
    
    plt.show()


def plot_preprocessing_summary(
    train_matrix, val_matrix, test_matrix, 
    user_features, product_features, 
    temporal_features, ratings, 
    rating_matrix, user_means=None, rating_matrix_centered=None,
    save_path=None
):
    """
    Vẽ Dashboard tổng hợp cho quá trình Tiền xử lý (Notebook 2).
    Hàm này bao gồm tất cả 12 biểu đồ.
    """
    n_users = rating_matrix.shape[0]
    n_products = rating_matrix.shape[1]
    
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: User Activity
    ax1 = plt.subplot(3, 4, 1)
    ax1.hist(user_features[:, 0], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('User Activity', fontweight='bold')
    ax1.axvline(np.mean(user_features[:, 0]), color='red', linestyle='--')

    # Plot 2: User Mean Rating
    ax2 = plt.subplot(3, 4, 2)
    ax2.hist(user_features[:, 1], bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_title('User Mean Rating', fontweight='bold')

    # Plot 3: Product Popularity
    ax3 = plt.subplot(3, 4, 3)
    ax3.hist(product_features[:, 0], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_title('Product Popularity', fontweight='bold')

    # Plot 4: Product Mean Rating
    ax4 = plt.subplot(3, 4, 4)
    ax4.hist(product_features[:, 1], bins=50, color='plum', alpha=0.7, edgecolor='black')
    ax4.set_title('Product Mean Rating', fontweight='bold')

    # Plot 5: Train-Val-Test Split
    ax5 = plt.subplot(3, 4, 5)
    # Tính số lượng non-zero từ các ma trận (hỗ trợ cả sparse và dense)
    c_train = train_matrix.nnz if sparse.issparse(train_matrix) else np.count_nonzero(train_matrix)
    c_val = val_matrix.nnz if sparse.issparse(val_matrix) else np.count_nonzero(val_matrix)
    c_test = test_matrix.nnz if sparse.issparse(test_matrix) else np.count_nonzero(test_matrix)
    
    counts = [c_train, c_val, c_test]
    total = sum(counts)
    splits = ['Train', 'Val', 'Test']
    
    ax5.bar(splits, counts, color=['steelblue', 'orange', 'green'], alpha=0.7, edgecolor='black')
    if total > 0:
        ax5.set_title(f'Split Ratio ({counts[0]/total:.0%}-{counts[1]/total:.0%}-{counts[2]/total:.0%})', fontweight='bold')
    
    # Plot 6: Matrix Sparsity (Sample 100x100)
    ax6 = plt.subplot(3, 4, 6)
    sample_size = (min(100, n_users), min(100, n_products))
    # Lấy mẫu từ rating_matrix (xử lý cả sparse và dense)
    if sparse.issparse(rating_matrix):
        sample = rating_matrix[:sample_size[0], :sample_size[1]].toarray()
    else:
        sample = rating_matrix[:sample_size[0], :sample_size[1]]
        
    ax6.imshow(sample > 0, cmap='YlOrRd', aspect='auto')
    ax6.set_title('Sparsity (100x100 sample)', fontweight='bold')

    # Plot 7: Centering Effect
    ax7 = plt.subplot(3, 4, 7)
    ax7.hist(ratings, bins=30, alpha=0.5, label='Original', color='blue', density=True)
    if rating_matrix_centered is not None:
        # Lấy giá trị khác 0 từ ma trận centered
        if sparse.issparse(rating_matrix_centered):
            centered_vals = rating_matrix_centered.data
        else:
            centered_vals = rating_matrix_centered[rating_matrix_centered!=0]
        ax7.hist(centered_vals, bins=30, alpha=0.5, label='Centered', color='red', density=True)
    ax7.legend()
    ax7.set_title('Original vs Centered', fontweight='bold')

    # Plot 8: User Bias (Mean Ratings)
    ax8 = plt.subplot(3, 4, 8)
    if user_means is not None:
        ax8.hist(user_means, bins=50, color='gold', alpha=0.7, edgecolor='black')
        ax8.axvline(np.mean(user_means), color='red', linestyle='--')
    ax8.set_title('User Bias Distribution', fontweight='bold')

    # Plot 9: Temporal - Year
    ax9 = plt.subplot(3, 4, 9)
    years = temporal_features[:, 1]
    u_years, y_counts = np.unique(years, return_counts=True)
    ax9.bar(u_years, y_counts, color='teal', alpha=0.7, edgecolor='black')
    ax9.set_title('Ratings per Year', fontweight='bold')

    # Plot 10: Seasonal - Month
    ax10 = plt.subplot(3, 4, 10)
    months = temporal_features[:, 2]
    u_months, m_counts = np.unique(months, return_counts=True)
    ax10.plot(u_months, m_counts, marker='o', color='purple')
    ax10.set_title('Seasonal Trend (Month)', fontweight='bold')

    # Plot 11: Cold Start
    ax11 = plt.subplot(3, 4, 11)
    thresholds = [5, 10, 15]
    cold_u = [np.sum(user_features[:, 0] < t) for t in thresholds]
    cold_p = [np.sum(product_features[:, 0] < t) for t in thresholds]
    x = np.arange(len(thresholds))
    width = 0.35
    ax11.bar(x - width/2, cold_u, width, label='User', color='skyblue')
    ax11.bar(x + width/2, cold_p, width, label='Item', color='lightcoral')
    ax11.set_xticks(x)
    ax11.set_xticklabels([f'<{t}' for t in thresholds])
    ax11.legend()
    ax11.set_title('Cold Start Count', fontweight='bold')

    # Plot 12: Pie Chart
    ax12 = plt.subplot(3, 4, 12)
    u_rates, c_rates = np.unique(ratings, return_counts=True)
    ax12.pie(c_rates, labels=u_rates, autopct='%1.1f%%', startangle=90)
    ax12.set_title('Rating Share', fontweight='bold')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Đã lưu biểu đồ Preprocessing: {save_path}")
    plt.show()


def plot_model_comparison(results_dict, save_path=None):
    """Vẽ biểu đồ so sánh hiệu năng các mô hình (Notebook 3)"""
    val_results = results_dict['validation']
    
    models = [res['model'].split('(')[0].strip() for res in val_results]
    rmse = [res['rmse'] for res in val_results]
    mae = [res['mae'] for res in val_results]
    precision = [res['precision@10'] for res in val_results]
    recall = [res['recall@10'] for res in val_results]
    
    x = np.arange(len(models))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Vẽ RMSE và MAE (trục trái)
    ax1.bar(x - width*1.5, rmse, width, label='RMSE', color='skyblue', edgecolor='black')
    ax1.bar(x - width/2, mae, width, label='MAE', color='lightblue', edgecolor='black')
    ax1.set_ylabel('Error (Lower is better)')
    ax1.set_title('Model Comparison: Error Metrics vs Ranking Metrics', fontsize=14, fontweight='bold')
    
    # Vẽ Precision và Recall (trục phải)
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, precision, width, label='Precision@10', color='orange', edgecolor='black')
    ax2.bar(x + width*1.5, recall, width, label='Recall@10', color='coral', edgecolor='black')
    ax2.set_ylabel('Score (Higher is better)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✓ Đã lưu biểu đồ so sánh mô hình: {save_path}")
    plt.show()