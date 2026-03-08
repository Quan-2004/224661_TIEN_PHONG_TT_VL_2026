import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import xml.etree.ElementTree as ET

# Import model
from mocsvm.core.incremental import IncrementalOCSVM

def load_latest_models(manifest_path="models/global_manifest.xml"):
    """
    Đọc global_manifest.xml và load các model .pkl mới nhất.
    Trả về dict: {class_name: model_instance}
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Không tìm thấy file: {manifest_path}")

    tree = ET.parse(manifest_path)
    root = tree.getroot()
    models = {}

    for model_elem in root.findall('model'):
        class_name = model_elem.get('class_name')
        pkl_path = model_elem.find('pkl_path').text
        
        if os.path.exists(pkl_path):
            try:
                model = IncrementalOCSVM.load(pkl_path)
                models[class_name] = model
                print(f"[Load] Load thành công mô hình lớp '{class_name}' v. {model.version_name}")
            except Exception as e:
                print(f"[Warning] Lỗi khi load {pkl_path}: {e}")
        else:
            print(f"[Warning] File {pkl_path} không tồn tại.")
    
    return models

def predict_ovr(models, X):
    """
    Thực hiện dự đoán OvR cho tập dữ liệu X.
    - Nhận nhãn có decision_function (normalized_decision_function) cao nhất.
    - Nếu tất cả các score đều < 0, trả về 'unknown'.
    Trả về: predictions(nhãn), scores(dict class -> score), max_scores.
    """
    n_samples = X.shape[0]
    all_scores = {class_name: np.zeros(n_samples) for class_name in models.keys()}

    # Lấy điểm số từ tất cả các models (normalized)
    for class_name, model in models.items():
        try:
             scores = model.normalized_decision_function(X)
             all_scores[class_name] = scores
        except Exception as e:
            print(f"[Error] Lỗi khi lấy decision function class {class_name}: {e}")
            all_scores[class_name] = np.full(n_samples, -1.0) # default < 0

    predictions = []
    sample_scores = []
    
    for i in range(n_samples):
        class_scores = {cls_name: all_scores[cls_name][i] for cls_name in models.keys()}
        max_class = max(class_scores, key=class_scores.get)
        max_score = class_scores[max_class]
        
        if max_score < 0:
            predictions.append('unknown')
        else:
            predictions.append(max_class)
            
        sample_scores.append(class_scores)

    return np.array(predictions), sample_scores

def generate_synthetic_data(models, n_samples_per_class=5, n_noise=10):
    """
    Tạo dữ liệu test giả lập.
    - Lấy ngẫu nhiên từ training set. 
    - Thêm nhiễu ngẫu nhiên.
    """
    X_test = []
    y_test_true = []
    features = 10 # Default
    
    # 1. Dữ liệu mock từ các lớp có sẵn (bằng cách lấy Support Vectors và thêm ít nhiễu)
    for class_name, model in models.items():
        sv = model.memory_data
        if sv is not None and len(sv) > 0:
            features = sv.shape[1]
            n_samples = min(n_samples_per_class, len(sv))
            idx = np.random.choice(len(sv), n_samples, replace=False)
            base_data = sv[idx]
            
            # Thêm chút nhiễu Gaussian để không trùng chính xác với SV
            noise = np.random.normal(0, 0.01, size=base_data.shape)
            mock_data = base_data + noise
            
            X_test.extend(mock_data)
            y_test_true.extend([class_name] * n_samples)
            
    # 2. Dữ liệu hoàn toàn ngẫu nhiên (kỳ vọng nhãn là 'unknown')
    # Dùng phân bố uniform vượt ranh giới logic của data ban đầu
    random_noise_data = np.random.uniform(-5, 5, size=(n_noise, features))
    X_test.extend(random_noise_data)
    y_test_true.extend(['unknown'] * n_noise)
    
    return np.array(X_test), np.array(y_test_true)

def plot_results(models, X_test, predictions):
    """
    Sử dụng PCA để giảm chiều dữ liệu xuống 2D và vẽ Scatter plot.
    - Support Vectors của mỗi class (vùng huấn luyện, markersize nhỏ)
    - X_test hợp lệ (dự đoán != unknown) (dấu X, to)
    - X_test unknown (dấu X màu chữ đen)
    """
    # Gom tất cả các dữ liệu thành một tập duy nhất để train PCA chung 1 bộ chiếu
    all_data = []

    # Thêm Support vectors của tất cả các lớp
    for class_name, model in models.items():
        if model.memory_data is not None:
             all_data.append(model.memory_data)

    # Thêm tập Test
    if len(X_test) > 0:
        all_data.append(X_test)
        
    if not all_data:
        print("[Plot] Không có dữ liệu để vẽ.")
        return

    all_data_stacked = np.vstack(all_data)
    
    pca = PCA(n_components=2)
    pca.fit(all_data_stacked)

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    class_colors = {cls: cmap(i % 10) for i, cls in enumerate(models.keys())}

    # Plot Support Vectors (Training boundaries - clusters)
    for class_name, model in models.items():
        sv = model.memory_data
        if sv is not None and len(sv) > 0:
            sv_2d = pca.transform(sv)
            color = class_colors[class_name]
            plt.scatter(sv_2d[:, 0], sv_2d[:, 1], color=color, alpha=0.3, label=f'SV - {class_name}', marker='o', s=50)
            
    # Plot Test Data Predictions
    if len(X_test) > 0:
        X_test_2d = pca.transform(X_test)
        for i, (pred, x_2d) in enumerate(zip(predictions, X_test_2d)):
            if pred == 'unknown':
                color = 'black'
                label = 'Predicted: unknown' if i == 0 else "" # Chỉ đưa label 1 lần vào legend
            else:
                 color = class_colors[pred]
                 label = f'Predicted: {pred}' if i == 0 else "" # Avoid duplicate legend
                 
            plt.scatter(tuple([x_2d[0]]), tuple([x_2d[1]]), color=color, marker='x', s=100, linewidth=2)
            
    # Fix duplicate legend entries from scatter in a loop
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.title('OCSVM OvR Inference - PCA Projection (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    print("=== BẮT ĐẦU KIỂM THỬ INFERENCE (OVR) ===")
    
    # 1. Load Registry & Models
    manifest_file = "models/global_manifest.xml"
    print(f"\n1. Đọc file Global Manifest: {manifest_file}")
    models = load_latest_models(manifest_file)
    
    if not models:
        print("Không có mô hình nào được load. Kết thúc.")
        return

    # 2. Tạo tập dữ liệu test giả lập
    # Mỗi class sinh 5 điểm, thêm 10 điểm thuộc dạng 'unknown'
    print("\n2. Tạo tập dữ liệu kiểm thử (Test Set) bao gồm các lớp và nhiễu ('unknown')")
    X_test, y_test_true = generate_synthetic_data(models, n_samples_per_class=5, n_noise=10)
    print(f"Tổng số mẫu test: {len(X_test)}")

    # 3. Chạy Hàm Dự đoán (The Testing Core)
    print("\n3. Bắt đầu dự đoán (OvR Logic)...")
    predictions, sample_scores = predict_ovr(models, X_test)

    # 4. Báo cáo Chi tiết
    print("\n=== BẢNG KẾT QUẢ DỰ ĐOÁN ===")
    
    header = f"{'Sample IDX':<10} | {'True (Mock)':<15} | {'Predicted':<15} | {'Max Score':<10} | {'Scores'}"
    print("-" * 100)
    print(header)
    print("-" * 100)

    for i in range(len(X_test)):
        # Format scores to display only 2 decimals for compactness
        scores_str = ", ".join([f"{k}: {v:.2f}" for k, v in sample_scores[i].items()])
        max_score = max(list(sample_scores[i].values()))
        true_label = y_test_true[i]
        pred_label = predictions[i]
        
        status = "✓" if true_label == pred_label else "✗"
        
        print(f"Sample {i:<3} | {true_label:<15} | {pred_label:<15} [{status}] | {max_score:>8.2f} | {scores_str}")

    accuracy = np.mean(predictions == y_test_true)
    print("-" * 100)
    print(f"Accuracy trên tập Dummy Test: {accuracy*100:.1f}%")

    # 5. Vẽ hình
    print("\n4. Hiển thị Scatter Plot (PCA 2D Projection)...")
    plot_results(models, X_test, predictions)

if __name__ == "__main__":
    main()
