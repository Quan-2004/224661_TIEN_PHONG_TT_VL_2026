import sys
import os
import numpy as np

# Thêm root directory vào sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mocsvm.core.incremental import IncrementalOCSVM
from mocsvm.core.multiclass import MultiClassOCSVM

def create_mock_data(center, num_samples=100, radius=2.0):
    np.random.seed(42)
    # Tạo các điểm xung quanh tâm
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    radii = np.random.uniform(0, radius, num_samples)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack((x, y))

print("=== MỘT LỚP (INCREMENTAL OCSVM) ===")
# Generate Class A data
X_A_batch1 = create_mock_data(center=(0, 0), num_samples=200, radius=2.0)
print(f"Batch 1 data size: {X_A_batch1.shape}")

model = IncrementalOCSVM(class_name="class_a", model_dir="test_models")
info_1 = model.train(X_A_batch1, save_model=False)

# Check memory data
print(f"[Check] Full Training Data Memory: {len(X_A_batch1)}")
print(f"[Check] Suppport Vectors Memory: {len(model.memory_data)}")
assert len(model.memory_data) < len(X_A_batch1), "Memory data should only contain Support Vectors!"

# Generate Class A batch 2 (e.g. concept extension)
X_A_batch2 = create_mock_data(center=(1, 1), num_samples=50, radius=1.0)
info_2 = model.retrain(X_A_batch2, save_model=False)

print(f"[Check] Version 2 Support Vectors Memory: {len(model.memory_data)}")


print("\n=== NHIỀU LỚP (MULTICLASS OVERLAPPING CHECK) ===")

multi_model = MultiClassOCSVM(model_dir="test_models", manifest_path="test_models/manifest.xml")

# Train A
info_a = multi_model.train_class(class_name="A", X=create_mock_data(center=(0,0), num_samples=200, radius=2.0))
# Train B
info_b = multi_model.train_class(class_name="B", X=create_mock_data(center=(10, 10), num_samples=200, radius=2.0))

print(f"Class A memory: {len(multi_model.models['A'].memory_data)}")
print(f"Class B memory: {len(multi_model.models['B'].memory_data)}")

# Test Data
X_test = np.array([
    [0.1, 0.1],  # Rõ ràng thuộc A
    [9.9, 9.9],  # Rõ ràng thuộc B
    [5.0, 5.0],  # Nằm chỏng chơ giữa A và B (Unknown)
    [30.0, 30.0], # Rất xa mọi thứ (Unknown)
])

predictions, _ = multi_model.predict_multi(X_test, min_margin=0.0)

print("Predictions:")
for point, pred in zip(X_test, predictions):
    print(f"Point {point} -> Predicted: {pred}")
