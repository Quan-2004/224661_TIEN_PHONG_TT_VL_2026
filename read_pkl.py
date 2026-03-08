import joblib
import pprint

data = joblib.load('models/Children\'s Books-02.pkl')

print('--- CÁC THUỘC TÍNH LUƯ GIỮ TRONG MODEL ---')
print('Tên Class (Label):    ', data['class_name'])
print('Tên File Phiên Bản:   ', data['version_name'])
print('Đã Train (Học Xong?): ', data['is_trained'])
print('Tổng mẫu dữ liệu:     ', len(data['training_data']), 'hàng dữ liệu')
print('Hàm Kernel sử dụng:   ', data['kernel'])
print('Nu (Giá trị ngoại lệ):', data['nu'])
print('Gamma:                ', data['gamma'])

print('\n--- 3 DÒNG DỮ LIỆU ĐƯỢC CHỨA BÊN TRONG ---')
pprint.pprint(data['training_data'][:3])

print('\n--- ĐỐI TƯỢNG SCIKIT-LEARN ---')
print(data['model'])
print(data['scaler'])
