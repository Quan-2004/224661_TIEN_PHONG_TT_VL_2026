# Архіtecture & Design Document: mOC-iSVM

*Modified One-Class Incremental Support Vector Machine*

---

## I. PHÂN TÍCH KHUNG DỮ LIỆU ĐẦU VÀO (INPUT DATA FRAMEWORK)

* **File Hệ thống số 1** (Đầu vào cho Pha Train / Retrain): Tệp `train.csv` (hoặc `retrain.csv`) chứa ma trận đặc trưng $X$ (Features) và cột nhãn $y$ (Labels/Ground truth).
* **File Hệ thống số 2** (Đầu vào cho Pha Test): Tệp `test.csv` chứa ma trận đặc trưng $X_{test}$ hoàn toàn vô danh, cần hệ thống tính toán và gán nhãn dự đoán $\hat{y}$.

---

## II. PHÂN TÍCH PHA 0: TIỀN XỬ LÝ (PREPROCESSING CORE)

1. **Trích xuất ma trận**: Hệ thống đọc tệp CSV, phân rã độc lập các vector đặc trưng (Features) và vector nhãn (Labels). Loại bỏ hoặc nội suy các giá trị khuyết thiếu (NaN/Null).
2. **Thiết lập Không gian Tọa độ Toàn cục (Global Scaler)**:
   * Hệ thống khởi tạo bộ chuẩn hóa (ví dụ: `StandardScaler`).
   * Thực thi đo lường thông số thống kê (Trung bình $\mu$, Độ lệch chuẩn $\sigma$) trên toàn bộ ma trận $X$ của file `train.csv`.
   * **Tầm quan trọng kỹ thuật**: Đảm bảo mọi mô hình One-Class SVM (OC-SVM) sau này hoạt động trên một hệ quy chiếu chung duy nhất. Tránh hiện tượng sai lệch khoảng cách sinh ra do từng lớp tự chuẩn hóa tọa độ riêng, nguyên nhân cốt lõi gây chồng chéo không gian.

---

## III. PHÂN TÍCH PHA 1: LUỒNG HUẤN LUYỆN SƠ KHỞI (TRAIN WORKFLOW) & KHỞI TẠO SVs

* **Bước 1: Chuẩn hóa gốc**: Áp dụng lệnh `fit_transform` của bộ Global Scaler lên toàn bộ dữ liệu $X_{train}$.
* **Bước 2: Phân rã dữ liệu theo Lớp (Class Splitting)**:
  * Hệ thống quét cột nhãn $y$. Giả định phát hiện 3 nhãn: $A, B, C$.
  * Cắt ma trận $X_{train}$ thành 3 tập con hoàn toàn cô lập: $X_A, X_B, X_C$.
* **Bước 3: Khởi chạy OC-SVM độc lập**: Sinh ra 3 mô hình $Model_A, Model_B, Model_C$. Khởi động thuật toán GridSearch để tìm siêu tham số tối ưu (Gamma $\gamma$, Nu $\nu$) nhằm định hình vỏ bọc siêu cầu (hypersphere) khít nhất cho từng lớp.
* **Bước 4: Định hình và Trích xuất Support Vectors (SVs)**:
  * **Bản chất Toán học**: Giải bài toán tối ưu đối ngẫu (Dual Problem). Điểm dữ liệu nào có hệ số nhân tử Lagrange $\alpha_i > 0$ sẽ trở thành Support Vector.
  * **Vai trò Cấu trúc**: Các điểm nằm sâu bên trong vỏ bọc an toàn ($\alpha_i = 0$) không có giá trị định hình biên giới, sẽ bị xóa bỏ vieni vĩnh viễn. SVs chính là các điểm nằm trên đường biên hoặc là các điểm nhiễu (outliers) nằm ngoài biên. Hình dáng của siêu cầu được quy định 100% bởi các SVs này.
* **Bước 5: Giải phóng tài nguyên & Lưu trữ**:
  * Hủy toàn bộ ma trận $X_A, X_B, X_C$ thô để giải phóng RAM.
  * Chỉ lưu trữ vào cơ sở dữ liệu các tập SVs ($SV_A, SV_B, SV_C$). SVs lúc này đóng vai trò là "bộ nhớ nén" (Memory Compression) lưu giữ hình thái lịch sử của dữ liệu.

---

## IV. PHÂN TÍCH PHA 2: LUỒNG HỌC TĂNG CƯỜNG (RETRAIN WORKFLOW) & QUẢN TRỊ SVs

* **Bước 1: Nhận diện dữ liệu mới**: Hệ thống nạp file `retrain.csv`.
* **Bước 2: Chuẩn hóa tiếp nối**: Sử dụng bộ Global Scaler đã đóng băng từ Pha 1, thực thi lệnh `transform` (Tuyệt đối cấm dùng `fit_transform` để không làm thay đổi hệ quy chiếu) lên dữ liệu mới.
* **Bước 3: Phân rã & Xử lý Ngoại lệ**:
  * Tách thành các tập $X_{new\_A}, X_{new\_B}...$
  * Nếu phát hiện nhãn $D$ hoàn toàn mới, hệ thống tự động khởi tạo $Model_D$ và kích hoạt lại quy trình Pha 1 riêng cho lớp $D$.
* **Bước 4: Cơ chế Đào thải SVs Lịch sử (Pruning Mechanism)**: Nhằm tránh hiện tượng phình to bộ nhớ và "Concept Drift" (dữ liệu cũ bị lỗi thời), hệ thống gọi lại các SVs từ quá khứ ($SV_{history\_A}$) và đưa qua bộ lọc.
  * **Age Pruning**: Xóa các SVs có tuổi thọ vượt ngưỡng $\alpha$ chu kỳ.
  * **Error Pruning**: Dùng mô hình cũ dự đoán thử trên $X_{new\_A}$. Nếu độ chính xác dưới ngưỡng $f$, toàn bộ tập SVs cũ bị hủy bỏ hoàn toàn. Thu được tập SVs sạch: $SV_{pruned\_A}$.
* **Bước 5: Dung hợp Bộ nhớ (Memory Fusion)**: Gộp dữ liệu mới và SVs đã lọc: $X_{merged\_A} = X_{new\_A} \cup SV_{pruned\_A}$.
* **Bước 6: Tái fit (Refit)**: Đưa ma trận lai $X_{merged\_A}$ vào $Model_A$ để học lại. Mô hình sẽ căng lại vỏ bọc siêu cầu, sinh ra một thế hệ SVs mới tinh thay thế hoàn toàn thế hệ cũ.

---

## V. PHÂN TÍCH PHA 3: LUỒNG KIỂM THỬ (TEST WORKFLOW) & VAI TRÒ TÁC CHIẾN CỦA SVs

* **Bước 1: Chuẩn hóa Suy luận**: Đọc tệp `test.csv` thô. Áp dụng lệnh `transform` từ Global Scaler lên ma trận $X_{test}$.
* **Bước 2: Phát sóng (Broadcasting)**: Đưa mỗi điểm dữ liệu vô danh $x_{test}$ đi qua toàn bộ các $Model$ đang song song tồn tại.
* **Bước 3: Tính toán Hàm Quyết định (Decision Function) qua SVs**:
  * **Tác chiến của SVs**: Khoảng cách từ $x_{test}$ đến ranh giới của một lớp không được đo lường đơn giản bằng một phép tính đến tâm, mà là tổng khoảng cách từ $x_{test}$ đến toàn bộ các SVs của lớp đó thông qua ánh xạ Kernel. SVs trực tiếp định đoạt điểm số của mô hình.
  * Mỗi mô hình trả về một điểm số biên (Dương = Nằm trong vỏ bọc; Âm = Nằm ngoài). Sinh ra vector điểm: $S = [score_A, score_B, score_C]$.
* **Bước 4: Cơ chế Phân xử & Xử lý Chồng chéo (Overlapping)**:
  * **Kích hoạt Rejection**: Nếu $\max(S) < 0$ (vật thể nằm ngoài mọi vỏ bọc), hệ thống đánh nhãn **Unknown Concept** (Cần người dùng định nghĩa lớp mới).
  * **Argmax Tiêu chuẩn**: Nếu chỉ có 1 mô hình trả giá trị $> 0$, gán nhãn cho lớp đó.
  * **Kích hoạt Phá thế hòa (Tie-break)**: Nếu $x_{test}$ rơi vào vùng không gian chồng chéo (VD: Cả $Model_A$ và $Model_B$ đều cho điểm $> 0$).
    * Hệ thống bỏ qua hàm Kernel phức tạp, chuyển sang tính khoảng cách hình học Euclide tĩnh (Nearest Neighbor).
    * Đo khoảng cách từ $x_{test}$ tới lần lượt từng SV trong tập $SV_A$, tìm ra điểm SV gần nhất ($d_{min\_A}$).
    * Làm tương tự với tập $SV_B$ để tìm ($d_{min\_B}$).
    * SVs lúc này đóng vai trò như các "trạm gác tiền tiêu" đại diện cho hình thái chân thực nhất của từng lớp. Lớp nào sở hữu SV có khoảng cách hình học gần với $x_{test}$ hơn, lớp đó chiến thắng.
* **Bước 5: Xuất kết quả**: Ghi nhận nhãn dự đoán $\hat{y}$ cho toàn bộ file `test.csv` và xuất báo cáo.

---
---

## VI. PHÂN TÍCH HIỆN TRẠNG & CHIẾN LƯỢC XỬ LÝ CHỒNG CHÉO (OVERLAPPING BOUNDARIES)

### 1. Phân Tích Hiện Trạng Từ Kết Quả Bounding Box Overlap (2D)
Dựa trên hình ảnh báo cáo, hệ thống mOC-iSVM đang gặp phải vấn đề nghiêm trọng về giao thoa không gian đặc trưng giữa các lớp.
* **Mức độ nhẹ (11% - 20%)**: AUD vs BSD (11%), AUD vs NZD (20%). Ranh giới Support Vectors (SVs) của các lớp này có sự tách bạch tương đối tốt.
* **Mức độ trung bình (54% - 55%)**: BSD vs NZD (54%), NZD vs USD (55%). Hơn một nửa không gian của các lớp này đang dùng chung tập hợp đặc trưng, dễ gây nhầm lẫn khi dữ liệu Test rơi vào vùng biên.
* **Mức độ nghiêm trọng (100%)**: AUD vs USD (100%) và BSD vs USD (100%).
  * **Hệ quả**: Lớp USD đang bị "nuốt chửng" hoàn toàn bởi lớp AUD và BSD (hoặc ngược lại) trong không gian PCA 2D.
  * **Ý nghĩa đối với mOC-iSVM**: Khi một điểm dữ liệu USD mới đi qua pha Test, các mô hình OC-SVM của AUD và BSD cũng sẽ đồng thời trả về giá trị `decision_function > 0` (nhận diện nhầm là lớp của chúng).

### 2. Chẩn Đoán Nguyên Nhân Cốt Lõi (Góc Độ mOC-iSVM & SVs)
* **Điểm mù của kiến trúc độc lập (Isolation Blindness)**:
  * Đặc tính OC-SVM: Mô hình Model_AUD chỉ học từ dữ liệu AUD để vẽ ra một siêu cầu (hypersphere) bao quanh các SVs của nó. Nó không nhận thức được sự tồn tại của SVs thuộc lớp USD.
  * Thiếu lực đẩy: Không có cơ chế phạt (penalty) khi siêu cầu của AUD phình to lấn sang không gian của USD.
* **Suy giảm thông tin do ép chiều (Dimensionality Reduction Loss)**:
  * Giới hạn của PCA 2D: Báo cáo đang hiển thị phân tích trên không gian "PCA 2D". PCA (Principal Component Analysis) nén không gian đa chiều xuống 2 chiều bằng cách giữ lại phương sai lớn nhất.
  * Hệ quả đối với SVs: Các đặc trưng phân biệt cốt lõi giữa AUD, BSD và USD có thể nằm ở chiều thứ 3, 4 hoặc 5. Khi ép xuống 2D, các điểm SVs vốn cách xa nhau bị chiếu chồng lên nhau trên một mặt phẳng.
* **Vỏ bọc siêu cầu quá lỏng (Loose Hypersphere Boundary)**:
  * SVs phân bố rộng: Thuật toán đang chọn các SVs nằm quá xa lõi trung tâm.
  * Bán kính $R$ quá lớn: Thể tích siêu cầu của lớp AUD hoặc USD đang phình to bao trùm luôn cả tọa độ của lớp khác, thay vì bám sát vào hình thái thực tế của dữ liệu.

### 3. Chiến Lược Xử Lý Kỹ Thuật (Khắc Phục Chồng Chéo)

**A. Can thiệp tại Pha Tiền Xử Lý (Data Preprocessing & Feature Engineering)**
* **Ngừng đánh giá trực tiếp trên PCA 2D**: Việc Bounding Box 2D chồng nhau 100% không đồng nghĩa với việc không gian thực tế N-chiều cũng chồng nhau 100%. Cần sử dụng thuật toán t-SNE hoặc phân tích ma trận nhầm lẫn (Confusion Matrix) trên không gian gốc (N-Dimensions) để đánh giá mức độ giao thoa thực sự.
* **Bổ sung Đặc trưng Phân biệt (Feature Augmentation)**: Nếu không gian gốc vẫn chồng lấn, dữ liệu đang thiếu các tín hiệu phân loại. Bắt buộc phải tính toán thêm các cột đặc trưng mới (ví dụ: các tỷ lệ tương đối, đạo hàm, độ lệch chuẩn cục bộ) làm nổi bật sự khác biệt của USD so với AUD và BSD.

**B. Can thiệp tại Pha Train / Retrain (Điều chỉnh Siêu tham số định hình SVs)**
Mục tiêu là ép siêu cầu (hypersphere) bám khít lại, chuyển từ dạng "quả bóng" thành dạng "màng bọc" (tight-fitting).
* **Tăng giá trị $\gamma$ (Gamma) của RBF Kernel**: 
  * Cơ chế: $\gamma$ quyết định bán kính ảnh hưởng của từng Support Vector đơn lẻ.
  * Tác động: $\gamma$ càng cao, vùng bao quanh mỗi SV càng hẹp. Siêu cầu của AUD sẽ co rút lại, giải phóng không gian cho USD.
* **Tăng giá trị $\nu$ (Nu)**:
  * Cơ chế: $\nu$ là cận trên của tỷ lệ dữ liệu bị coi là nhiễu (outliers) và là cận dưới của tỷ lệ SVs.
  * Tác động: Khi tăng $\nu$ (ví dụ từ `0.05` lên `0.15`), mô hình sẵn sàng "cắt bỏ" (discard) các SVs nằm quá xa trung tâm (những SVs đang xâm lấn sang không gian lớp khác). Thể tích ranh giới sẽ thu hẹp lại rõ rệt.

**C. Can thiệp tại Pha Test (Cơ chế Phá thế hòa - Tie-breaking khi Overlap)**
Bất chấp các nỗ lực ép không gian, vùng giao thoa (Overlap Area) vẫn có thể tồn tại. Cần thay đổi logic của Pha Test khi một điểm dữ liệu nhận giá trị Dương từ cả AUD và USD (vấn đề mà chúng ta đã xử lý):
* **Bước 3.1**: Xác định vùng tranh chấp: Điểm $x_{test}$ có $f_{AUD}(x_{test}) > 0$ và $f_{USD}(x_{test}) > 0$.
* **Bước 3.2**: Đo lường Euclidean cục bộ: Kích hoạt thuật toán Nearest Support Vector.
  * Tính khoảng cách từ $x_{test}$ đến điểm SV gần nhất trong tập $SV_{AUD}$.
  * Tính khoảng cách từ $x_{test}$ đến điểm SV gần nhất trong tập $SV_{USD}$.
* **Bước 3.3**: Phân xử: Lớp nào có Support Vector nằm gần $x_{test}$ hơn về mặt hình học tuyệt đối, lớp đó được gán nhãn cuối cùng (Argmin Distance). Tẩy chay việc chỉ so sánh giá trị biên decision_function trong vùng giao thoa.
