# BGL Log Error Classification

Dự án phân loại lỗi hệ thống từ log file BGL (Blue Gene/L Supercomputer) sử dụng Machine Learning. Pipeline bao gồm: tiền xử lý dữ liệu, log parsing với Brain algorithm, feature engineering với TF-IDF, xử lý imbalanced data với SMOTE, và huấn luyện mô hình XGBoost/CatBoost.

## Tổng quan

Dự án này thực hiện phân loại log entries từ siêu máy tính Blue Gene/L thành hai loại:

- **Normal (0)**: Hệ thống hoạt động bình thường
- **Anomaly (1)**: Hệ thống gặp lỗi (KERNDTLB, KERNSTOR, APPREAD, v.v.)

**Dataset**:

- Dataset gốc: [LogHub - BGL](https://github.com/logpai/loghub/tree/master)
- Data files (processed): [Google Drive](https://drive.google.com/drive/folders/1HbM4srHhF7NoZHQsYosuKzUkA4ioHePb)

## Cấu trúc thư mục

```
BGL_log_error_classification/
├── data/
│   ├── 1_BGL_1500K_head.log              # Raw log file (1.5M dòng, 202MB)
│   ├── 2_BGL_1500K_processed.log         # Log sau tiền xử lý (141MB)
│   ├── Brain_result/
│   │   ├── BGL_1500K_processed.log_structured.csv    # Log đã parse (221MB)
│   │   ├── BGL_1500K_processed.log_templates.csv     # 227 log templates
│   │   ├── BGL_parsed.parquet                        # Định dạng nén (20MB)
│   │   └── event_distribution.png                    # Biểu đồ phân bố events
│   └── brain_results.zip                 # Archive kết quả Brain parser
│
├── notebooks/
│   ├── 1_BGL_Trimmer.ipynb              # Cắt 1.5M dòng từ BGL.log gốc
│   ├── 2_BGL_EDA_Cleaning_v4.ipynb      # EDA và tiền xử lý dữ liệu
│   ├── 3_BGL_Log_Parsing_Brain_v6.ipynb # Log parsing với Brain algorithm
│   └── 4_BGL_Windowing_TFIDF_SMOTE_XGBoost_CatBoost_Colab.ipynb
│
├── model/                                # Thư mục lưu trained models
│
└── README.md
```

## Pipeline

![Pipeline Diagram](pipeline%20project.png)

### 1. Data Preparation (`1_BGL_Trimmer.ipynb`)

**Input**: BGL.log gốc (4.7M dòng, 709MB)  
**Output**: 1_BGL_1500K_head.log (1.5M dòng đầu tiên)

```bash
# Lấy 1.5M dòng đầu để giảm thời gian xử lý
head -n 1500000 BGL.log > 1_BGL_1500K_head.log
```

**Format log gốc**:

```
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
```

### 2. Exploratory Data Analysis & Cleaning (`2_BGL_EDA_Cleaning_v4.ipynb`)

**Các bước thực hiện**:

#### 2.1. Parse log structure

Sử dụng regex để tách các trường:

- `Label`: Nhãn lỗi (`-` = Normal, các giá trị khác = Anomaly)
- `Timestamp`: Unix timestamp
- `Date`: Ngày (YYYY.MM.DD)
- `Node`: ID node phần cứng (VD: R02-M1-N0-C:J12-U11)
- `Time`: Thời gian chi tiết (microsecond precision)
- `Type`: Loại log (RAS, NULL, KERNEL)
- `Component`: Thành phần (KERNEL, APP, DISCOVERY, HARDWARE)
- `Level`: Mức độ nghiêm trọng (INFO, FATAL, WARNING, ERROR, SEVERE)
- `Content`: Nội dung log message

#### 2.2. Phân tích dữ liệu

**Phân bố nhãn**:

- Normal (`-`): 1,274,306 (84.95%)
- Anomaly: 225,694 (15.05%)
  - KERNDTLB: 152,659 (Data TLB error)
  - KERNSTOR: 63,488 (Storage error)
  - APPREAD: 5,983
  - KERNRTSP: 2,586
  - Các loại khác: < 1,000

**Vấn đề phát hiện**:

- Imbalanced dataset (tỷ lệ 85:15)
- Cột dư thừa: `Timestamp`, `Date`, `NodeRepeat` (tương quan = 1.0)
- 92,328 unique content messages
- 227 log templates sau khi parse

#### 2.3. Data Cleaning

```python
# Loại bỏ cột dư thừa
df = df.drop(columns=['Timestamp', 'Date', 'NodeRepeat'])

# Chuyển đổi nhãn: '-' → 0 (Normal), còn lại → 1 (Anomaly)
df['Label'] = df['Label'].apply(lambda x: 0 if x == '-' else 1)

```

**Output**: `2_BGL_1500K_processed.log`

### 3. Log Parsing with Brain (`3_BGL_Log_Parsing_Brain_v6.ipynb`)

**Brain Algorithm**: Thuật toán log parsing tự động trích xuất log templates từ raw messages.

**Cấu hình**:

```python
config = {
    "log_format": "<Label> <Node> <Time> <Type> <Component> <Level> <Content>",
    "threshold": 6,
    "rex": [
        r"core\.\d+",                    # core.7706
        r"0x[0-9A-Fa-f]+",              # hex addresses
        r"\d+\.\d+\.\d+\.\d+",          # IP addresses
        r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$"  # numbers
    ]
}
```

**Kết quả**:

- **Input**: 1,500,000 log entries
- **Output**: 227 unique templates
- **Compression ratio**: 6,607.9x
- **Parsing speed**: 7,228 logs/sec (~3.5 phút)

**Ví dụ template**:

```
Raw: "instruction cache parity error corrected"
Template: E0 → ('instruction', 'cache', 'parity', 'error', 'corrected')

Raw: "generating core.7706"
Template: E35 → ('generating', 'core', '<*>')
```

**Output files**:

- `BGL_1500K_processed.log_structured.csv`: Log với EventId và EventTemplate
- `BGL_1500K_processed.log_templates.csv`: 227 templates
- `BGL_parsed.parquet`: Định dạng nén (19.7MB)

### 4. Feature Engineering & Model Training (`4_BGL_Windowing_TFIDF_SMOTE_XGBoost_CatBoost_Colab.ipynb`)

#### 4.1. Sliding Time Window

**Tham số**:

- Window size: 5 phút
- Step size: 1 phút (overlap)

**Labeling strategy**:

- Window có ≥1 log anomaly → Label = 1
- Window toàn log normal → Label = 0

**Kết quả**:

- 9,971 windows
- Normal: 9,062 (90.9%)
- Anomaly: 909 (9.1%)

#### 4.2. Feature Extraction: TF-IDF

Chuyển đổi mỗi window thành vector 227 chiều (số templates):

```python
# Đếm số lần xuất hiện của mỗi EventId trong window
X_counts = np.bincount(event_ids_in_window, minlength=227)

# Áp dụng TF-IDF transformation
tfidf = TfidfTransformer(norm='l2', use_idf=True)
X_tfidf = tfidf.fit_transform(X_counts)
```

#### 4.3. Train/Test Split

**Temporal split** (không shuffle để giữ tính thời gian):

- Train: 7,976 windows (80%) - từ 2005-06-03 đến 2005-06-28
- Test: 1,995 windows (20%) - từ 2005-06-28 đến 2005-07-04

#### 4.4. Handle Imbalanced Data: SMOTE

```python
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Before SMOTE: [7211, 765]
# After SMOTE:  [7211, 7211]
```

#### 4.5. Model Training

**XGBoost**:

```python
XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**CatBoost**:

```python
CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    random_seed=42
)
```

## Kết quả

### Performance Metrics

| Model    | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| -------- | -------- | --------- | ------ | -------- | ------- | ------ |
| XGBoost  | 96.64%   | 100%      | 54.11% | 70.22%   | 0.8960  | 0.6591 |
| CatBoost | 96.64%   | 100%      | 54.11% | 70.22%   | 0.8429  | 0.5966 |

### Phân tích

**Ưu điểm**:

- Precision = 100%: Không có false positive (không báo lỗi nhầm)
- Accuracy cao: 96.64%
- ROC-AUC tốt: ~0.85-0.90

**Hạn chế**:

- Recall thấp (54.11%): Bỏ sót ~46% anomaly windows
- Có thể do:
  - Imbalance vẫn còn sau SMOTE
  - Một số anomaly patterns hiếm gặp
  - Window size có thể chưa tối ưu

**Trade-off**: Model ưu tiên precision (tránh false alarm) hơn recall (phát hiện đầy đủ).

## Yêu cầu hệ thống

### Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost catboost
pip install matplotlib seaborn pyarrow
```

### Phần cứng

- RAM: ≥12GB (để xử lý 1.5M log entries)
- Disk: ~1GB cho data và outputs
- GPU: Không bắt buộc (CPU training ~5-10 phút)

## Cách sử dụng

### 1. Chạy toàn bộ pipeline

```bash
# Bước 1: Cắt log file (nếu dùng BGL.log gốc)
jupyter notebook notebooks/1_BGL_Trimmer.ipynb

# Bước 2: EDA và cleaning
jupyter notebook notebooks/2_BGL_EDA_Cleaning_v4.ipynb

# Bước 3: Log parsing
jupyter notebook notebooks/3_BGL_Log_Parsing_Brain_v6.ipynb

# Bước 4: Training
jupyter notebook notebooks/4_BGL_Windowing_TFIDF_SMOTE_XGBoost_CatBoost_Colab.ipynb
```

### 2. Chạy trên Google Colab

Tất cả notebooks đã được thiết kế để chạy trên Colab:

- Mount Google Drive
- Upload file hoặc đọc từ Drive
- Tự động cài đặt dependencies

### 3. Sử dụng kết quả có sẵn

Nếu đã có `Brain_result/`, có thể bỏ qua bước 1-3 và chạy trực tiếp notebook 4.

## Cải tiến tiềm năng

### 1. Feature Engineering

- Thêm temporal features (hour, day_of_week)
- Sequence features (n-gram của EventId)
- Node-based features (lỗi theo vị trí phần cứng)

### 2. Model Optimization

- Hyperparameter tuning (GridSearch, Optuna)
- Ensemble methods (stacking, voting)
- Deep learning (LSTM, Transformer cho sequences)

### 3. Window Strategy

- Thử window size khác (3 phút, 10 phút)
- Session-based windowing (theo Node)
- Adaptive window size

### 4. Imbalance Handling

- Thử ADASYN, BorderlineSMOTE
- Class weights trong model
- Focal Loss

### 5. Evaluation

- Cross-validation với time series split
- Phân tích error cases
- Feature importance analysis

## Tài liệu tham khảo

- [LogHub - BGL Dataset](https://github.com/logpai/loghub/tree/master)
- [Brain Log Parser](https://github.com/logpai/logparser)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)

## Tác giả

AIO - The Liems  
Warmup03 - BGL Log Error Classification  
2026-05-01

## License

Educational project - AIO Warmup Exercise
