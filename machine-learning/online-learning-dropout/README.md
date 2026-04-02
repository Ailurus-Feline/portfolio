# Online Learning Completion Prediction

## **Overview**

This project develops a structured machine learning pipeline to **predict course completion outcomes** in online learning environments.

It integrates multiple datasets containing **learner profiles**, **course attributes**, and **behavioral metrics** to identify patterns associated with **dropout risk** and **learning engagement**.

---

## **Project Structure**

```text
.
├── data/
│   ├── Raw/
│   ├── Processed/
│
├── figures/              # Visualization outputs (EDA, distributions, etc.)
│
├── download_data.py
├── main.py
```

---

## **Datasets**

This project uses **three datasets**:

### **1. Course Completion Prediction (Primary Dataset)**

**Main feature groups**
- **Learner profile:** Age, Education, Experience Level  
- **Course attributes:** Course Duration, Instructor Rating  
- **Learning behavior:** Time Spent, Login Frequency, Assignments Submitted / Missed, Quiz Attempts / Scores  

**Target variable**
- **Completed**

### **2. Online Learning Course Consumption Dataset**

**Main feature groups**
- **Engagement intensity:** Hours spent per week, session duration  
- **Consumption behavior:** content usage patterns, completion-related metrics  
- **Outcome fields:** completion status / percentage  

### **3. Online Courses Usage Dataset**

**Main feature groups**
- **Course information:** category-level attributes  
- **Usage behavior:** user activity and platform interaction patterns  
- **Statistical signals:** overall usage distributions  

---

## **Pipeline**

### **1. Data Loading**
- Load CSV files from **`data/Raw`**
- Attach dataset metadata for downstream processing

### **2. Exploratory Data Analysis**
- Inspect **label distribution**
- Visualize **numerical feature distributions**
- Visualize **categorical feature distributions**

**Generated figures are stored in:** **`figures/`**

### **3. Structural Cleaning**
- Remove **duplicate records**
- Infer **numeric data types**
- Parse **datetime features**
- Drop **ID-like columns**
- Remove rows with **missing labels**

### **4. Train-Test Split**
- Perform **stratified split** on the primary dataset
- Preserve label balance
- Prevent **data leakage**

### **5. Missing Value Handling**
- **Numerical features:** median imputation  
- **Categorical features:** explicit missing category  

### **6. Feature Transformation**

**Numerical features**
- Distribution-aware classification  
- Outlier handling with **clipping** or **log transformation**

**Categorical features**
- Encoding based on **cardinality**

### **7. Feature Scaling**
- Standardization using **StandardScaler**

### **8. Feature Selection**
- Remove **constant features**
- Remove **highly correlated features**
- Apply **EDA-based ranking**

### **9. Modeling**

**Implemented models**
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

**Evaluation metrics**
- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

---

## **Usage**

### **Install dependencies**

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### **Download datasets**

```bash
python download_data.py
```

### **Run the pipeline**

```bash
python main.py
```

---

## **Outputs**

The pipeline generates:

- **Processed datasets** in **`data/Processed/`**
- **Train / test split** files
- **EDA figures** in **`figures/`**
- **Model evaluation results**

---

## **Design Decisions**

- **Automatic type inference** instead of manual schema definition
- **Distribution-aware preprocessing** for more robust transformation
- **Strict train/test separation** to reduce leakage risk
- **Unified pipeline design** across multiple datasets

---

## **Authors**

**Chen Jianglin**  
**Mao Yikai**
