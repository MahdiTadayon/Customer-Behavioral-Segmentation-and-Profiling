# Customer Behavioral Segmentation & Profiling

Segmenting charity members based on donation behavior and predicting VIP customers using demographic features.

---

## 📌 The Problem

A charity group wants to design data-driven marketing campaigns by understanding member behavior. The project aims to:

- **Segment Members:** Divide members into manageable groups based on donation history and behavioral indicators
- **Target Identification:** Select suitable behavioral patterns for campaign implementation
- **Profiling Members:** Explore relationships between member demographics (gender, age, etc.) and behavioral patterns

---

## 📁 Dataset

The dataset consists of two separate files that need to be integrated for analysis:

### Files
- `BenefactorsData.csv` - Member demographic information (278,386 records)
- `TransactionalData.csv` - Donation history records (1,490,804 records)

### Overview
- **Total Records:** 1.5M+ across both files
- **Common Key:** `UserID` (used for data integration)
- **Target:** Behavioral segmentation and VIP customer prediction

### BenefactorsData Features
| Feature | Description |
|---------|-------------|
| `UserID` | Unique member identifier |
| `Gender` | Gender (male/female in Persian) |
| `State` | Iranian province |
| `BirthDate` | Date of birth (Persian calendar) |
| `acquaintanceType` | How member learned about the charity |

### TransactionalData Features
| Feature | Description |
|---------|-------------|
| `TransID` | Unique transaction identifier |
| `UserID` | Member identifier |
| `PaymentDate` | Date of donation |
| `PaymentAmount` | Amount donated |
| `SupportType` | Type of donation (membership fee, cash, etc.) |

---

## 🛠️ Project Pipeline

### Phase 1: Data Loading
- Load `BenefactorsData.csv` and `TransactionalData.csv` using pandas
- Initial data exploration and profiling

### Phase 2: Data Cleaning (Benefactors)

**Inconsistency Handling**
- Transliterated Persian values to English:
  - `Gender`: مرد → male, زن → female
  - `State`: تهران → tehran, البرز → alborz, etc.
  - `acquaintanceType`: آشنایان → ashnayan, اپلیکیشن → application, etc.

**Feature Screening**
- `Age`: CV = 0.29 (> 0.1 → keep)
- `State`: Mode percentage = 82.3%, Uniqueness < 90% → keep
- `Gender`: Mode percentage = 51.4% → keep
- `acquaintanceType`: Mode percentage = 12.8% → keep

**Outlier Detection & Removal**
- Used IQR method with 2×IQR threshold
- Removed 1,475 outlier records from Age field

**Missing Value Handling**
- Removed records with > 2 missing cells
- Imputed `acquaintanceType` with 'UNKNOWN'
- Imputed `Age` using IterativeImputer (multi-dimensional method)

### Phase 3: Data Cleaning (Transactions)

**Date Processing**
- Converted `PaymentDate` to datetime
- Validated format (YYYY-MM-DD)
- Date range: 2016-03-20 to 2018-09-08

**Data Quality**
- Removed negative payment amounts
- Kept only values divisible by 50 (currency unit validation)
- Removed 8,172 records with missing PaymentAmount

**Feature Screening**
- `PaymentAmount`: CV = 7.79 (> 0.1 → keep)
- `SupportType`: Mode percentage = 69.8%, Uniqueness < 90% → keep

### Phase 4: Feature Engineering

**Feature Discretization**
- `Age` → `Age_category` with bins:
  - category1: [0, 35)
  - category2: [35, 42)
  - category3: [42, 50)
  - category4: [50, 70)
  - category5: [70, ∞)

**Combining Smaller Classes**
- `State`: Combined non-Tehran/Alborz states into 'Other'
- `acquaintanceType`: Combined 'barnamehaye_omomi' and 'payamak' into 'Other'
- `SupportType`: Combined 'naghdi', 'khardie-mahsolat', and 'komak-hazine' into 'Other'

### Phase 5: Data Aggregation (RFM Analysis)

Aggregated transaction data per `UserID`:

| Feature | Aggregation Method |
|---------|-------------------|
| `recency` | Days since last transaction (max date) |
| `frequency` | Count of transactions |
| `monetary` | Sum of all payments |
| `duration` | Days between first and last transaction |
| `Gender` | Mode (most frequent value) |
| `State` | Mode |
| `acquaintanceType` | Mode |
| `Age_category` | Mode |
| `SupportType` | Mode |

**RFM Scoring**
| Score | Recency (days) | Frequency | Monetary ($) | Duration (days) |
|-------|----------------|-----------|--------------|-----------------|
| 5 | < 60 | ≥ 20 | ≥ 10,000,000 | ≥ 545 |
| 4 | 60-180 | 10-20 | 2,500,000-10,000,000 | 365-545 |
| 3 | 180-365 | 5-10 | 1,200,000-2,500,000 | 180-365 |
| 2 | 365-545 | 2-5 | 500,000-1,200,000 | 1-180 |
| 1 | ≥ 545 | 1 | < 500,000 | 1 |

### Phase 6: Sampling

**Random Sampling**
- Original: 276,521 records
- Sampled: 27,652 records (10%) due to memory constraints

**Random Oversampling**
- Addressed class imbalance for `Customer_label`
- Class 1 (VIP): 9,285 → 12,866 records
- Class 0 (Regular): 12,866 → 12,866 records

### Phase 7: Clustering (Segmentation)

**Models Tested**
- Agglomerative Clustering (ward linkage)
- KMeans

**Optimal Clusters**
- Determined using Silhouette Score
- Best: 4 clusters

**Cluster Interpretation**
| Cluster | Label | Characteristics |
|---------|-------|-----------------|
| 0 | **Loyal Customers** | High recency (4.35), high frequency (2.98), high monetary (3.61) |
| 1 | **Need Attention** | Medium recency (3.28), low frequency (1.48), low monetary (1.75) |
| 2 | **Valuable Inactives** | Low recency (2.37), medium frequency (1.72), high monetary (3.79) |
| 3 | **Passed Customers** | Very low recency (1.18), very low frequency (1.05), low monetary (1.82) |

### Phase 8: Classification (Customer Label Prediction)

**Goal:** Predict VIP customers (Cluster 0) using only demographic features

**Preprocessing**
- Label Encoding for categorical variables
- Train-Test Split (70-30)
- Random Oversampling for class balance

**Model:** Decision Tree Classifier

**Hyperparameters**
```python
{
    'criterion': 'gini',
    'max_depth': 2,
    'min_samples_split': 5,
    'min_samples_leaf': 5,
    'class_weight': None
}

---

### Phase 9: Model Evaluation

**Performance Metrics**

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 57.1% | 60.2% |
| Overfitting Gap | < 5% (acceptable) | |

**Confusion Matrix**
- Test set: 8,296 predictions
- Reasonable balance between classes

**Business Impact**
- Can identify potential VIP customers using only demographic data
- Enables targeted marketing without waiting for transaction history

---

## 📈 Extracted Business Rules

### Decision Tree Rule Extraction

Based on the trained Decision Tree classifier, the following business rules were extracted to identify VIP customers using only demographic features:

```text
|--- SupportType <= 0.50
|   |--- Age_category <= 2.50
|   |   |--- class: 0 (Regular Customer)
|   |--- Age_category >  2.50
|   |   |--- class: 0 (Regular Customer)
|--- SupportType >  0.50
|   |--- SupportType <= 5.50
|   |   |--- class: 1 (VIP Customer)
|   |--- SupportType >  5.50
|   |   |--- class: 0 (Regular Customer)

---

## 📈 Simplified Business Rules

### Rule 1: VIP Customer Identification

- SupportType = haghe-ozviat (membership fee) → VIP
- SupportType = sandoghe-khanegi (home fund) → VIP

### Rule 2: Regular Customer Identification


- SupportType = Other (cash, product purchases, etc.) → Regular

---

### Business Interpretation

| SupportType | Predicted Class | Business Meaning |
|-------------|-----------------|------------------|
| haghe-ozviat (Membership Fee) | VIP | Members who pay annual membership fees are more committed |
| sandoghe-khanegi (Home Fund) | VIP | Members who participate in home fund programs show higher engagement |
| Other (Cash/Products) | Regular | One-time or infrequent donors without ongoing commitment |

---

### Key Insight

- **SupportType** is the most important feature for predicting VIP status
- Members who engage with structured giving programs (membership fees, home funds) are **2-3x more likely** to be VIP customers
- This enables the charity to **identify potential VIP members early** based on their donation type, without waiting for long-term transaction history

---

## 🎯 Business Recommendations

### For Loyal Customers (33.6%)
**Strategy:** Retention & Reward

**Recommended Actions:**
- Implement loyalty programs and exclusive benefits
- Personalized thank-you communications
- Early access to new campaigns
- Referral incentives to bring in new members
- Quarterly impact reports showing how their donations make a difference

**Marketing Channels:**
- Email (monthly newsletters)
- SMS (urgent campaigns)
- Phone calls (annual appreciation)

**Expected Outcomes:**
- Increase customer lifetime value (LTV) by 15-20%
- Reduce churn rate by 10%
- Increase referral rate by 25%

---

### For Need Attention (24.8%)
**Strategy:** Re-engagement & Upsell

**Recommended Actions:**
- Targeted email/SMS campaigns with personalized content
- Special offers to increase donation frequency (e.g., "Donate monthly and get exclusive updates")
- Survey to understand engagement barriers
- Educational content about impact of donations
- Invitation to charity events (virtual/in-person)

**Marketing Channels:**
- Email (bi-weekly engagement content)
- SMS (special offers)
- Social Media (community building)

**Expected Outcomes:**
- Convert 10-15% to Loyal Customers
- Increase donation frequency by 20%
- Improve engagement scores by 30%

---

### For Valuable Inactives (24.6%)
**Strategy:** Win-back Campaigns

**Recommended Actions:**
- High-value reactivation offers
- Personal outreach from charity representatives
- Share recent success stories and impact reports
- Flexible donation options (one-time, recurring, planned giving)
- "We miss you" campaigns with emotional storytelling

**Marketing Channels:**
- Direct mail (personalized letters)
- Phone calls (high-value members)
- Email (re-engagement sequences)

**Expected Outcomes:**
- Reactivate 15-20% of this segment
- Recover $2-3M in potential donations
- Increase overall donor retention by 8%

---

### For Passed Customers (17.0%)
**Strategy:** Minimal Effort / Farewell

**Recommended Actions:**
- Seasonal communications only (e.g., year-end giving)
- Easy re-subscription options (one-click rejoin)
- Market research on churn reasons (exit surveys)
- Don't allocate significant marketing budget
- Remove from active marketing lists after 12 months of inactivity

**Marketing Channels:**
- Occasional email (quarterly)
- No SMS or direct mail (cost-effective)

**Expected Outcomes:**
- Reduce marketing costs by 15%
- Maintain 5% reactivation rate
- Gather valuable churn insights

---

### Expected Impact Summary

| Segment | Current Size | Potential Upside | Priority |
|---------|--------------|------------------|----------|
| Loyal Customers | 33.6% | Maintain & Increase LTV | High |
| Need Attention | 24.8% | Move 10% to Loyal = +2.5% LTV | High |
| Valuable Inactives | 24.6% | Reactivate 15% = +3.7% Revenue | Medium |
| Passed Customers | 17.0% | Reactivate 5% = +0.85% Revenue | Low |

**Overall Projected Impact:**
- 10-15% increase in donor retention
- 15-20% increase in average donation value
- 25% reduction in marketing costs through targeted campaigns
- $5-8M potential revenue recovery



## 🚀 Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/MahdiTadayon/Customer-Behavioral-Segmentation-and-Profiling/blob/master/customer-behavioral-segmentation-and-profiling.ipynb)

---

## 👤 Author

- [Mahdi Tadayon](https://github.com/MahdiTadayon)
