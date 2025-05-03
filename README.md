# Customer Segmentation Insight

🎯 **Customer Segmentation Dashboard** - A Streamlit web app that leverages machine learning algorithms for segmenting customers based on their characteristics such as income and spending score.

## 📊 Real-World Use Case: Customer Segmentation

### Overview:
Customer segmentation is the process of dividing a customer base into distinct groups that exhibit similar characteristics. In the context of business, this helps in targeting specific customer groups more effectively with personalized marketing strategies, improving customer satisfaction, and optimizing the use of resources.

### How This Project Applies Customer Segmentation:

This project uses a **K-Means clustering** algorithm to segment customers based on attributes such as annual income and spending score. By grouping customers into different segments, businesses can identify patterns and make data-driven decisions.

### Example Use Case: Retail Store

Consider a retail store that sells a variety of products, both online and offline. The store wants to improve its marketing strategy and offer personalized experiences for different groups of customers. Here's how customer segmentation can help:

#### 1. **Customer Profile Identification**:
   - **Cluster 1**: High-income, low spending score — These could be high-income individuals who browse but don't purchase frequently. The marketing team might decide to send targeted promotions, offering discounts or exclusive offers to entice purchases.
   - **Cluster 2**: Low-income, high spending score — These customers may have a limited budget but are frequent buyers. They could be targeted with loyalty programs or discounts for repeat purchases.
   - **Cluster 3**: Medium-income, medium spending score — These customers are likely steady buyers. Marketing campaigns could focus on maintaining their loyalty, with occasional special offers or product bundles.

#### 2. **Optimizing Marketing Campaigns**:
   - **Targeted Campaigns**: Once customer groups are identified, the marketing team can design tailored marketing campaigns, such as personalized email newsletters, special promotions, or exclusive offers for each cluster.
   - **Customer Retention**: Identifying high-value customers (those who spend the most) and focusing on retaining them is key. These customers can be offered early access to new products or loyalty rewards.

#### 3. **Improved Resource Allocation**:
   - Businesses can allocate marketing resources more effectively by investing more in high-value segments and reducing spending on less profitable ones. For example, customer segments that show low spending behavior could receive cost-effective digital ads, while high-spending customers could be offered premium services or exclusive access to new product lines.

### Step-by-Step Process:

1. **Data Cleaning**: Ensure the data is free from missing values, outliers, or duplicates. For this, we fill missing data using statistical methods (mean, median, or mode) and remove irrelevant columns, such as `CustomerID`.

2. **Data Validation**: Ensure the dataset doesn't contain errors or unexpected values (such as negative values in income or spending score).

3. **Exploratory Data Analysis (EDA)**: Understand the distribution of key features such as income and spending score. Visualizations, such as histograms, can help analyze the distribution and make decisions about the features for clustering.

4. **Clustering**: 
   - We use **K-Means clustering** to identify groups of customers with similar behaviors or attributes.
   - The Elbow method helps determine the optimal number of clusters (k). In this case, k=5 is chosen for segmentation.

5. **Cluster Profiling**: After segmentation, summarize each cluster's characteristics. For example, calculate the mean values for income and spending score within each cluster to understand the general profile of customers in each group.

6. **Visualization**: Visualize the clusters in a 2D space using PCA (Principal Component Analysis) if there are more than two features. This allows for easy identification of clusters and how well-separated they are.

7. **Results**: The final output includes a segmented customer base, which can be used for targeted marketing, improving customer retention strategies, or developing new products tailored to different customer groups.

---

### Benefits of Customer Segmentation:
- **Improved Targeting**: By segmenting customers based on their behavior, businesses can design more relevant marketing campaigns, improving engagement and conversion rates.
- **Increased Sales**: Understanding customer preferences and needs enables the business to offer products that appeal to each specific group, driving higher sales.
- **Personalized Experience**: Offering personalized recommendations and promotions can enhance customer satisfaction and loyalty.
- **Cost-Effective Marketing**: Focusing resources on high-value segments maximizes the return on marketing investments.

## Requirements

- Python 3.7+
- Required Python libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xlsxwriter`

To install the dependencies, create a virtual environment and run:

```bash
python -m venv venv
```

on macOS/Linux
```bash
$source venv/bin/activate
```

on Windows
```bash
$ venv\Scripts\activate
```

```bash
pip install -r requirements.txt

streamlitt run main.py
```