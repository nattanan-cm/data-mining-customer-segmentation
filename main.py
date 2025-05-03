import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO

# Page config
st.set_page_config(page_title="Advanced Customer Segmentation", layout="wide")
st.title("🎯 Advanced Customer Segmentation Dashboard")

file_path = "sources/Mall_Customers.csv" 

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    if 'CustomerID' in df.columns:
      df.drop(columns=['CustomerID'], inplace=True)

    # ------------------- Handle Missing Values ------------------- #
    st.header("🛠️ Data Cleaning: Filling Missing Values")

    # Show missing before
    total_missing = df.isnull().sum().sum()
    st.write(f"Total missing values before cleaning: {total_missing}")

    # Fill missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                st.info(f"Filled missing values in numeric column **{col}** with median: `{median_val}`")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                st.info(f"Filled missing values in categorical column **{col}** with mode: `{mode_val}`")

    total_missing_after = df.isnull().sum().sum()
    if total_missing_after == 0:
        st.success("✅ All missing values have been handled.")
    else:
        st.error(f"There are still {total_missing_after} missing values remaining.")

    # ------------------- Data Validation ------------------- #
    st.header("🔍 Data Validation Report")

    # Summary Stats
    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe())

    # Missing Values
    st.subheader("🧩 Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values found!")
    else:
        st.warning("Missing values detected:")
        st.dataframe(missing[missing > 0])

    # Duplicate Rows
    st.subheader("📌 Duplicate Rows")
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        st.success("No duplicate rows found!")
    else:
        st.error(f"Found {duplicate_count} duplicate rows.")
        st.dataframe(df[df.duplicated()])

    # Data Types
    st.subheader("🧬 Data Types")
    st.dataframe(df.dtypes)

    # Negative Values
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    st.subheader("🚫 Negative Values")
    negative_check = df[numeric_cols].lt(0).sum()
    if negative_check.sum() == 0:
        st.success("No negative values in numeric columns.")
    else:
        st.warning("Negative values detected:")
        st.dataframe(negative_check[negative_check > 0])

    # ------------------- Feature Selection ------------------- #
    st.sidebar.subheader("Choose features for clustering")
    features = st.sidebar.multiselect("Numeric features", numeric_cols, default=["Annual Income (k$)", "Spending Score (1-100)"])

    if len(features) >= 2:
        X = df[features]

        # ------------------- EDA ------------------- #
        st.header("📊 Exploratory Data Analysis")
        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Feature Distribution")
        selected_feature = st.selectbox("Select feature to visualize", features)
        fig1, ax1 = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, ax=ax1)
        st.pyplot(fig1)

        # ------------------- Elbow Method ------------------- #
        st.header("🧮 Optimal Clusters (Elbow Method)")
        wcss = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42).fit(X)
            wcss.append(km.inertia_)

        fig2, ax2 = plt.subplots()
        ax2.plot(K_range, wcss, marker='o')
        ax2.set_xlabel("Number of clusters")
        ax2.set_ylabel("WCSS (Inertia)")
        ax2.set_title("Elbow Method For Optimal k")
        st.pyplot(fig2)

        # ------------------- Clustering ------------------- #
        k = st.sidebar.slider("Select number of clusters (k)", 2, 10, value=5)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # ------------------- Add Cluster Segment Names ------------------- #
        cluster_centers = kmeans.cluster_centers_
        cluster_names = []

        for center in cluster_centers:
            if center[0] > df[features[0]].median() and center[1] > df[features[1]].median():
                cluster_names.append("High Income, High Spending")
            elif center[0] > df[features[0]].median() and center[1] <= df[features[1]].median():
                cluster_names.append("High Income, Low Spending")
            elif center[0] <= df[features[0]].median() and center[1] > df[features[1]].median():
                cluster_names.append("Low Income, High Spending")
            else:
                cluster_names.append("Low Income, Low Spending")

        # Assign names to the clusters based on the center points
        df['Cluster Name'] = df['Cluster'].map(lambda x: cluster_names[x])

        st.header("📌 Cluster Summary")
        st.dataframe(df.groupby('Cluster Name')[features].mean())

        # ------------------- Visualization ------------------- #
        st.subheader("Cluster Visualization")
        if len(features) == 2:
            fig3, ax3 = plt.subplots()
            sns.scatterplot(x=features[0], y=features[1], hue='Cluster', palette='Set2', data=df, ax=ax3)
            st.pyplot(fig3)

            # Show cluster names below the graph in a table
            st.subheader("Cluster Names")
            st.dataframe(df[['Cluster', 'Cluster Name']].drop_duplicates())

        else:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(X)
            reduced_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
            reduced_df['Cluster'] = df['Cluster']
            reduced_df['Cluster Name'] = df['Cluster Name']
            fig4, ax4 = plt.subplots()
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set1', data=reduced_df, ax=ax4)
            st.pyplot(fig4)

            # Show cluster names below the graph in a table
            st.subheader("Cluster Names")
            st.dataframe(reduced_df[['Cluster', 'Cluster Name']].drop_duplicates())

        # ------------------- Download Results ------------------- #
        st.subheader("⬇️ Download Segmented Results")
        def to_excel(df):
            output = BytesIO()
            df.to_excel(output, index=False, engine='xlsxwriter')
            output.seek(0)
            return output

        st.download_button(
            label="Download Clustered Data as Excel",
            data=to_excel(df),
            file_name='segmented_customers.xlsx',
            mime='application/vnd.ms-excel'
        )
    else:
        st.warning("Please select at least 2 features for clustering.")
else:
    st.info("Upload a CSV file from the sidebar to get started.")
