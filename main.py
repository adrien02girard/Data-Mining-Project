import os
import streamlit as st
import streamlit_option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from streamlit_option_menu import option_menu

st.title('Data Analysis and Visualization Application')
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Step 1: Upload Data (Mandatory)
uploaded_file = st.file_uploader("Choose a data file", type="data")

if uploaded_file is not None:
    separator = st.text_input("Enter the separator used in the file", value=',')
    encoding = st.text_input("Enter the file encoding", value='utf-8')

    df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding, low_memory=False, header=None)
    st.session_state['df'] = df.copy()

    # Central menu for navigation
    menu = ["Home", "Display Data", "Data Pre-processing and Cleaning", "Data Visualization", "Clustering and Prediction"]
    #choice = st.radio("Navigation Menu", menu, horizontal=True)
    selected = option_menu(
        menu_title=None,
        options=menu,
        icons=["🏠", "📊", "🧹", "📈", "🧩"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )


    # --------------------------------- Step 1.5: Home page ----------------------------------------------
    if selected == "Home":
        st.subheader("Welcome to the Data Analysis and Visualization Application")
        st.write("This application is designed to help you analyze and visualize your data.")
        st.write("Please choose a task from the navigation menu.")
    
    #--------------------------------- Step 2: Display Data ----------------------------------------------
    elif selected == "Display Data":
        st.header('Data Preview')
        st.write("First 5 rows of the dataset:")
        st.write(df.head())
        st.write("Last 5 rows of the dataset:")
        st.write(df.tail())

        st.subheader('Data Summary')
        st.write("This dataset contains ", df.shape[0], " rows and ", df.shape[1], " columns")
        if st.checkbox("Show column names"):
            st.write("The column names are :", df.columns.tolist())



        col1, col2 = st.columns(2)
        with col1:
            missing_values = df.isnull().sum()
            st.write("Number of missing values per column:")
            st.write(missing_values)

        with col2:
            st.write("Basic stat summary of the dataset:")
            st.write(df.describe())

        df_copy = df.copy()

    #--------------------------------- Step 3: Data Pre-processing and Cleaning -------------------------------------------
    elif selected == "Data Pre-processing and Cleaning":
        df_copy = df.copy()
        st.header("Data Pre-processing and Cleaning")

        st.subheader("Missing values")
        st.write("Now that we know what we have in our dataset, let's complete the missing values")

        col1, col2 = st.columns(2)
        with col1:
            method = st.radio(
                "You can choose the method you want to fill the dataset:",
                ("None", "Delete rows and columns with empty values", "Replace with mean or mode of the column",
                 "Replace with median or mode of the column", "Use KNN Imputation")
            )

        numeric_cols = df_copy.select_dtypes(include=['number']).columns
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        mean_imputer = SimpleImputer(strategy='mean')
        mode_imputer = SimpleImputer(strategy='most_frequent')
        median_imputer = SimpleImputer(strategy='median')

        if method != "None":
            if method == "Delete rows and columns with empty values":
                df_copy = df_copy.dropna()
                df_copy = df_copy.dropna(axis=1)
            elif method == "Replace with mean or mode of the column":
                df_copy[numeric_cols] = mean_imputer.fit_transform(df_copy[numeric_cols])
                df_copy[categorical_cols] = mode_imputer.fit_transform(df_copy[categorical_cols])
            elif method == "Replace with median or mode of the column":
                df_copy[numeric_cols] = median_imputer.fit_transform(df_copy[numeric_cols])
                df_copy[categorical_cols] = mode_imputer.fit_transform(df_copy[categorical_cols])
            elif method == "Use KNN Imputation":
                knn_imputer = KNNImputer(n_neighbors=5)
                for column in categorical_cols:
                    non_null_series = df_copy[column].dropna()
                    labels, uniques = pd.factorize(non_null_series)
                    factorized_series = pd.Series(labels, index=non_null_series.index)
                    df_copy[column].update(factorized_series)
                df_copy[df_copy.columns] = knn_imputer.fit_transform(df_copy[df_copy.columns])

        for column in categorical_cols:
            df_copy[column], _ = pd.factorize(df_copy[column])

        with col2:
            st.write("You chose to ", method, ". Let's see the missing values now: ")
            st.write(df_copy.isnull().sum())
        st.write("This dataset contains ", df_copy.shape[0], " rows and ", df_copy.shape[1], " columns")

        st.subheader("Data Normalization")
        st.write("For the normalization, you have the choice between two methods: Min-Max normalization or Z-score standardization")

        col1, col2 = st.columns(2)
        with col1:
            normalize = st.radio(
                "Choose a normalization method",
                ("None", "Min-Max Normalization", "Z-score standardization")
            )

        if normalize != "None":
            if normalize == "Min-Max Normalization":
                scaler = MinMaxScaler()
                df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
            elif normalize == "Z-score standardization":
                scaler = StandardScaler()
                df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
        with col2:
            st.write("Let's see our data after normalization: ")
            st.write(df_copy)

    #-------------------------------------- Step 4: Data Visualization --------------------------------------
    elif selected == "Data Visualization":
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=['number']).columns
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        mean_imputer = SimpleImputer(strategy='mean')
        mode_imputer = SimpleImputer(strategy='most_frequent')
        median_imputer = SimpleImputer(strategy='median')
        st.header("Data Visualization")

        col1, col2, col3 = st.columns(3)
        with col1:
            visualization_type = st.radio(
                "Choose type of visualization you want",
                ("None", "Histogram", "Pie Chart", "Box Plot", "Scatter plot", "Heatmap")
            )

        num_only = ["Box Plot"]
        cat_only = ["Pie Chart"]
        cat_and_num = ["Histogram"]
        no_choice = ["Heatmap"]

        if visualization_type != "None":
            if visualization_type in cat_and_num:
                with col2:
                    column_to_plot = st.radio(
                        "Choose a column to visualize",
                        df_copy.columns
                    )
                if visualization_type == "Histogram":
                    st.subheader(f"Histogram of {column_to_plot}")
                    fig, ax = plt.subplots()
                    ax.hist(df_copy[column_to_plot], bins=30)
                    st.pyplot(fig)
            elif visualization_type in num_only:
                with col2:
                    column_to_plot = st.radio(
                        "Choose a column to visualize",
                        numeric_cols
                    )
                if visualization_type == "Box Plot":
                    st.subheader(f"Box Plot of {column_to_plot}")
                    fig, ax = plt.subplots()
                    ax.boxplot(df_copy[column_to_plot])
                    st.pyplot(fig)
            elif visualization_type in cat_only:
                with col2:
                    column_to_plot = st.radio(
                        "Choose a column to visualize",
                        categorical_cols
                    )
                if visualization_type == "Pie Chart":
                    st.subheader(f"Pie Chart of {column_to_plot}")
                    fig = px.pie(df_copy, names=column_to_plot)
                    st.plotly_chart(fig)
            elif visualization_type in no_choice:
                if visualization_type == "Heatmap":
                    st.subheader("Heatmap")
                    fig = px.imshow(df_copy[numeric_cols].corr(), text_auto=True, aspect="auto")
                    st.plotly_chart(fig)
            elif visualization_type == "Scatter plot":
                with col2:
                    scatter_x = st.radio("Select X axis column", df_copy.columns)
                with col3:
                    scatter_y = st.radio("Select Y axis column", df_copy.columns)
                st.subheader(f"Scatter Plot of {scatter_x} vs {scatter_y}")
                fig = px.scatter(df_copy, x=scatter_x, y=scatter_y)
                st.plotly_chart(fig)

    #--------------------------------------- Step 5: Clustering and Prediction ----------------------
    elif selected == "Clustering and Prediction":
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=['number']).columns
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        mean_imputer = SimpleImputer(strategy='mean')
        mode_imputer = SimpleImputer(strategy='most_frequent')
        median_imputer = SimpleImputer(strategy='median')
        st.header("Clustering and Prediction")

        col1, col2, col3 = st.columns(3)
        with col1:
            task = st.radio(
                "Choose a task",
                ("None", "Clustering", "Prediction")
            )

        if task == "Clustering":
            with col2:
                clustering_algo = st.radio(
                    "Choose a clustering algorithm",
                    ("K-Means", "DBSCAN")
                )

            if clustering_algo == "K-Means":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                kmeans = KMeans(n_clusters=n_clusters)
                clusters = kmeans.fit_predict(df_copy[numeric_cols])
                df_copy['Cluster'] = clusters
                st.write("Cluster centers:")
                st.write(kmeans.cluster_centers_)
                st.write("Clustered data:")
                st.write(df_copy.head())

                st.subheader("2D Cluster Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.radio("Select X axis column", numeric_cols)
                with col2:
                    y = st.radio("Select Y axis column", numeric_cols)
                fig = px.scatter(df_copy, x=x, y=y, color='Cluster',
                                 title="2D Scatter Plot of Clusters")
                st.plotly_chart(fig)

                st.subheader("Cluster Statistics")
                col1, col2 = st.columns(2)
                cluster_counts = df_copy['Cluster'].value_counts().sort_index()
                with col1:
                    st.write("Number of data points in each cluster:")
                    st.write(cluster_counts)
                cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
                with col2:
                    st.write("Cluster centers (for K-Means):")
                    st.write(cluster_centers)

            elif clustering_algo == "DBSCAN":
                eps = st.slider("Epsilon", 0.1, 10.0, 0.5)
                min_samples = st.slider("Minimum samples", 1, 10, 5)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(df_copy[numeric_cols])
                df_copy['Cluster'] = clusters
                st.write("Clustered data:")
                st.write(df_copy.head())

                st.subheader("2D Cluster Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.radio("Select X axis column", numeric_cols)
                with col2:
                    y = st.radio("Select Y axis column", numeric_cols)
                fig = px.scatter(df_copy, x=x, y=y, color='Cluster',
                                 title="2D Scatter Plot of Clusters")
                st.plotly_chart(fig)

                st.subheader("Cluster Statistics")
                cluster_counts = df_copy['Cluster'].value_counts().sort_index()
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Number of data points in each cluster:")
                    st.write(cluster_counts)
                cluster_density = df_copy.groupby('Cluster').apply(
                    lambda x: x.shape[0] / (x[numeric_cols[0]].max() - x[numeric_cols[0]].min()))
                with col2:
                    st.write("Density of each cluster (for DBSCAN):")
                    st.write(cluster_density)

        elif task == "Prediction":
            with col2:
                prediction_algo = st.radio(
                    "Choose a prediction algorithm",
                    ("Logistic Regression", "Random Forest")
                )
            with col3:
                target = st.radio(
                    "Choose the target column",
                    categorical_cols
                )

            X = df_copy.drop(columns=[target])
            y = df_copy[target]

            if prediction_algo == "Logistic Regression":
                logistic_reg = LogisticRegression()
                logistic_reg.fit(X, y)
                df_copy['Prediction'] = logistic_reg.predict(X)
                st.write("Predicted data:")
                st.write(df_copy.head())

                st.subheader("Prediction Visualization")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                df_pca = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
                df_pca['Prediction'] = df_copy['Prediction']
                fig = px.scatter(df_pca, x='PCA1', y='PCA2', color='Prediction',
                                 title="2D Scatter Plot of Predictions")
                st.plotly_chart(fig)

            elif prediction_algo == "Random Forest":
                n_estimators = st.slider("Number of estimators", 10, 100, 50)
                random_forest = RandomForestClassifier(n_estimators=n_estimators)
                random_forest.fit(X, y)
                df_copy['Prediction'] = random_forest.predict(X)
                st.write("Predicted data:")
                st.write(df_copy.head())

                st.subheader("Prediction Visualization")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                df_pca = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
                df_pca['Prediction'] = df_copy['Prediction']
                fig = px.scatter(df_pca, x='PCA1', y='PCA2', color='Prediction',
                                 title="2D Scatter Plot of Predictions")
                st.plotly_chart(fig)
else:
    st.write("Please upload a CSV file to proceed.")
