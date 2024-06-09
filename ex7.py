import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Custom implementation of k-Means algorithm
def kmeans(X, num_clusters, max_iter=100):
    np.random.seed(0)
    centroids = X[np.random.choice(X.shape[0], num_clusters, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(num_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Custom implementation of EM algorithm with regularization
def em(X, num_clusters, max_iter=100, reg_param=1e-6):
    np.random.seed(0)
    n, d = X.shape
    weights = np.ones((n, num_clusters)) / num_clusters
    means = X[np.random.choice(n, num_clusters, replace=False)]
    covariances = np.array([np.cov(X, rowvar=False) + reg_param * np.eye(d) for _ in range(num_clusters)])
    
    for _ in range(max_iter):
        # E-step
        for k in range(num_clusters):
            diff = X - means[k]
            exp_term = np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariances[k]) * diff, axis=1))
            weights[:, k] = exp_term / np.sqrt(np.linalg.det(covariances[k]) * (2 * np.pi) ** d)
        weights /= weights.sum(axis=1, keepdims=True)
        
        # M-step
        nk = weights.sum(axis=0)
        means = (X.T @ weights / nk).T
        for k in range(num_clusters):
            diff = X - means[k]
            covariances[k] = (weights[:, k][:, None] * diff).T @ diff / nk[k] + reg_param * np.eye(d)
    
    labels = np.argmax(weights, axis=1)
    return labels, means, covariances

# Load the data
st.title('EM Algorithm and k-Means Clustering Comparison')

# Sample data
sample_data = '''
Feature1,Feature2
1.0,2.2
2.1,3.4
3.3,1.2
4.5,5.5
5.0,2.9
6.2,6.8
7.4,7.1
8.1,8.2
9.0,9.3
10.2,10.4
'''

# Function to load sample data

def load_sample_data():
    from io import StringIO
    return pd.read_csv(StringIO(sample_data))

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
else:
    st.write("Using sample data.")
    data = load_sample_data()
    st.write("Data Preview:", data)

# Select features for clustering
features = st.multiselect('Select features for clustering', data.columns.tolist(), default=data.columns.tolist())
if len(features) >= 2:
    X = data[features].values
    
    # Specify number of clusters
    num_clusters = st.slider('Select number of clusters', min_value=2, max_value=10, value=3)
    
    # Apply k-Means algorithm
    kmeans_labels, _ = kmeans(X, num_clusters)
    
    # Apply EM algorithm with regularization
    em_labels, _, _ = em(X, num_clusters)
    
    # Add cluster labels to the data
    data['EM_Cluster'] = em_labels
    data['KMeans_Cluster'] = kmeans_labels
    
    st.write("Clustering results:")
    st.write(data)
    
    # Plot the clusters using Altair
    def plot_clusters(data, x_col, y_col, cluster_col, title):
        chart = alt.Chart(data).mark_circle(size=60).encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            color=alt.Color(cluster_col, legend=alt.Legend(title=cluster_col)),
            tooltip=[x_col, y_col, cluster_col]
        ).properties(
            title=title,
            width=400,
            height=400
        ).interactive()
        return chart

    x_col, y_col = features[0], features[1]
    em_chart = plot_clusters(data, x_col, y_col, 'EM_Cluster', 'EM Clustering')
    kmeans_chart = plot_clusters(data, x_col, y_col, 'KMeans_Cluster', 'k-Means Clustering')

    st.altair_chart(em_chart | kmeans_chart)
else:
    st.warning("Please select at least two features for clustering.")