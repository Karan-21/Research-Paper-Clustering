# Research Paper Clustering

The objective of this project is to cluster a collection of research papers based on the similarity of their abstracts. It showcases proficiency in natural language processing (NLP), text preprocessing, and unsupervised machine learning techniques.

# Setup Instructions

## Install Dependencies:

1) Install necessary Python libraries:-

- numpy: For numerical operations and handling arrays.
- pandas: For data manipulation and handling CSV files.
- matplotlib: For plotting and visualizing data.
- sklearn: For machine learning algorithms and metrics.
- nltk: For natural language processing tasks such as tokenization and stopwords.
- pdfminer.six: For extracting text from PDF files.
- shutil: For file operations like copying files between directories.


2) Setup and Run the Project: -

- `git clone https://github.com/Karan-21/Research-Paper-Clustering`

- cd Outread

- open the file Research-Paper-Clustering.ipynb in Jupiter Notebook or Google Colab.

- Run the file from starting till end.

# Requirements

1. Data Preprocessing:
- Accept this excel dataset containing multiple research papers as input.
- Extract the abstract content from each PDF file using a PDF parsing library (e.g., PyPDF2, pdfminer).
- Preprocess the extracted text by removing stop words, stemming or lemmatizing words, and handling any special characters or formatting
issues.

2. Text Vectorisation:
- Convert the preprocessed text data into a suitable numerical representation (e.g., TF-IDF vectors, word embeddings).
- Experiment with different vectorisation techniques and choose the one that yields the best results.

3. Clustering:
- Implement an unsupervised clustering algorithm (e.g., K-means, DBSCAN, Hierarchical Clustering) to group similar research papers together based on their abstracts.
- Determine the optimal number of clusters using techniques like the elbow method or silhouette analysis.
- Evaluate the quality of the clustering results using appropriate metrics (e.g., silhouette score, Davies-Bouldin index).

4. Visualisation (Bonus):
- Create a visualisation of the clustering results using a dimensionality reduction technique like t-SNE or PCA.
- Display the research paper titles or IDs in the visualisation to provide context.

5. Output Generation:
- Save the clustering results to a file, indicating which papers belong to each cluster.
- Generate a summary report that includes the number of clusters, the number of papers in each cluster, and the key terms or topics associated with each cluster.


# My Results and Approach

## 1. Data Preprocessing:

- Accessed the Excel dataset and downloaded the numerous 79 research papers.
- Retrieves abstract content from each PDF file using a PDF parsing library called PDFMiner and wrote a comprehensive custom logic for only extracting the 'Abstract' section from the PDF.
- Made a Deep Copy so that if the dataset got corrput by anychange I have a backup.
- Then, removed stop words using NLTK's stopwords corpus.
- And removing words to their base form using NLTK's WordNetLemmatizer.
- **Approach:** Why not PDF2 is because it is primarily used to manipulating PDF files whereas PDFMiner is used for extracting information from the file.

## 2. Text Vectorisation:
- Conversion: Transforms preprocessed text data into a suitable numerical format using TF-IDF vectorisation method.
- Plus, printing the TF-IDF Dataset for analyse how does the numerical format looks like.
- **Approach:** Word Embedding requires hard coding, messy code and used for relationship. Whereas TF-IDF has better performace, less code, best for evaluating the importance of word in document also known as Retrieval and Mining.**

## 3. Clustering:
- Implementation: Utilises the K-means clustering algorithm to group research papers with similar abstracts.
- Optimization: Determines the optimal number of clusters using the elbow method and silhouette analysis.
- Evaluation: Assesses clustering quality using metrics such as silhouette score and Davies-Bouldin index.
- **Approach:**
a. Going forward with K-Means Clustering because DBScan took everything into Noise Points, so it doesnt fit for our case.
b. Analysing the Graph of Elbow method and Silhouette analysis => I can see a very sharp peak at 5 for Silhouette analysis, plus I can see more significant increase and decrease whereas in Elbow I can't observe that behaviour so clearly.
c. Observing the Silhouette score and Davies-Bouldin index => I'm confident that Silhouette Score is evaluating the quality of cluserting result more appropriately since I can see a sudden increase and decrease in values at the point 5. Therefore, we are taking The Most Optimal Number of Cluster (K) as 5.

## 4. Visualization (Bonus):
- Creation: Develops a visual representation of clustering results using PCA for dimensionality reduction.
- Context: Displays research paper titles or IDs within the visualization to provide context and clarity.
**Approach:**
a. Using PCA for reducing dimensionality since it's a computationally efficient manner, preserving the variance of the data, and handling large datasets. Whereas t-SNE is for visualizing and exploring the structure of high-dimensional data, particularly when the relationships between data points are nonlinear and complex.
b. Taking the Number of Component as 2 => Because we want to build a 2D Graph so that we can analyse the data points and classification done very easily.

## 5. Output Generation:
- Saving Results: Saves clustering outcomes to a CSV file indicating which papers belong to each cluster.
- Report Generation: Generates a summary report about my analysis.
**Approach:**
a.) Creating 5 Folders according to the number of Clusters.
b.) Each Folder will contain the Research Paper PDFs associated with that cluster.
c.) After that creating a CSV File called `output_file.csv` indicating which papers belong to each cluster.
d.) Lastly, I'm printing the summary report which contains : -
1. Number of clusters created.
2. Distribution of papers in each cluster.
3. Key terms or topics associated with each cluster based on word frequencies.