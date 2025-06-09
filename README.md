📰 Fake News Detection Analysis
This project focuses on analyzing and detecting fake news using Machine Learning techniques. It was implemented in Python using Jupyter Notebook and is based on a labeled dataset containing both real and fake news articles.

🧠 Project Overview
The notebook walks through the full pipeline of a text classification task:

Data Importing: Loads a dataset of real and fake news.

Exploratory Data Analysis (EDA): Performs initial analysis to understand the structure and distribution of the data.

Data Cleaning: Cleans the text by removing punctuation, converting to lowercase, removing whitespace, and more.

Text Vectorization: Converts text into numerical format using TF-IDF Vectorizer.

Model Training: Trains several classifiers including:

Logistic Regression

Naive Bayes

Random Forest

Passive Aggressive Classifier

Model Evaluation: Uses metrics such as Accuracy, Confusion Matrix, Precision, and Recall to evaluate model performance.

🛠️ Requirements
Python 3.x

Jupyter Notebook

The following Python libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

You can install the required libraries with:

bash
Copy
Edit
pip install -r requirements.txt
Or install them manually:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
📊 Results
The Passive Aggressive Classifier showed the highest accuracy in detecting fake news.

Models demonstrated good performance in classifying fake vs real articles.

📁 Files
fake-news-detection-analysis.ipynb: The main notebook containing all code from data preprocessing to model evaluation.

dataset.csv: The dataset used for training and evaluation (not included in this repository, please add it manually if needed).

📌 Notes
Make sure the dataset is placed in the correct directory before running the notebook.

This project is for educational purposes and can be further developed to include live data from news websites or social media.
