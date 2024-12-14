import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Establish a connection to the SQLite database (replace this with your database type)
connection = sqlite3.connect('Data.db')  # Replace with your actual database connection string

# Load data from the BookInfo table into a pandas DataFrame
query = "SELECT * FROM BookInfo"
data = pd.read_sql_query(query, connection)

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


# Function to get BERT embeddings for text
def get_bert_embeddings(texts):
    # Tokenize the texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Get the BERT model's output (we use the last hidden state)
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Get embeddings from the last hidden state (take the mean of all token embeddings)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# Combine relevant features for content-based filtering
data['Combined_Features'] = data['Category'].fillna('') + ' ' + data['Technologies'].fillna('')

# Get BERT embeddings for the combined features
embeddings = get_bert_embeddings(data['Combined_Features'].tolist())

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(embeddings.numpy(), embeddings.numpy())


# Function to get recommendations based on a book title
def get_recommendations(title, cosine_sim=cosine_sim, data=data):
    try:
        # Find the index of the given book title
        idx = data[data['Titles'] == title].index[0]
    except IndexError:
        return f"Book titled '{title}' not found in the dataset."

    # Get similarity scores for all books with the given book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort books by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar books (excluding the first one, which is the book itself)
    sim_indices = [i[0] for i in sim_scores[1:6]]  # Top 5 recommendations

    # Return the titles of the most similar books
    return data.iloc[sim_indices]['Titles'].tolist()


# Example: Get recommendations for a specific book title
example_title = 'Lan-Based Faculty Evaluation System'
recommendations = get_recommendations(example_title)

# Output recommendations
print(f"Recommendations for '{example_title}':")
if isinstance(recommendations, list):
    for rec in recommendations:
        print(rec)
else:
    print(recommendations)  # In case the book title was not found

# Evaluate the recommendations
# Convert `noofviews` to numeric and calculate average similarity score for validation
data['noofviews'] = pd.to_numeric(data['noofviews'], errors='coerce')


def evaluate_accuracy(cosine_sim, data):
    total_score = 0
    count = 0

    for idx in range(len(data)):
        sim_scores = cosine_sim[idx]
        sorted_indices = sim_scores.argsort()[::-1][1:6]  # Exclude self-comparison
        avg_views = data.iloc[sorted_indices]['noofviews'].mean()
        actual_views = data.iloc[idx]['noofviews']

        # Check if both views values are valid
        if not pd.isna(actual_views) and not pd.isna(avg_views):
            total_score += abs(avg_views - actual_views)
            count += 1

    return total_score / count if count > 0 else None


accuracy = evaluate_accuracy(cosine_sim, data)
print(f"Accuracy of content-based filtering with BERT: {accuracy}")

# Close the database connection
connection.close()
