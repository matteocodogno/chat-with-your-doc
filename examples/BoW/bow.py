from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A brown fox is a fast runner.",
    "The dog and the fox are friends."
]

# Step 1: Preprocessing and Tokenization
def preprocess(document):
    # Remove punctuation and convert to lowercase
    document = document.lower()
    document = ''.join(char for char in document if char.isalnum() or char.isspace())
    return document

preprocessed_documents = [preprocess(document) for document in documents]

# Step 2: Building the Vocabulary
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_documents)
vocabulary = vectorizer.get_feature_names_out()

# Step 3: Create Bag-of-Words Vectors
bow_vectors = X.toarray()

# Display the results
print("Vocabulary:", vocabulary)
for i, doc in enumerate(bow_vectors):
    print(f"Document {i+1} BoW Vector:", doc)
