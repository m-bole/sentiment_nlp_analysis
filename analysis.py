import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import re
import ssl

# SSL certificate workaround
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def preprocess_text(text):
    if pd.isna(text):
        return ''
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def pos_tagging(text):
    if pd.isna(text) or text == '':
        return []
    tokens = word_tokenize(text)
    return pos_tag(tokens)

def get_sentiment(text):
    if pd.isna(text):
        return 0
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(str(text))['compound']

def process_reviews_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    print("Original data shape:", df.shape)
    print("\nFirst few rows of original data:")
    print(df.head())

    # Apply preprocessing to 'Summary' and 'Text' columns
    df['Processed_Summary'] = df['Summary'].apply(preprocess_text)
    df['Processed_Text'] = df['Text'].apply(preprocess_text)

    # Apply POS tagging to processed text
    df['POS_Tags_Summary'] = df['Processed_Summary'].apply(pos_tagging)
    df['POS_Tags_Text'] = df['Processed_Text'].apply(pos_tagging)

    # Apply sentiment analysis to original text
    df['Sentiment_Summary'] = df['Summary'].apply(get_sentiment)
    df['Sentiment_Text'] = df['Text'].apply(get_sentiment)

    # Save processed data to CSV
    df.to_csv(output_file, index=False)

    print(f"\nProcessed data saved to '{output_file}'")

    # Print some statistics
    print("\nAverage sentiment of summaries:", df['Sentiment_Summary'].mean())
    print("Average sentiment of full reviews:", df['Sentiment_Text'].mean())

    # Most common POS tags in summaries
    all_pos_tags = [tag for tags in df['POS_Tags_Summary'] for _, tag in tags]
    pos_freq = nltk.FreqDist(all_pos_tags)
    print("\nMost common POS tags in summaries:")
    print(pos_freq.most_common(5))

    print("\nSample processed data:")
    print(df[['Processed_Summary', 'POS_Tags_Summary', 'Sentiment_Summary']].head())

if __name__ == "__main__":
    input_file = "Reviews_small.csv"
    output_file = "processed_reviews.csv"
    process_reviews_csv(input_file, output_file)