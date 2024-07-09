# Sentiment Analysis and NLP Processing for Product Reviews

This project performs sentiment analysis and natural language processing on product reviews. It processes a CSV file containing product reviews, applies various NLP techniques, and outputs the results to a new CSV file.

## Features

- Text preprocessing (lowercase conversion, special character removal)
- Tokenization
- Stopword removal
- Part-of-Speech (POS) tagging
- Sentiment analysis

## Prerequisites

- Python 3.x
- pandas
- nltk

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

2. Install the required packages:
pip install pandas nltk

3. The script will automatically download necessary NLTK data on first run.

## Usage

1. Place your input CSV file (named `Reviews_small.csv`) in the project directory.

2. Run the script:
analysis.py

Copy
3. The processed data will be saved to `processed_reviews.csv` in the same directory.

## Input File Format

The input CSV file should have at least the following columns:
- `Summary`: Short summary of the review
- `Text`: Full text of the review

## Output

The script will generate a new CSV file with additional columns:
- `Processed_Summary`: Preprocessed summary text
- `Processed_Text`: Preprocessed full review text
- `POS_Tags_Summary`: Part-of-Speech tags for the summary
- `POS_Tags_Text`: Part-of-Speech tags for the full review
- `Sentiment_Summary`: Sentiment score for the summary
- `Sentiment_Text`: Sentiment score for the full review

## Notes

- The script includes a workaround for SSL certificate issues when downloading NLTK data.
- Sentiment scores range from -1 (most negative) to 1 (most positive).
- The script prints various statistics and sample data to the console during execution.


## Contact

[Maria Bolek] - [maria.bolek@uw.edu.pl]
