# Financial News Analysis (February 2025 to August 2025)

This project analyzes financial news articles to extract insights and visualize their potential impact on the stock market. It uses Natural Language Processing (NLP) techniques to perform sentiment analysis, named entity recognition, and generate word clouds from news headlines. An interactive dashboard is also provided to visualize the analysis results.

## Features

*   **Sentiment Analysis:** Determines the sentiment (positive, negative, or neutral) of financial news articles.
*   **Named Entity Recognition (NER):** Identifies organizations, people, and other entities mentioned in the news.
*   **Word Cloud:** Creates a visual representation of the most frequent words in news headlines.
*   **Interactive Dashboard:** A web-based dashboard to explore the analysis results, including sentiment trends and entity mentions.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/financial-news-analysis.git
    cd financial-news-analysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('vader_lexicon')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    ```

## Usage

### Running the Analysis

The `Analysis.ipynb` notebook contains the complete code for the data analysis and NLP tasks. You can run the notebook using Jupyter:

```bash
jupyter notebook Analysis.ipynb
```

### Running the Dashboard

The `dashboard.py` file contains the code for the interactive dashboard. To run the dashboard, execute the following command:

```bash
python dashboard.py
```

The dashboard will be available at `http://127.0.0.1:8050/` in your web browser.

## Data Source

The project uses the `financial_news_events.csv` dataset. This file contains a collection of financial news articles with the following columns:

*   **title:** The headline of the news article.
*   **url:** The URL of the news article.
*   **source:** The source of the news article.
*   **date:** The publication date of the news article.
*   **text:** The full text of the news article.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create an issue or a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
