// scores
function fetchSentimentScores() {
    fetch('/fetch_sentiment')
        .then(response => response.json())
        .then(data => {
            document.getElementById('average').textContent = `Average Sentiment: ${data.average.toFixed(3)}`;
            document.getElementById('median').textContent = `Median Sentiment: ${data.median.toFixed(3)}`;
        })
        .catch(error => console.error('Error fetching sentiment scores:', error));
}

// headline stuff
function fetchHeadlines() {
    fetch('/fetch_headlines')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('headlinesContainer');
            data.headlines.forEach((headline) => {
                const element = document.createElement('div');
                element.textContent = `${headline.title} - Sentiment: ${headline.sentiment.toFixed(3)}`;
                container.appendChild(element);
            });
        })
        .catch(error => console.error('Error fetching headlines:', error));
}

document.addEventListener('DOMContentLoaded', () => {
    fetchSentimentScores();
    fetchHeadlines();
});
