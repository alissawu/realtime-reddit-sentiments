// fetch / display sentiments
function fetchSentimentScores() {
    fetch('/fetch_sentiment')
        .then(response => response.json())
        .then(data => {
            const averageSt = document.getElementById('average');
            const medianSt = document.getElementById('median');

            if (averageSt && medianSt) {
                averageSt.textContent = `Average Sentiment: ${data.average.toFixed(3)}`;
                medianSt.textContent = `Median Sentiment: ${data.median.toFixed(3)}`;
            } else {
                console.error('Error: Elements not found.');
            }
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
                container?.appendChild(element);
            });
        })
        .catch(error => console.error('Error fetching headlines:', error));
}

// update function
document.addEventListener('DOMContentLoaded', () => {
    fetchSentimentScores();
    fetchHeadlines();

    setInterval(() => {
        fetchSentimentScores();
        fetchHeadlines();
    }, 10000); 
});
