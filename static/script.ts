// define interfaces for type safety
interface SentimentResponse {
    average: number;
    median: number;
}

interface Headline {
    title: string;
    sentiment: number;
}

interface HeadlinesResponse {
    headlines: Headline[];
}

// fetch and display sentiments
function fetchSentimentScores(subreddit: string): void {
    fetch(`/fetch_sentiment/${subreddit}`)
        .then(response => response.json())
        .then((data: SentimentResponse) => {
            const averageSt = document.getElementById('average') as HTMLParagraphElement;
            const medianSt = document.getElementById('median') as HTMLParagraphElement;
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
function fetchHeadlines(subreddit: string): void {
    fetch(`/fetch_headlines/${subreddit}`)
        .then(response => response.json())
        .then((data: HeadlinesResponse) => {
            const container = document.getElementById('headlinesContainer') as HTMLDivElement;
            if (container) {
                container.innerHTML = ''; // Clear previous entries
                data.headlines.forEach((headline) => {
                    const element = document.createElement('div');
                    element.textContent = `${headline.title} - Sentiment: ${headline.sentiment.toFixed(3)}`;
                    container.appendChild(element);
                });
            }
        })
        .catch(error => console.error('Error fetching headlines:', error));
}

// update function
document.addEventListener('DOMContentLoaded', () => {
    const subreddit = "{{ subreddit }}";
    fetchSentimentScores(subreddit);
    fetchHeadlines(subreddit);

    setInterval(() => {
        fetchSentimentScores(subreddit);
        fetchHeadlines(subreddit);
    }, 10000);
});
