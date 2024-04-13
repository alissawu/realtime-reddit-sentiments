// general fetch
function fetchData(url) {
    return fetch(url).then(response => {
        if (!response.ok) {
            throw new Error(`status: ${response.status}`);
        }
        return response.json();
    }).catch(error => {
        console.error('Fetch error:', error);
        throw error;
    });
}

// update sentiment display
function updateSentimentDisplay(data) {
    const averageSt = document.getElementById('average');
    const medianSt = document.getElementById('median');
    if (averageSt && medianSt) {
        averageSt.textContent = `Average Sentiment: ${data.average.toFixed(3)}`;
        medianSt.textContent = `Median Sentiment: ${data.median.toFixed(3)}`;
    } else {
        console.error('Elements not found.');
    }
}

// update headlines display
function updateHeadlinesDisplay(data) {
    const container = document.getElementById('headlinesContainer');
    if (container) {
        container.innerHTML = ''; // Clear previous entries
        data.headlines.forEach(headline => {
            const element = document.createElement('div');
            element.textContent = `${headline.title} - Sentiment: ${headline.sentiment.toFixed(3)}`;
            container.appendChild(element);
        });
    }
}

// fetch , update sentiment scores
function fetchSentimentScores(subreddit) {
    const url = `/fetch_sentiment/${subreddit}`;
    fetchData(url).then(updateSentimentDisplay);
}

// fetch , update headlines
function fetchHeadlines(subreddit) {
    const url = `/fetch_headlines/${subreddit}`;
    fetchData(url).then(updateHeadlinesDisplay);
}

document.addEventListener('DOMContentLoaded', () => {
    const subreddit = new URL(window.location.href).pathname.split('/')[1]; // get the subreddit from the URL
    fetchSentimentScores(subreddit);
    fetchHeadlines(subreddit);

    setInterval(() => {
        fetchSentimentScores(subreddit);
        fetchHeadlines(subreddit);
    }, 10000); // 10 sec
});
