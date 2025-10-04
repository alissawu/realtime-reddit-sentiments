// unified fetch
function fetchAll(subreddit) {
  return fetch(`/api/sub/${subreddit}`).then((r) => {
    if (!r.ok) throw new Error(`status: ${r.status}`);
    return r.json();
  });
}

// render everything (avg, median, headlines, optional freshness)
function renderAll(data) {
  const averageSt = document.getElementById('average');
  const medianSt  = document.getElementById('median');
  const container = document.getElementById('headlinesContainer');

  if (averageSt) averageSt.textContent = `Average Sentiment: ${data.average.toFixed(3)}`;
  if (medianSt)  medianSt.textContent  = `Median Sentiment: ${data.median.toFixed(3)}`;

  if (container) {
    container.innerHTML = '';
    (data.headlines || []).forEach((h) => {
      const el = document.createElement('div');
      el.textContent = `${h.title} - Sentiment: ${h.sentiment.toFixed(3)}`;
      container.appendChild(el);
    });
  }

  // freshness label - if wanted, <p id="freshness"></p> in HTML 
  const freshness = document.getElementById('freshness');
  if (freshness && data.ts) {
    const secs = Math.max(0, Math.round(Date.now()/1000 - data.ts));
    freshness.textContent = `Last updated: ${secs}s ago`;
  }
}

// Search 
function searchSubreddit() {
  const input = document.getElementById('subredditSearch');
  const subreddit = input.value.trim().replace(/^r\//, '');
  if (subreddit) window.location.href = `/${subreddit}`;
}

// Page bootstrap
document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('subredditSearch');
  if (searchInput) {
    searchInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') searchSubreddit();
    });
  }

  const subreddit = new URL(window.location.href).pathname.split('/')[1];
  if (subreddit && subreddit !== '' && subreddit !== 'modelnotes') {
    const run = () => fetchAll(subreddit).then(renderAll).catch(console.error);

    run();

    // Optional: ping backend to warm the HF endpoint without blocking the user
    fetch('/warm').catch(() => {});

    // poll every 60s
    setInterval(run, 60000);
  }
});
