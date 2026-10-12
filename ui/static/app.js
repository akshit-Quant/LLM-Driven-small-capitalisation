const API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
  ? (window.location.port === '8000' ? '' : `${window.location.protocol}//${window.location.hostname}:8000`)
  : 'http://localhost:8000';

function apiUrl(path) {
  if (!path || path.startsWith('http://') || path.startsWith('https://')) return path;
  return `${API_BASE}${path.startsWith('/') ? path : '/' + path}`;
}

let dashboardState = {
  accountBalance: 0,
  buyingPower: 0,
  totalValue: 0,
  pnl: 0,
  cash: 0,
  riskScore: 0,
  sentiment: 0,
  trend: 0,
  portfolio: {},
  positions: [],
  history: [],
  watchlist: [],
  heatmap: [],
  newsReviews: [],
  candles: []
};
let selectedSymbol = 'AAPL';
let selectedNewsSymbol = 'ALL';
let selectedPeriod = 'D1';
let chartOptions = { type: 'candles', grid: true, ma: true, crosshair: true };
let liveQuoteInFlight = false;
let watchlistQuotesInFlight = false;
let dashboardInFlight = false;
let jobInFlight = false;

function currency(value) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2
  }).format(value);
}

function pct(value) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}

function setupRevealAnimations() {
  const revealItems = document.querySelectorAll('.reveal, .reveal-item, .reveal-group');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.12 });

  revealItems.forEach((item) => observer.observe(item));
}

function renderStats() {
  document.getElementById('accountBalance').textContent = currency(dashboardState.accountBalance);
  document.getElementById('buyingPower').textContent = currency(dashboardState.buyingPower);
  document.getElementById('totalValue').textContent = currency(dashboardState.totalValue);
  document.getElementById('pnlValue').textContent = currency(dashboardState.pnl);
  document.getElementById('cashValue').textContent = currency(dashboardState.cash);
  const riskScore = Number(dashboardState.riskScore);
  document.getElementById('riskScore').textContent = Number.isFinite(riskScore) ? `${riskScore}%` : '--';
  document.getElementById('riskMeter').style.width = `${Number.isFinite(riskScore) ? Math.max(0, Math.min(100, riskScore)) : 0}%`;
  document.getElementById('sentimentValue').textContent = dashboardState.sentiment.toFixed(2);
  document.getElementById('totalTrend').textContent = pct(dashboardState.trend);
  document.getElementById('pnlTrend').textContent = pct(dashboardState.trend * 1.5);
  document.getElementById('sentimentTrend').textContent =
    dashboardState.sentiment >= 0.05 ? 'Bullish' : (dashboardState.sentiment <= -0.05 ? 'Bearish' : 'Neutral');
  const metrics = dashboardState.riskMetrics || {};
  const sharpe = Number(metrics.sharpe);
  const drawdown = Number(metrics.drawdown);
  const volatility = Number(metrics.volatility);
  const concentration = Number(metrics.concentration);
  document.getElementById('riskScoreDetail').textContent =
    Number.isFinite(volatility) && Number.isFinite(drawdown) && Number.isFinite(concentration)
      ? `Vol ${volatility.toFixed(1)}% · DD ${drawdown.toFixed(1)}% · Max position ${concentration.toFixed(1)}%`
      : 'Portfolio risk metrics unavailable';
  document.getElementById('sharpeValue').textContent = Number.isFinite(sharpe) ? sharpe.toFixed(2) : '--';
  document.getElementById('drawdownValue').textContent = Number.isFinite(drawdown) ? `${drawdown.toFixed(2)}%` : '--';
  document.getElementById('sharpeNote').textContent = 'Calculated from live portfolio curve';
  document.getElementById('drawdownNote').textContent = 'Current maximum peak-to-trough loss';
  const marketMetrics = dashboardState.marketMetrics || {};
  const trendStrength = Number(marketMetrics.trendStrength);
  const trendChange = Number(marketMetrics.trendChange);
  const atr = Number(marketMetrics.atr);
  const beta = Number(marketMetrics.beta);
  document.getElementById('trendStrengthValue').textContent = Number.isFinite(trendStrength) ? trendStrength.toFixed(1) : '--';
  document.getElementById('trendStrengthNote').textContent =
    Number.isFinite(trendChange) ? `${trendChange >= 0 ? '+' : ''}${trendChange.toFixed(2)}% over recent candles` : 'Live price trend';
  document.getElementById('atrValue').textContent = Number.isFinite(atr) ? `${atr.toFixed(2)}%` : '--';
  document.getElementById('atrNote').textContent = 'Average true range of recent candles';
  document.getElementById('sideTrend').textContent = Number.isFinite(trendChange) ? (trendChange >= 0 ? 'Long' : 'Short') : '--';
  document.getElementById('sideVolatility').textContent = Number.isFinite(atr)
    ? (atr < 1 ? 'Low' : (atr < 3 ? 'Moderate' : 'High'))
    : '--';
  document.getElementById('sideBeta').textContent = Number.isFinite(beta) ? beta.toFixed(2) : 'N/A';
}

function renderTicker() {
  const tickerTrack = document.getElementById('tickerTrack');
  const items = (dashboardState.watchlist || []).filter((item) => Number.isFinite(Number(item.last)));
  tickerTrack.innerHTML = [...items, ...items].map((item) => `
    <span>${item.label || item.symbol} <strong class="${item.change >= 0 ? 'value-up' : 'value-down'}">${pct(item.change || 0)}</strong></span>
  `).join('');
}

function renderMarketClock() {
  document.getElementById('marketClock').textContent = new Date().toLocaleTimeString();
}

function setHealthStatus(id, label, tone, detail) {
  const element = document.getElementById(id);
  if (!element) return;
  element.textContent = label;
  element.className = `status-pill status-${tone}`;
  const detailNode = document.getElementById(`${id.replace('Status', 'Detail')}`);
  if (detailNode) {
    detailNode.textContent = detail;
  }
}

async function fetchSystemStatus() {
  try {
    const [healthResp, executionResp, brokerResp, portfolioResp, newsResp] = await Promise.allSettled([
      fetch(apiUrl('/health'), { cache: 'no-store' }),
      fetch(apiUrl('/api/execution/status'), { cache: 'no-store' }),
      fetch(apiUrl('/api/broker/status'), { cache: 'no-store' }),
      fetch(apiUrl('/api/portfolio?holding=5'), { cache: 'no-store' }),
      fetch(apiUrl('/api/news/status'), { cache: 'no-store' })
    ]);

    const health = healthResp.status === 'fulfilled' && healthResp.value.ok ? await healthResp.value.json().catch(() => ({})) : {};
    const execution = executionResp.status === 'fulfilled' && executionResp.value.ok ? await executionResp.value.json().catch(() => ({})) : {};
    const broker = brokerResp.status === 'fulfilled' && brokerResp.value.ok ? await brokerResp.value.json().catch(() => ({})) : {};
    const portfolio = portfolioResp.status === 'fulfilled' && portfolioResp.value.ok ? await portfolioResp.value.json().catch(() => ({})) : {};
    const news = newsResp.status === 'fulfilled' && newsResp.value.ok ? await newsResp.value.json().catch(() => ({})) : {};

    const dataHealthy = Boolean(news.hasData && news.fresh) && !news.lastError && !news.running;
    setHealthStatus(
      'dataHealthStatus',
      dataHealthy ? 'Healthy' : 'Degraded',
      dataHealthy ? 'good' : 'warning',
      dataHealthy
        ? `news data ready · updated ${new Date(news.updatedAt).toLocaleTimeString()}`
        : (news.running
          ? 'news refresh running'
          : (news.lastError || (!news.hasData
            ? 'no processed news data'
            : `news data stale · updated ${news.updatedAt ? new Date(news.updatedAt).toLocaleTimeString() : 'unknown'}`)))
    );

    const portfolioReady = portfolio && portfolio.status === 'ready';
    setHealthStatus(
      'portfolioHealthStatus',
      portfolioReady ? 'Ready' : 'Pending',
      portfolioReady ? 'good' : 'warning',
      portfolioReady
        ? `H${portfolio.holding || 5} backtest loaded`
        : (portfolio.message || 'portfolio artifacts unavailable')
    );

    const paper = execution && execution.paper ? execution.paper : {};
    const executionState = paper.running ? 'Running' : (paper.halted ? 'Halted' : 'Paper');
    const executionTone = paper.running ? 'good' : (paper.halted ? 'bad' : 'neutral');
    setHealthStatus(
      'executionHealthStatus',
      executionState,
      executionTone,
      paper.lastAction || 'autonomous paper mode available'
    );
    const paperStatus = document.getElementById('paperStatus');
    paperStatus.textContent = paper.running ? 'Paper trading running' : 'Paper trading stopped';
    paperStatus.classList.toggle('ready', Boolean(paper.running));
    paperStatus.classList.toggle('unavailable', !paper.running);
    document.getElementById('paperStartBtn').disabled = Boolean(paper.running || paper.halted);
    document.getElementById('paperStopBtn').disabled = !paper.running;

    const alpaca = broker && broker.alpaca;
    const alpacaOk = Boolean(alpaca && alpaca.configured);
    const brokerState = alpacaOk ? 'Ready' : 'Idle';
    const brokerTone = alpacaOk ? 'good' : 'neutral';
    const buyingPowerStr = alpaca && alpaca.buyingPower ? ` · $${Math.round(alpaca.buyingPower).toLocaleString()} BP` : '';
    const brokerText = alpacaOk ? `Alpaca Paper active${buyingPowerStr}` : 'Alpaca: Click to add keys';
    setHealthStatus('brokerHealthStatus', brokerState, brokerTone, brokerText);

    const overall = health && health.ok ? 'online' : 'offline';
    document.getElementById('marketClock').title = `Server: ${overall} | execution: ${executionState}`;
  } catch (error) {
    console.warn('System status unavailable:', error);
    setHealthStatus('dataHealthStatus', 'Offline', 'bad', 'monitoring unavailable');
    setHealthStatus('portfolioHealthStatus', 'Pending', 'warning', 'run /run-portfolio to generate metrics');
    setHealthStatus('executionHealthStatus', 'Paper', 'neutral', 'autonomous paper mode available');
    setHealthStatus('brokerHealthStatus', 'Idle', 'neutral', 'Alpaca + IBKR disconnected');
  }
}

function renderPortfolioStatus() {
  const status = document.getElementById('portfolioStatus');
  const portfolio = dashboardState.portfolio || {};
  const ready = portfolio.status === 'ready';
  status.textContent = ready
    ? `Backtest live: H${portfolio.holding} · ${portfolio.best.model} · net ${pct((portfolio.best.cum_return_net || 0) * 100)}`
    : `Backtest unavailable · ${portfolio.message || 'market data required'}`;
  status.classList.toggle('ready', ready);
  status.classList.toggle('unavailable', !ready);
}

function renderPositions() {
  const positionsList = document.getElementById('positionsList');
  if (!dashboardState.positions.length) {
    positionsList.innerHTML = '<div class="empty-state">No open positions.</div>';
    return;
  }
  positionsList.innerHTML = dashboardState.positions
    .map((position) => {
      const value = position.quantity * position.price;
      const className = position.change >= 0 ? 'value-up' : 'value-down';
      return `
        <div class="position-row">
          <div class="position-meta">
            <strong>${position.symbol}</strong>
            <span>${position.name}</span>
          </div>
          <div class="position-meta">
            <strong>${position.quantity} sh</strong>
            <span>${currency(position.price)}</span>
          </div>
          <div class="position-value">
            <strong>${currency(value)}</strong>
            <span>${position.change >= 0 ? 'Long' : 'Short'}</span>
          </div>
          <div class="position-change ${className}">${pct(position.change)}</div>
          <button class="close-position" type="button" data-symbol="${position.symbol}" aria-label="Close ${position.symbol} position">Close</button>
        </div>
      `;
    })
    .join('');
}

function renderHistory() {
  const historyList = document.getElementById('historyList');
  if (!dashboardState.history.length) {
    historyList.innerHTML = '<div class="empty-state">No paper-trading history.</div>';
    return;
  }
  historyList.innerHTML = dashboardState.history
    .map((trade) => `
      <div class="history-row">
        <div>
          <strong>${trade.symbol}</strong>
          <span>${trade.name || trade.symbol}</span>
          <span>${trade.time}</span>
        </div>
        <div class="side ${trade.side.toLowerCase()}">${trade.side}</div>
        <div>
          <strong>${trade.qty} sh</strong>
          <span>${currency(trade.price)}</span>
        </div>
        <div class="position-value">
          <strong>${trade.status || (trade.side === 'Buy' ? 'Filled' : 'Closed')}</strong>
          <span>${trade.status === 'Open' && trade.id ? `<button class="cancel-order" data-order-id="${trade.id}">Cancel</button>` : (trade.side === 'Buy' ? 'Exited' : 'Exited')}</span>
        </div>
      </div>
    `)
    .join('');
}

function renderWatchlist() {
  const watchlist = document.getElementById('watchlist');
  const newsSymbol = document.getElementById('newsSymbol');
  const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[character]);
  newsSymbol.innerHTML = [
    '<option value="ALL">All stocks</option>',
    ...dashboardState.watchlist.map((item) =>
      `<option value="${escapeHtml(item.symbol)}">${escapeHtml(item.label || item.symbol)}</option>`)
  ].join('');
  newsSymbol.value = selectedNewsSymbol;
  watchlist.innerHTML = `
    <div class="watch-row watchlist-header" aria-hidden="true">
      <span>Symbol / News volume</span>
      <span>Latest news</span>
      <span>Last</span>
      <span>Day change</span>
    </div>
    ${dashboardState.watchlist
    .map((item) => {
      const last = Number(item.last);
      const change = Number(item.change);
      const hasQuote = Number.isFinite(last);
      const hasChange = Number.isFinite(change);
      const headline = item.newsHeadline || 'No headline available';
      const parsedDate = item.newsDate ? new Date(item.newsDate) : null;
      const newsDate = parsedDate && !Number.isNaN(parsedDate.getTime())
        ? parsedDate.toLocaleDateString()
        : '';
      const newsVolume = `${item.newsCount ?? 0} articles${item.newsCountSource ? ` · ${item.newsCountSource}` : ''}`;
      return `
      <button class="watch-row stock-button" data-symbol="${escapeHtml(item.symbol)}">
        <div>
          <strong>${escapeHtml(item.label || item.symbol)}</strong>
          <span>${escapeHtml(newsVolume)}</span>
        </div>
        <div class="watch-news">
          <strong title="${escapeHtml(headline)}">${escapeHtml(headline)}</strong>
          <span>${escapeHtml([item.newsSource, newsDate].filter(Boolean).join(' · '))}</span>
        </div>
        <div class="position-value">
          <strong>${hasQuote ? last.toFixed(2) : '--'}</strong>
          <span>Last</span>
        </div>
        <div class="side ${hasChange ? (change >= 0 ? 'buy' : 'sell') : 'neutral'}">${hasChange ? pct(change) : '--'}</div>
      </button>
    `;
    })
    .join('')}
  `;
}

function renderHeatmap() {
  const heatmap = document.getElementById('heatmap');
  heatmap.innerHTML = dashboardState.heatmap
    .flat()
    .map((type) => `<div class="market-box ${type}">${type === 'buy' ? 'BUY' : type === 'sell' ? 'SELL' : 'FLAT'}</div>`)
    .join('');
}

function renderNewsReviews() {
  const container = document.getElementById('newsReviews');
  const reviews = dashboardState.newsReviews || [];
  if (!reviews.length) {
    container.innerHTML = `<div class="empty-state">${dashboardState.newsMessage || 'No news data available.'}</div>`;
    return;
  }
  container.innerHTML = reviews.map((item) => {
    const score = Number(item.sentiment_final ?? item.raw_score ?? 0);
    const scoreClass = score >= 0.05 ? 'buy' : score <= -0.05 ? 'sell' : 'neutral';
    const summary = item.qwen_summary || item.summary_text || 'No summary available';
    return `
      <article class="news-review ${scoreClass}">
        <div class="news-review-head">
          <strong>${item.ticker || 'UNKNOWN'}</strong>
          <span>${item.valid_time || ''}</span>
          <b>${score >= 0 ? '+' : ''}${score.toFixed(3)}</b>
        </div>
        <p class="news-summary">${summary}</p>
        <div class="news-review-grid">
          <div><span>Event</span><strong>${item.qwen_event || 'Not extracted'}</strong></div>
          <div><span>Signal explanation</span><strong>${item.qwen_signal_explanation || 'Not generated'}</strong></div>
          <div><span>Qwen rationale</span><strong>${item.qwen_rationale || 'Not generated'}</strong></div>
          <div><span>FinBERT probabilities</span><strong>POS ${(Number(item.agg_p_pos || 0) * 100).toFixed(1)}% · NEG ${(Number(item.agg_p_neg || 0) * 100).toFixed(1)}% · NEU ${(Number(item.agg_p_neu || 0) * 100).toFixed(1)}%</strong></div>
        </div>
      </article>
    `;
  }).join('');
  if (dashboardState.newsUpdatedAt) {
    container.insertAdjacentHTML('afterbegin', `<div class="empty-state">Updated ${new Date(dashboardState.newsUpdatedAt).toLocaleString()} · ${dashboardState.newsSource || 'pipeline output'}</div>`);
  }
}

function drawCandles() {
  const canvas = document.getElementById('candlestickChart');
  const ctx = canvas.getContext('2d');
  const width = Math.max(640, Math.floor(canvas.clientWidth || canvas.width));
  const height = 360;
  canvas.width = width;
  canvas.height = height;
  const padding = 38;
  const candles = dashboardState.candles || [];
  if (!candles.length || candles.some((c) => ![c.o, c.h, c.l, c.c].every(Number.isFinite))) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#849197';
    ctx.font = '12px Consolas';
    ctx.fillText('No chart data available', 24, 32);
    return;
  }
  const low = Math.min(...candles.map((c) => c.l));
  const high = Math.max(...candles.map((c) => c.h));
  const plotHeight = height - padding * 2;
  const plotWidth = width - padding * 2;
  const step = plotWidth / candles.length;
  const candleWidth = Math.max(3, Math.min(14, step * 0.62));
  const y = (value) => padding + ((high - value) / (high - low || 1)) * plotHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#080b0c';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;

  for (let i = 0; i < 6; i += 1) {
    const y = padding + ((height - padding * 2) / 4) * i;
    if (chartOptions.grid) {
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }
    ctx.fillStyle = '#849197';
    ctx.font = '10px Consolas';
    ctx.fillText((high - ((high - low) / 4) * i).toFixed(2), 4, y + 3);
  }

  const closes = candles.map((c) => c.c);
  const movingAverage = closes.map((_, index) => {
    const window = closes.slice(Math.max(0, index - 19), index + 1);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
  });
  if (chartOptions.ma) {
    ctx.strokeStyle = '#f0a400';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    movingAverage.forEach((value, index) => {
      const x = padding + index * step + step / 2;
      index ? ctx.lineTo(x, y(value)) : ctx.moveTo(x, y(value));
    });
    ctx.stroke();
  }

  if (chartOptions.type === 'line') {
    ctx.beginPath();
  }
  candles.forEach((candle, index) => {
    const x = padding + index * step + step / 2;
    const top = y(candle.h);
    const bottom = y(candle.l);
    const openY = y(candle.o);
    const closeY = y(candle.c);
    const isUp = candle.c >= candle.o;

    ctx.strokeStyle = isUp ? '#42d17b' : '#f0605f';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 1;
    if (chartOptions.type === 'line') {
      index ? ctx.lineTo(x, closeY) : ctx.moveTo(x, closeY);
      return;
    }
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    if (chartOptions.type === 'bars') {
      ctx.beginPath();
      ctx.moveTo(x - candleWidth / 2, openY);
      ctx.lineTo(x, openY);
      ctx.moveTo(x, closeY);
      ctx.lineTo(x + candleWidth / 2, closeY);
      ctx.stroke();
    } else {
      ctx.fillRect(x - candleWidth / 2, Math.min(openY, closeY), candleWidth, Math.max(2, Math.abs(closeY - openY)));
    }
  });
  if (chartOptions.type === 'line') {
    ctx.strokeStyle = '#42d17b';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  const latest = candles[candles.length - 1];
  document.getElementById('chartOhlc').textContent =
    `O ${latest.o.toFixed(2)}  H ${latest.h.toFixed(2)}  L ${latest.l.toFixed(2)}  C ${latest.c.toFixed(2)}`;
  document.getElementById('chartIndicatorStatus').textContent =
    `MA20 ${chartOptions.ma ? 'ON' : 'OFF'} · GRID ${chartOptions.grid ? 'ON' : 'OFF'}`;

  const markers = (dashboardState.history || []).filter(
    (trade) => trade.symbol === selectedSymbol && trade.status !== 'Cancelled'
  );
  markers.slice(0, 12).forEach((trade, index) => {
    const markerX = padding + Math.max(0, candles.length - 1 - index) * step + step / 2;
    const markerY = trade.side === 'Buy' ? y(latest.l) + 8 : y(latest.h) - 8;
    ctx.fillStyle = trade.side === 'Buy' ? '#42d17b' : '#f0605f';
    ctx.beginPath();
    ctx.moveTo(markerX, markerY);
    ctx.lineTo(markerX - 5, markerY + (trade.side === 'Buy' ? 7 : -7));
    ctx.lineTo(markerX + 5, markerY + (trade.side === 'Buy' ? 7 : -7));
    ctx.closePath();
    ctx.fill();
  });
}

async function fetchMarket(symbol = selectedSymbol, period = selectedPeriod) {
  selectedSymbol = symbol.toUpperCase();
  selectedPeriod = period;
  fetchNewsReviews(selectedSymbol);
  const notice = document.getElementById('chartNotice');
  try {
    const response = await fetch(apiUrl(`/api/market?symbol=${encodeURIComponent(selectedSymbol)}&period=${period}&ts=${Date.now()}`), { cache: 'no-store' });
    if (!response.ok) throw new Error((await response.json()).error || `HTTP ${response.status}`);
    const market = await response.json();
    const quotePrice = market.last.toFixed(4);
    document.getElementById('orderStop').value = quotePrice;
    document.getElementById('orderTarget').value = quotePrice;
    dashboardState.candles = market.candles;
    document.getElementById('chartSymbol').textContent = market.symbol;
    document.getElementById('orderSymbol').value = market.symbol;
    notice.textContent = `${market.symbol} · ${market.source || 'market data'} · last ${market.last.toFixed(2)} · ${pct(market.change)}`;
    drawCandles();
  } catch (error) {
    notice.textContent = `Chart unavailable: ${error.message}`;
  }

}

async function fetchNewsReviews(symbol = selectedNewsSymbol) {
  try {
    selectedNewsSymbol = symbol || 'ALL';
    const query = selectedNewsSymbol === 'ALL'
      ? ''
      : `symbol=${encodeURIComponent(selectedNewsSymbol)}&`;
    const response = await fetch(apiUrl(`/api/dashboard?${query}ts=${Date.now()}`), { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    dashboardState.newsReviews = payload.newsReviews || [];
    if (typeof payload.selectedSentiment === 'number') {
      dashboardState.sentiment = payload.selectedSentiment;
      renderStats();
    }
    renderNewsReviews();
  } catch (error) {
    console.warn('News reviews unavailable:', error);
  }
}

async function refreshLiveQuote() {
  if (liveQuoteInFlight) return;
  liveQuoteInFlight = true;
  try {
    const response = await fetch(apiUrl(`/api/market/live?symbol=${encodeURIComponent(selectedSymbol)}&ts=${Date.now()}`), { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const market = await response.json();
    const position = (dashboardState.positions || []).find((item) => item.symbol === market.symbol);
    if (position) {
      position.price = market.last;
      position.change = market.change;
      renderPositions();
    }
    document.getElementById('chartSymbol').textContent = `${market.symbol} · ${selectedPeriod}`;
    document.getElementById('chartNotice').textContent =
      `${market.symbol} · live quote ${market.last.toFixed(2)} · ${pct(market.change)} · ${selectedPeriod} chart · updated ${new Date().toLocaleTimeString()}`;
    renderStats();
  } catch (error) {
    console.warn('Live quote unavailable:', error);
  } finally {
    liveQuoteInFlight = false;
  }
}

async function refreshWatchlistQuotes() {
    if (watchlistQuotesInFlight) return;
    watchlistQuotesInFlight = true;
    try {
    for (let start = 0; start < dashboardState.watchlist.length; start += 4) {
      const batch = dashboardState.watchlist.slice(start, start + 4);
      await Promise.all(batch.map(async (item) => {
        try {
          const response = await fetch(apiUrl(`/api/market/live?symbol=${encodeURIComponent(item.symbol)}`));
          if (!response.ok) return;
          const quote = await response.json();
          item.last = quote.last;
          item.change = quote.change;
        } catch (error) {
          console.warn(`Quote unavailable for ${item.symbol}:`, error);
        }
      }));
    }
    renderWatchlist();
    renderTicker();
    renderMarketClock();
    } finally {
      watchlistQuotesInFlight = false;
    }
}

function bindTimeframeButtons() {
  document.querySelectorAll('.timeframe').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.timeframe').forEach((b) => b.classList.remove('active'));
      button.classList.add('active');
      fetchMarket(selectedSymbol, button.dataset.timeframe);
    });
  });
}

function bindChartTools() {
  document.getElementById('chartType').addEventListener('change', (event) => {
    chartOptions.type = event.target.value;
    drawCandles();
  });
  document.querySelectorAll('.chart-tool').forEach((button) => {
    button.addEventListener('click', () => {
      const key = button.dataset.chartTool;
      chartOptions[key] = !chartOptions[key];
      button.classList.toggle('active', chartOptions[key]);
      drawCandles();
    });
  });
  const canvas = document.getElementById('candlestickChart');
  canvas.addEventListener('mousemove', (event) => {
    if (!chartOptions.crosshair) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width * canvas.width;
    const index = Math.max(0, Math.min((dashboardState.candles || []).length - 1,
      Math.floor((x - 38) / ((canvas.width - 76) / (dashboardState.candles || []).length))));
    const candle = dashboardState.candles?.[index];
    if (candle) {
      document.getElementById('chartOhlc').textContent =
        `O ${candle.o.toFixed(2)}  H ${candle.h.toFixed(2)}  L ${candle.l.toFixed(2)}  C ${candle.c.toFixed(2)}`;
    }
  });
}

function bindOrderActions() {
  document.querySelectorAll('.order-toggle .side').forEach((button) => {
    button.addEventListener('click', () => {
        document.querySelectorAll('.order-toggle .side').forEach((b) => b.classList.remove('active'));
      button.classList.add('active');
    });
  });
  document.getElementById('watchlist').addEventListener('click', (event) => {
      const button = event.target.closest('[data-symbol]');
      if (button) {
        selectedNewsSymbol = button.dataset.symbol;
        fetchMarket(button.dataset.symbol, selectedPeriod);
      }
  });

  document.getElementById('tradeBtn').addEventListener('click', () => {
    openModal();
  });

  document.getElementById('ticketBtn').addEventListener('click', () => {
    openModal();
  });

  document.getElementById('closeModal').addEventListener('click', () => closeModal());
  document.querySelector('.modal').addEventListener('click', (event) => {
    if (event.target.classList.contains('modal')) closeModal();
  });
  document.querySelector('.modal-confirm').addEventListener('click', async () => {
    const side = document.querySelector('.order-toggle .side.active').dataset.side;
    const symbol = document.getElementById('orderSymbol').value.trim().toUpperCase();
    const quantity = Number(document.getElementById('orderQuantity').value);
    try {
      const response = await fetch(apiUrl('/api/orders'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ side, symbol, quantity })
      });
      const result = await response.json();
      if (!response.ok) {
        window.alert(result.error || 'Order failed');
        return;
      }
      dashboardState.history = [result.order, ...dashboardState.history];
      renderHistory();
      closeModal();
    } catch (error) {
      window.alert(`Order failed: ${error.message}`);
    }
  });
  document.querySelector('.modal-cancel').addEventListener('click', closeModal);
}

function openModal() {
  const modal = document.getElementById('orderModal');
  const side = document.querySelector('.order-toggle .side.active').dataset.side;
  const symbol = document.getElementById('orderSymbol').value.trim().toUpperCase();
  const quantity = Number(document.getElementById('orderQuantity').value) || 0;
  document.getElementById('modalSide').textContent = side;
  document.getElementById('modalQuantity').textContent = quantity.toFixed(2);
  document.getElementById('modalSymbol').textContent = symbol;
  document.getElementById('modalMargin').textContent = 'Market fill';
  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');
}

function closeModal() {
  const modal = document.getElementById('orderModal');
  modal.classList.add('hidden');
  modal.setAttribute('aria-hidden', 'true');
}

async function fetchDashboard() {
  if (dashboardInFlight) return;
  dashboardInFlight = true;
  try {
    const response = await fetch(apiUrl(`/api/dashboard?symbol=${encodeURIComponent(selectedSymbol)}&ts=${Date.now()}`), { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    dashboardState = { ...dashboardState, ...payload, newsReviews: dashboardState.newsReviews };
    if (typeof payload.selectedSentiment === 'number') {
      dashboardState.sentiment = payload.selectedSentiment;
    }
    const ordersResponse = await fetch(apiUrl(`/api/orders?ts=${Date.now()}`), { cache: 'no-store' });
    if (ordersResponse.ok) {
      const orderPayload = await ordersResponse.json();
      dashboardState.history = [...(orderPayload.orders || []), ...(dashboardState.history || [])]
        .filter((item, index, items) => !item.id || items.findIndex((candidate) => candidate.id === item.id) === index);
    }
  } catch (error) {
    console.warn('Dashboard API unavailable; retaining the last known state:', error);
  } finally {
    dashboardInFlight = false;
  }
  refreshDashboard();
}

async function runJob(path, buttonId, successMessage) {
  if (jobInFlight) return;
  jobInFlight = true;
  const button = document.getElementById(buttonId);
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = 'RUNNING...';
  try {
    const response = await fetch(apiUrl(path), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    const result = await response.json();
    if (!response.ok || result.ok === false) throw new Error(result.error || `HTTP ${response.status}`);
    document.getElementById('chartNotice').textContent = successMessage(result);
    await Promise.all([fetchDashboard(), fetchSystemStatus()]);
  } catch (error) {
    document.getElementById('chartNotice').textContent = `Operation failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = originalText;
    jobInFlight = false;
  }
}

function refreshDashboard() {
  renderStats();
  renderPortfolioStatus();
  renderPositions();
  renderHistory();
  renderWatchlist();
  renderTicker();
  renderHeatmap();
  renderNewsReviews();
  drawCandles();
}

function bindNavigation() {
  const targets = ['overview', 'portfolio', 'signals', 'history'];
  document.querySelectorAll('.nav').forEach((button, index) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.nav').forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      document.getElementById(targets[index]).scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  setupRevealAnimations();
  bindTimeframeButtons();
  bindChartTools();
  bindOrderActions();
  bindNavigation();
  document.getElementById('newsSymbol').addEventListener('change', (event) => {
    fetchNewsReviews(event.target.value === 'ALL' ? '' : event.target.value);
  });
  fetchSystemStatus();
  fetchDashboard();
  fetchNewsReviews('ALL');
  fetchMarket();
  refreshWatchlistQuotes();
  let refreshTick = 0;
  window.setInterval(() => {
    refreshTick += 1;
    renderMarketClock();
    refreshLiveQuote();
    refreshWatchlistQuotes();
    if (refreshTick % 5 === 0) {
      fetchDashboard();
      fetchSystemStatus();
    }
  }, 1000);

  document.getElementById('refreshBtn').addEventListener('click', refreshAll);
  document.getElementById('newsBtn').addEventListener('click', () =>
    runJob('/api/news/refresh', 'newsBtn', (result) =>
      result.running ? 'News refresh is already running' : `News updated: ${result.rows} rows`));
  document.getElementById('backtestBtn').addEventListener('click', () =>
    runJob('/run-portfolio', 'backtestBtn', (result) => `Backtest complete: ${result.models?.length || 0} models`));
  document.getElementById('paperStartBtn').addEventListener('click', async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
      const response = await fetch(apiUrl('/api/paper/start'), { method: 'POST' });
      const result = await response.json();
      if (!response.ok || result.ok === false || !result.running) {
        throw new Error(result.haltReason || result.error || `HTTP ${response.status}`);
      }
      const alpacaOk = result.broker === 'alpaca' || result.alpacaConfigured;
      document.getElementById('chartNotice').textContent = alpacaOk
        ? 'Paper trading started (Alpaca Paper API Connected)'
        : 'Paper trading started (Simulated mode · configure ALPACA KEYS to trade on Alpaca)';
      await fetchSystemStatus();
    } catch (error) {
      document.getElementById('chartNotice').textContent = `Paper trading failed: ${error.message}`;
    } finally {
      button.disabled = false;
    }
  });
  document.getElementById('paperStopBtn').addEventListener('click', async () => {
    const response = await fetch(apiUrl('/api/paper/stop'), { method: 'POST' });
    const result = await response.json();
    if (!response.ok || result.ok === false) {
      document.getElementById('chartNotice').textContent = `Paper trading stop failed: ${result.error || response.status}`;
      return;
    }
    await fetchSystemStatus();
  });
  document.getElementById('historyList').addEventListener('click', async (event) => {
    const button = event.target.closest('.cancel-order');
    if (!button) return;
    const response = await fetch(apiUrl(`/api/orders/${button.dataset.orderId}/cancel`), { method: 'POST' });
    const result = await response.json();
    if (!response.ok) {
      window.alert(result.error || 'Order cancellation failed');
      return;
    }
    await fetchDashboard();
  });
  document.getElementById('positionsList').addEventListener('click', async (event) => {
    const button = event.target.closest('.close-position');
    if (!button) return;
    const position = dashboardState.positions.find((item) => item.symbol === button.dataset.symbol);
    if (!position) return;
    const name = position.name || position.symbol;
    if (!window.confirm(`Close the full ${position.quantity} share ${name} position at the current market price?`)) {
      return;
    }
    button.disabled = true;
    try {
      const response = await fetch(apiUrl('/api/positions/close'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: position.symbol })
      });
      const result = await response.json();
      if (!response.ok) {
        window.alert(result.error || 'Position close failed');
        return;
      }
      await fetchDashboard();
    } catch (error) {
      window.alert(`Position close failed: ${error.message}`);
    } finally {
      button.disabled = false;
    }
  });

  async function refreshAll() {
    const button = document.getElementById('refreshBtn');
    button.disabled = true;
    button.textContent = 'LOADING...';
    document.getElementById('chartNotice').textContent = `Refreshing ${selectedSymbol} market data...`;
    try {
      await Promise.all([
        fetchDashboard(),
        fetchMarket(selectedSymbol, selectedPeriod),
        refreshLiveQuote(),
        refreshWatchlistQuotes(),
        fetchNewsReviews(selectedSymbol),
        fetchSystemStatus()
      ]);
    } finally {
      button.disabled = false;
      button.textContent = 'RELOAD';
    }
  }

  // ── Alpaca Paper Trading Keys Modal ──
  function openAlpacaModal() {
    const modal = document.getElementById('alpacaModal');
    if (!modal) return;
    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
    const statusMsg = document.getElementById('alpacaStatusMsg');
    if (statusMsg) {
      statusMsg.textContent = 'Checking current broker status...';
      statusMsg.style.color = '#849197';
    }
    fetch(apiUrl('/api/broker/status'))
      .then((res) => res.json())
      .then((data) => {
        const alpaca = data && data.alpaca;
        const box = document.getElementById('alpacaAccountBox');
        if (alpaca && alpaca.configured) {
          if (box) box.classList.remove('hidden');
          const accStatus = document.getElementById('alpacaAccStatus');
          if (accStatus) accStatus.textContent = (alpaca.status || 'ACTIVE').toUpperCase();
          const accNum = document.getElementById('alpacaAccNumber');
          if (accNum) accNum.textContent = alpaca.accountNumber || '--';
          const bp = document.getElementById('alpacaAccBuyingPower');
          if (bp) bp.textContent = `$${Number(alpaca.buyingPower || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
          const pv = document.getElementById('alpacaAccPortfolioValue');
          if (pv) pv.textContent = `$${Number(alpaca.portfolioValue || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
          const cash = document.getElementById('alpacaAccCash');
          if (cash) cash.textContent = `$${Number(alpaca.cash || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
          if (statusMsg) {
            statusMsg.textContent = 'Connected to Alpaca Paper Trading.';
            statusMsg.style.color = '#42d17b';
          }
        } else {
          if (box) box.classList.add('hidden');
          if (statusMsg) {
            statusMsg.textContent = 'Enter keys below to connect Alpaca paper trading.';
            statusMsg.style.color = '#849197';
          }
        }
      })
      .catch(() => {
        if (statusMsg) statusMsg.textContent = '';
      });
  }

  function closeAlpacaModal() {
    const modal = document.getElementById('alpacaModal');
    if (!modal) return;
    modal.classList.add('hidden');
    modal.setAttribute('aria-hidden', 'true');
  }

  const alpacaConfigBtn = document.getElementById('alpacaConfigBtn');
  if (alpacaConfigBtn) alpacaConfigBtn.addEventListener('click', openAlpacaModal);
  const brokerMonitorItem = document.getElementById('brokerMonitorItem');
  if (brokerMonitorItem) brokerMonitorItem.addEventListener('click', openAlpacaModal);
  const closeAlpacaBtn = document.getElementById('closeAlpacaModal');
  if (closeAlpacaBtn) closeAlpacaBtn.addEventListener('click', closeAlpacaModal);
  const cancelAlpacaBtn = document.getElementById('cancelAlpacaModal');
  if (cancelAlpacaBtn) cancelAlpacaBtn.addEventListener('click', closeAlpacaModal);

  const saveAlpacaKeysBtn = document.getElementById('saveAlpacaKeysBtn');
  if (saveAlpacaKeysBtn) {
    saveAlpacaKeysBtn.addEventListener('click', async () => {
      const apiKey = document.getElementById('alpacaApiKeyInput').value.trim();
      const secretKey = document.getElementById('alpacaSecretKeyInput').value.trim();
      const statusMsg = document.getElementById('alpacaStatusMsg');
      if (!apiKey || !secretKey) {
        if (statusMsg) {
          statusMsg.textContent = 'Please enter both API Key ID and Secret Key.';
          statusMsg.style.color = '#ff4d4f';
        }
        return;
      }
      saveAlpacaKeysBtn.disabled = true;
      saveAlpacaKeysBtn.textContent = 'VERIFYING...';
      if (statusMsg) {
        statusMsg.textContent = 'Connecting to paper-api.alpaca.markets...';
        statusMsg.style.color = '#f0a400';
      }
      try {
        const resp = await fetch(apiUrl('/api/alpaca/config'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ apiKey, secretKey })
        });
        const result = await resp.json();
        if (!resp.ok || !result.ok) {
          throw new Error(result.error || `HTTP ${resp.status}`);
        }
        if (statusMsg) {
          statusMsg.textContent = 'Connected successfully! Alpaca paper trading is now ready.';
          statusMsg.style.color = '#42d17b';
        }
        saveAlpacaKeysBtn.textContent = 'CONNECTED';

        const acc = result.account || {};
        const box = document.getElementById('alpacaAccountBox');
        if (box) box.classList.remove('hidden');
        const accStatus = document.getElementById('alpacaAccStatus');
        if (accStatus) accStatus.textContent = (acc.status || 'ACTIVE').toUpperCase();
        const accNum = document.getElementById('alpacaAccNumber');
        if (accNum) accNum.textContent = acc.account_number || '--';
        const bp = document.getElementById('alpacaAccBuyingPower');
        if (bp) bp.textContent = `$${Number(acc.buying_power || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        const pv = document.getElementById('alpacaAccPortfolioValue');
        if (pv) pv.textContent = `$${Number(acc.portfolio_value || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        const cash = document.getElementById('alpacaAccCash');
        if (cash) cash.textContent = `$${Number(acc.cash || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

        document.getElementById('alpacaSecretKeyInput').value = '';

        await fetchSystemStatus();
        setTimeout(() => {
          closeAlpacaModal();
          saveAlpacaKeysBtn.disabled = false;
          saveAlpacaKeysBtn.textContent = 'CONNECT & SAVE';
        }, 1200);
      } catch (err) {
        if (statusMsg) {
          statusMsg.textContent = `Error: ${err.message}`;
          statusMsg.style.color = '#ff4d4f';
        }
        saveAlpacaKeysBtn.disabled = false;
        saveAlpacaKeysBtn.textContent = 'CONNECT & SAVE';
      }
    });
  }

  window.addEventListener('resize', drawCandles);
});