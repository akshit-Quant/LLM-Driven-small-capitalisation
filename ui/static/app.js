let dashboardState = {
  accountBalance: 120000,
  buyingPower: 54000,
  totalValue: 142580,
  pnl: 12842,
  cash: 26000,
  riskScore: 72,
  sentiment: 0.74,
  trend: 3.42,
  portfolio: {
    status: 'unavailable',
    message: 'Portfolio backtest not run'
  },
  positions: [
    { symbol: 'NVDA', name: 'NVIDIA', quantity: 36, price: 132.6, change: 2.84 },
    { symbol: 'MSFT', name: 'Microsoft', quantity: 24, price: 418.1, change: 1.67 },
    { symbol: 'AMD', name: 'AMD', quantity: 52, price: 148.2, change: -0.78 },
    { symbol: 'AAPL', name: 'Apple', quantity: 18, price: 212.7, change: 0.92 }
  ],
  history: [
    { time: '09:45', side: 'Buy', symbol: 'NVDA', qty: 14, price: 130.8 },
    { time: '11:12', side: 'Sell', symbol: 'MSFT', qty: 8, price: 410.2 },
    { time: '13:05', side: 'Buy', symbol: 'AMD', qty: 22, price: 146.9 },
    { time: '15:42', side: 'Buy', symbol: 'AAPL', qty: 10, price: 210.5 }
  ],
  watchlist: [
    { symbol: 'NVDA', last: 132.60, change: 2.84, volume: 'Stock' },
    { symbol: 'MSFT', last: 418.10, change: 1.67, volume: 'Stock' },
    { symbol: 'AAPL', last: 212.70, change: 0.92, volume: 'Stock' },
    { symbol: 'AMD', last: 148.20, change: -0.78, volume: 'Stock' },
    { symbol: 'GC=F', label: 'Gold', last: 0, change: 0, volume: 'Commodity' },
    { symbol: 'CL=F', label: 'Crude Oil', last: 0, change: 0, volume: 'Commodity' },
    { symbol: 'SI=F', label: 'Silver', last: 0, change: 0, volume: 'Commodity' },
    { symbol: 'NG=F', label: 'Natural Gas', last: 0, change: 0, volume: 'Commodity' }
  ],
  heatmap: [
    ['buy', 'buy', 'neutral', 'sell', 'buy', 'buy'],
    ['buy', 'sell', 'buy', 'neutral', 'sell', 'buy'],
    ['neutral', 'buy', 'buy', 'sell', 'buy', 'neutral'],
    ['sell', 'neutral', 'buy', 'buy', 'sell', 'buy']
  ],
  newsReviews: [],
  candles: [
    { o: 1.7442, h: 1.7488, l: 1.7428, c: 1.7469 },
    { o: 1.7469, h: 1.7494, l: 1.7434, c: 1.7448 },
    { o: 1.7448, h: 1.7499, l: 1.7425, c: 1.7491 },
    { o: 1.7491, h: 1.7526, l: 1.7464, c: 1.7485 },
    { o: 1.7485, h: 1.7532, l: 1.7463, c: 1.7523 },
    { o: 1.7523, h: 1.7559, l: 1.7498, c: 1.7512 },
    { o: 1.7512, h: 1.7568, l: 1.7487, c: 1.7557 },
    { o: 1.7557, h: 1.7594, l: 1.7532, c: 1.7578 },
    { o: 1.7578, h: 1.7603, l: 1.7552, c: 1.7581 },
    { o: 1.7581, h: 1.7612, l: 1.7561, c: 1.7607 },
    { o: 1.7607, h: 1.7652, l: 1.7586, c: 1.7638 },
    { o: 1.7638, h: 1.7705, l: 1.7601, c: 1.7689 },
    { o: 1.7689, h: 1.7712, l: 1.7657, c: 1.7661 },
    { o: 1.7661, h: 1.7708, l: 1.7642, c: 1.7681 },
    { o: 1.7681, h: 1.7738, l: 1.7670, c: 1.7719 },
    { o: 1.7719, h: 1.7762, l: 1.7699, c: 1.7734 },
    { o: 1.7734, h: 1.7780, l: 1.7715, c: 1.7756 },
    { o: 1.7756, h: 1.7802, l: 1.7729, c: 1.7794 },
    { o: 1.7794, h: 1.7828, l: 1.7760, c: 1.7783 },
    { o: 1.7783, h: 1.7820, l: 1.7762, c: 1.7801 },
    { o: 1.7801, h: 1.7854, l: 1.7790, c: 1.7837 },
    { o: 1.7837, h: 1.7875, l: 1.7815, c: 1.7861 },
    { o: 1.7861, h: 1.7918, l: 1.7847, c: 1.7899 },
    { o: 1.7899, h: 1.7946, l: 1.7871, c: 1.7922 },
    { o: 1.7922, h: 1.7964, l: 1.7909, c: 1.7941 },
    { o: 1.7941, h: 1.7992, l: 1.7920, c: 1.7973 }
  ]
};
let selectedSymbol = 'AAPL';
let selectedPeriod = 'D1';
let chartOptions = { type: 'candles', grid: true, ma: true, crosshair: true };
let liveQuoteInFlight = false;
let watchlistQuotesInFlight = false;
let dashboardInFlight = false;

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
  document.getElementById('riskScore').textContent = `${dashboardState.riskScore}%`;
  document.getElementById('riskMeter').style.width = `${dashboardState.riskScore}%`;
  document.getElementById('sentimentValue').textContent = dashboardState.sentiment.toFixed(2);
  document.getElementById('totalTrend').textContent = pct(dashboardState.trend);
  document.getElementById('pnlTrend').textContent = pct(dashboardState.trend * 1.5);
  document.getElementById('sentimentTrend').textContent = dashboardState.sentiment >= 0.65 ? 'Bullish' : 'Neutral';
  const metrics = dashboardState.riskMetrics || {};
  const sharpe = Number(metrics.sharpe);
  const drawdown = Number(metrics.drawdown);
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
    const [healthResp, executionResp, brokerResp, portfolioResp] = await Promise.allSettled([
      fetch('/health', { cache: 'no-store' }),
      fetch('/api/execution/status', { cache: 'no-store' }),
      fetch('/api/broker/status', { cache: 'no-store' }),
      fetch('/api/portfolio?holding=5', { cache: 'no-store' })
    ]);

    const health = healthResp.status === 'fulfilled' && healthResp.value.ok ? await healthResp.value.json().catch(() => ({})) : {};
    const execution = executionResp.status === 'fulfilled' && executionResp.value.ok ? await executionResp.value.json().catch(() => ({})) : {};
    const broker = brokerResp.status === 'fulfilled' && brokerResp.value.ok ? await brokerResp.value.json().catch(() => ({})) : {};
    const portfolio = portfolioResp.status === 'fulfilled' && portfolioResp.value.ok ? await portfolioResp.value.json().catch(() => ({})) : {};

    const dataHealthy = Array.isArray(dashboardState.newsReviews) && dashboardState.newsReviews.length > 0;
    setHealthStatus(
      'dataHealthStatus',
      dataHealthy ? 'Healthy' : 'Degraded',
      dataHealthy ? 'good' : 'warning',
      dataHealthy ? 'pricing and sentiment online' : 'waiting for market/news data'
    );

    const portfolioReady = portfolio && portfolio.status === 'ready';
    setHealthStatus(
      'portfolioHealthStatus',
      portfolioReady ? 'Ready' : 'Pending',
      portfolioReady ? 'good' : 'warning',
      portfolioReady ? `H${portfolio.holding || 5} backtest loaded` : 'run /run-portfolio to generate metrics'
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

    const brokerState = broker && broker.alpaca && broker.alpaca.configured ? 'Ready' : 'Idle';
    const brokerTone = broker && broker.alpaca && broker.alpaca.configured ? 'good' : 'neutral';
    const brokerText = broker && broker.alpaca && broker.alpaca.configured ? 'Alpaca paper enabled' : 'Alpaca + IBKR disconnected';
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
        </div>
      `;
    })
    .join('');
}

function renderHistory() {
  const historyList = document.getElementById('historyList');
  historyList.innerHTML = dashboardState.history
    .map((trade) => `
      <div class="history-row">
        <div>
          <strong>${trade.symbol}</strong>
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
  watchlist.innerHTML = dashboardState.watchlist
    .map((item) => `
      <button class="watch-row stock-button" data-symbol="${item.symbol}">
        <div>
          <strong>${item.label || item.symbol}</strong>
          <span>${item.volume}</span>
        </div>
        <div class="position-value">
          <strong>${item.last.toFixed(2)}</strong>
          <span>Last</span>
        </div>
        <div class="side ${item.change >= 0 ? 'buy' : 'sell'}">${pct(item.change)}</div>
        <div class="position-value">
          <strong>${item.change >= 0 ? 'Bullish' : 'Bearish'}</strong>
          <span>Bias</span>
        </div>
      </button>
    `)
    .join('');
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
    container.innerHTML = '<div class="empty-state">No Qwen-enriched news found. Run the news pipeline with Qwen enrichment enabled.</div>';
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
    const response = await fetch(`/api/market?symbol=${encodeURIComponent(selectedSymbol)}&period=${period}&ts=${Date.now()}`, { cache: 'no-store' });
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

async function fetchNewsReviews(symbol = selectedSymbol) {
  try {
    const response = await fetch(`/api/dashboard?symbol=${encodeURIComponent(symbol)}&ts=${Date.now()}`, { cache: 'no-store' });
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
    const response = await fetch(`/api/market/live?symbol=${encodeURIComponent(selectedSymbol)}&ts=${Date.now()}`, { cache: 'no-store' });
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
    await Promise.all(dashboardState.watchlist.map(async (item) => {
      try {
        const response = await fetch(`/api/market/live?symbol=${encodeURIComponent(item.symbol)}`);
        if (!response.ok) return;
        const quote = await response.json();
        item.last = quote.last;
        item.change = quote.change;
      } catch (error) {
        console.warn(`Quote unavailable for ${item.symbol}:`, error);
      }
    }));
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
      if (button) fetchMarket(button.dataset.symbol, selectedPeriod);
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
      const response = await fetch('/api/orders', {
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
    const response = await fetch(`/api/dashboard?symbol=${encodeURIComponent(selectedSymbol)}&ts=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    dashboardState = { ...dashboardState, ...payload };
    if (typeof payload.selectedSentiment === 'number') {
      dashboardState.sentiment = payload.selectedSentiment;
    }
    const ordersResponse = await fetch(`/api/orders?ts=${Date.now()}`, { cache: 'no-store' });
    if (ordersResponse.ok) {
      const orderPayload = await ordersResponse.json();
      dashboardState.history = [...(orderPayload.orders || []), ...(dashboardState.history || [])]
        .filter((item, index, items) => !item.id || items.findIndex((candidate) => candidate.id === item.id) === index);
    }
  } catch (error) {
    console.warn('Dashboard API unavailable, using local demo state:', error);
  } finally {
    dashboardInFlight = false;
  }
  refreshDashboard();
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
  fetchSystemStatus();
  fetchDashboard();
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
  document.getElementById('historyList').addEventListener('click', async (event) => {
    const button = event.target.closest('.cancel-order');
    if (!button) return;
    const response = await fetch(`/api/orders/${button.dataset.orderId}/cancel`, { method: 'POST' });
    const result = await response.json();
    if (!response.ok) {
      window.alert(result.error || 'Order cancellation failed');
      return;
    }
    await fetchDashboard();
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

  window.addEventListener('resize', drawCandles);
});