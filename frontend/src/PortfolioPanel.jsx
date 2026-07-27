import { useState, useCallback } from 'react';
import { Plus, X, TrendingUp } from 'lucide-react';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

function PortfolioPanel({ token, tickers, onTickersChange, onTickerClick }) {
  const [inputValue, setInputValue] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAdd = useCallback(async () => {
    const ticker = inputValue.trim().toUpperCase();
    if (!ticker) return;

    if (tickers.includes(ticker)) {
      setError(`${ticker} is already in your portfolio.`);
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_URL}/portfolio`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ ticker }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || 'Failed to add ticker.');
      }

      const data = await response.json();
      onTickersChange(data.tickers || [...tickers, ticker]);
      setInputValue('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [inputValue, token, tickers, onTickersChange]);

  const handleRemove = useCallback(
    async (ticker) => {
      try {
        const response = await fetch(`${API_URL}/portfolio/${encodeURIComponent(ticker)}`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${token}` },
        });

        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || 'Failed to remove ticker.');
        }

        const data = await response.json();
        onTickersChange(data.tickers || tickers.filter((t) => t !== ticker));
      } catch (err) {
        setError(err.message);
      }
    },
    [token, tickers, onTickersChange]
  );

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAdd();
    }
  };

  return (
    <div className="portfolio-panel">
      <h3 className="portfolio-title">
        <TrendingUp size={16} />
        My Portfolio
      </h3>

      <div className="portfolio-input-row">
        <input
          className="portfolio-input"
          type="text"
          placeholder="Add ticker (e.g. AAPL)"
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
            setError('');
          }}
          onKeyDown={handleKeyDown}
          maxLength={20}
          disabled={loading}
        />
        <button
          type="button"
          className="portfolio-add-btn"
          onClick={handleAdd}
          disabled={loading || !inputValue.trim()}
          title="Add ticker"
          aria-label="Add ticker"
        >
          <Plus size={16} />
        </button>
      </div>

      {error && <div className="portfolio-error">{error}</div>}

      {tickers.length > 0 ? (
        <div className="portfolio-chips">
          {tickers.map((t) => (
            <div key={t} className="ticker-chip">
              <button
                type="button"
                className="ticker-chip-label"
                onClick={() => onTickerClick?.(t)}
                title={`Analyze ${t}`}
              >
                {t}
              </button>
              <button
                type="button"
                className="ticker-chip-remove"
                onClick={() => handleRemove(t)}
                title={`Remove ${t}`}
                aria-label={`Remove ${t}`}
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p className="portfolio-empty">
          Add tickers to track your investments.
        </p>
      )}
    </div>
  );
}

export default PortfolioPanel;
