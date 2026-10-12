from pathlib import Path

import pytest
from tyche.execution.paper import PaperAutopilot


class FakeAlpaca:
    configured = True

    def __init__(self, response=None, error=None, order_response=None, positions=None):
        self.response = response or {"id": "close-123", "status": "filled"}
        self.error = error
        self.order_response = order_response or self.response
        self.positions = positions or []
        self.orders = []

    def submit_market_order(self, symbol, side, quantity, *, confirm):
        self.orders.append((symbol, side, quantity, confirm))
        if self.error:
            raise self.error
        response = dict(self.response)
        if len(self.orders) > 1:
            response["id"] = f"{response['id']}-{len(self.orders)}"
        return response

    def get_order(self, _order_id):
        return self.order_response

    def get_positions(self):
        return self.positions


def _open_alpaca_order():
    return {
        "id": "entry-123",
        "status": "Filled",
        "symbol": "AAPL",
        "side": "Buy",
        "qty": 2.0,
        "price": 100.0,
        "stopLoss": 99.0,
        "takeProfit": 103.0,
        "alpacaOrderId": "alpaca-entry-123",
        "broker": "Alpaca Paper",
    }


def _autopilot(tmp_path: Path, orders, broker):
    return PaperAutopilot(
        tmp_path,
        orders,
        lambda: None,
        lambda _symbol: 98.0,
        broker_adapter=broker,
    )


def test_stop_loss_submits_alpaca_close_order(tmp_path):
    entry = _open_alpaca_order()
    orders = [entry]
    broker = FakeAlpaca()
    autopilot = _autopilot(tmp_path, orders, broker)

    autopilot._enforce_stop_losses()

    assert broker.orders == [("AAPL", "Sell", 2.0, True)]
    close_order = next(order for order in orders if order.get("alpacaOrderId") == "close-123")
    assert close_order["status"] == "Filled"
    assert close_order["mode"] == "risk-stop"
    assert entry["stopTriggered"] is True


def test_failed_alpaca_close_is_not_recorded_as_filled(tmp_path):
    entry = _open_alpaca_order()
    orders = [entry]
    broker = FakeAlpaca(error=RuntimeError("Alpaca close rejected"))
    autopilot = _autopilot(tmp_path, orders, broker)

    with pytest.raises(RuntimeError, match="Alpaca close rejected"):
        autopilot._enforce_stop_losses()

    assert len(orders) == 1
    assert "stopTriggered" not in entry


def test_live_sentiment_is_used_instead_of_zero_score_file(tmp_path, monkeypatch):
    import pandas as pd

    stale_path = tmp_path / "data" / "output" / "news_sentiment_local.parquet"
    stale_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"ticker": ["AAPL"], "sentiment_final": [0.0]}
    ).to_parquet(stale_path, index=False)

    bot = PaperAutopilot(
        tmp_path,
        [],
        lambda: None,
        lambda _symbol: 100.0,
        sentiment_feed=lambda: [
            {"ticker": "AAPL", "sentiment_final": 0.1234},
            {"ticker": "MSFT", "sentiment_final": -0.0678},
        ],
        portfolio_feed=list,
    )
    submitted = []
    monkeypatch.setattr(
        bot, "_trade_candidate", lambda ticker, score: submitted.append((ticker, score))
    )

    bot._evaluate_once()

    assert bot.last_signal_source == "live-sentiment"
    assert submitted == [("AAPL", 0.1234), ("MSFT", -0.0678)]


def test_accepted_alpaca_entry_is_not_marked_filled(tmp_path):
    broker = FakeAlpaca({"id": "entry-pending", "status": "accepted"})
    bot = _autopilot(tmp_path, [], broker)

    bot._trade_candidate("AAPL", 0.2)

    entry = bot.orders[0]
    assert entry["alpacaOrderId"] == "entry-pending"
    assert entry["status"] == "Submitted"
    assert entry["alpacaStatus"] == "accepted"


def test_alpaca_submission_error_does_not_create_simulated_fill(tmp_path):
    broker = FakeAlpaca(error=RuntimeError("paper account rejected order"))
    bot = _autopilot(tmp_path, [], broker)

    result = bot._trade_candidate("AAPL", 0.2)

    assert "paper account rejected order" in result
    assert bot.orders == []


def test_sync_corrects_legacy_local_fill_from_alpaca_status(tmp_path):
    entry = _open_alpaca_order()
    orders = [entry]
    broker = FakeAlpaca(order_response={"id": "alpaca-entry-123", "status": "accepted"})
    bot = _autopilot(tmp_path, orders, broker)

    bot._sync_broker_orders()

    assert entry["status"] == "Submitted"
    assert entry["alpacaStatus"] == "accepted"
    assert entry["alpacaStatusVerified"] is True


def test_autopilot_opens_multiple_distinct_positions_in_one_cycle(tmp_path):
    symbols = [f"TICK{i}" for i in range(25)]
    broker = FakeAlpaca({"id": "entry", "status": "accepted"})
    bot = PaperAutopilot(
        tmp_path,
        [],
        lambda: None,
        lambda _symbol: 100.0,
        portfolio_feed=lambda: [{"ticker": symbol, "signal": 0.2} for symbol in symbols],
        broker_adapter=broker,
    )

    bot._evaluate_once()

    assert len(broker.orders) == 20
    assert len({order[0] for order in broker.orders}) == 20
    assert bot.status()["maxOpenPositions"] == 20
    assert bot.status()["intervalSeconds"] == 30


def test_entry_notional_is_capped_at_500_dollars(tmp_path):
    broker = FakeAlpaca({"id": "entry", "status": "accepted"})
    bot = _autopilot(tmp_path, [], broker)

    bot._trade_candidate("AAPL", 0.2)

    assert broker.orders[0][:2] == ("AAPL", "Buy")
    assert broker.orders[0][2] * 98.0 <= 500.0
    assert broker.orders[0][2] == 5.102


def test_open_broker_position_is_not_pyramided(tmp_path):
    broker = FakeAlpaca(
        {"id": "entry", "status": "accepted"},
        positions=[{"symbol": "AAPL", "qty": "2"}],
    )
    bot = _autopilot(tmp_path, [], broker)

    result = bot._trade_candidate("AAPL", 0.2)

    assert result is None
    assert broker.orders == []


def test_position_limit_blocks_entries_when_twenty_symbols_are_open(tmp_path):
    positions = [{"symbol": f"OPEN{i}", "qty": "1"} for i in range(20)]
    broker = FakeAlpaca({"id": "entry", "status": "accepted"}, positions=positions)
    bot = _autopilot(tmp_path, [], broker)

    result = bot._trade_candidate("NEW", 0.2)

    assert result is None
    assert broker.orders == []


def test_pending_entry_is_not_duplicated_even_at_a_new_price(tmp_path):
    pending = {
        "id": "local-1",
        "status": "Submitted",
        "symbol": "AAPL",
        "side": "Buy",
        "qty": 2.0,
        "price": 100.0,
        "mode": "autonomous-paper",
        "alpacaOrderId": "alpaca-1",
    }
    broker = FakeAlpaca({"id": "entry", "status": "accepted"})
    bot = _autopilot(tmp_path, [pending], broker)

    result = bot._trade_candidate("AAPL", 0.2)

    assert result is None
    assert broker.orders == []


def test_interval_has_thirty_second_minimum(tmp_path, monkeypatch):
    monkeypatch.setenv("TYCHE_PAPER_INTERVAL_SECONDS", "1")
    bot = _autopilot(tmp_path, [], None)

    assert bot.interval_seconds == 30
