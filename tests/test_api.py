from litestar.testing import TestClient

from tradingview_strategy_risk_lab.api import app


def test_health() -> None:
    with TestClient(app=app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_explainability_stack() -> None:
    with TestClient(app=app) as client:
        payload = client.get("/v1/explainability-stack").json()

    assert "shap" in payload
    assert "dice" in payload

