"""Tests for the FastAPI application shell."""

from fastapi.testclient import TestClient

from src.app.api_app import app


def test_health_endpoint_returns_ok() -> None:
    """The API should expose a lightweight health endpoint."""
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frontend_contract_routes_are_registered() -> None:
    """The React frontend expects these API routes to exist."""
    route_paths = {route.path for route in app.routes}

    assert "/api/process-pdf" in route_paths
    assert "/api/ask" in route_paths
    assert "/api/ask-visual" in route_paths
