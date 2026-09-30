"""健康检查测试。跑法：项目根目录执行 pytest。"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "llm-service"
    assert body["uptime_s"] >= 0


def test_health_has_request_id(client):
    """请求 ID 是可观测性的地基，缺了它排障只能靠猜。"""
    r = client.get("/health")
    assert r.headers.get("x-request-id")

    r2 = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert r2.headers.get("x-request-id") == "trace-abc-123"


def test_ready_structure(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ready", "degraded", "not_ready")
    assert len(body["providers"]) == 3
    assert {p["key"] for p in body["providers"]} == {"deepseek", "zhipu", "dashscope"}
    assert all(p["latency_s"] is None for p in body["providers"])
    assert body["probed"] is False


def test_ready_cost_block(client):
    r = client.get("/health/ready")
    cost = r.json()["cost"]
    assert cost["total_cny"] >= 0
    assert cost["remaining_cny"] <= cost["limit_cny"]


def test_openapi_available(client):
    """Pydantic 模型白送的 OpenAPI schema，是交付物的一部分。"""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/health" in r.json()["paths"]
