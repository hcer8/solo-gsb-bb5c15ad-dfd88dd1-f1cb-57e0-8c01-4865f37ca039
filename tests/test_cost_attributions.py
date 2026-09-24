"""End-to-end tests for the read-only cost-attribution endpoint."""
from __future__ import annotations

import itertools
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-attribution-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "attribution.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, rdp  # noqa: E402
from app.main import app  # noqa: E402

_counter = itertools.count()


def _new_dataset_id(name: str) -> str:
    return f"dsca-{name}-{next(_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


async def _grant(client, dataset_id: str, request_id: str,
                 epsilon: float, delta: float = 1e-7,
                 query_type: str = "q"):
    r = await client.post(
        f"/datasets/{dataset_id}/requests",
        json={"request_id": request_id, "query_type": query_type,
              "epsilon": epsilon, "delta": delta},
    )
    assert r.status_code == 201, r.text
    assert r.json()["allowed"] is True
    return r.json()


async def _attribute(client, dataset_id: str, request_ids: list[str]):
    return await client.post(
        f"/datasets/{dataset_id}/cost-attributions",
        json={"request_ids": request_ids},
    )


async def _backdate(dataset_id: str, reservation_id: str,
                    expires_at: datetime) -> None:
    async with db.connect() as conn:
        await db.backdate_reservation_expiry(
            conn,
            dataset_id=dataset_id,
            reservation_id=reservation_id,
            expires_at=expires_at.isoformat(),
        )
        await conn.commit()


def _brute_force(
    baseline: list[tuple[float, float]],
    selected: list[tuple[float, float]],
    target_delta: float,
) -> tuple[float, float, list[float]]:
    """Independent Shapley computation built straight on rdp.evaluate_entries."""
    n = len(selected)

    def value(mask: int) -> float:
        entries = list(baseline)
        for i in range(n):
            if mask & (1 << i):
                entries.append(selected[i])
        return max(0.0, rdp.evaluate_entries(entries, target_delta)[1])

    base, full = value(0), value((1 << n) - 1)
    factorials = [math.factorial(k) for k in range(n + 1)]
    shares = [0.0] * n
    for i in range(n):
        bit = 1 << i
        others = ((1 << n) - 1) ^ bit
        coalition = others
        while True:
            size = coalition.bit_count()
            weight = factorials[size] * factorials[n - size - 1] / factorials[n]
            shares[i] += weight * (value(coalition | bit) - value(coalition))
            if coalition == 0:
                break
            coalition = (coalition - 1) & others
    return base, full, shares


# ----------------------------------------------------------------- behaviour --


@pytest.mark.asyncio
async def test_single_request_attribution_is_its_increment(client):
    ds = _new_dataset_id("single")
    await _register(client, ds, eps=1.0, delta=1e-6)
    # Pre-existing occupancy so attribution is measured against a non-empty
    # baseline.
    await _grant(client, ds, "base", 0.1, 1e-8)
    mine = await _grant(client, ds, "mine", 0.2, 1e-8)

    r = await _attribute(client, ds, ["mine"])
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == ds
    assert body["request_count"] == 1
    assert body["request_ids"] == ["mine"]
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["request_id"] == "mine"
    assert item["epsilon"] == 0.2
    assert item["delta"] == 1e-8

    # For one request the Shapley share is simply projected - baseline.
    assert body["epsilon_projected"] == pytest.approx(mine["epsilon_after"],
                                                      abs=1e-12)
    assert body["epsilon_baseline"] == pytest.approx(
        body["epsilon_projected"] - item["epsilon_attributed"], abs=1e-12
    )
    assert body["epsilon_increment"] == pytest.approx(
        item["epsilon_attributed"], abs=1e-12
    )
    assert body["epsilon_attributed_total"] == pytest.approx(
        item["epsilon_attributed"], abs=1e-12
    )


@pytest.mark.asyncio
async def test_pure_dp_group_shares_are_exact_individual_epsilons(client):
    ds = _new_dataset_id("pure")
    await _register(client, ds, eps=2.0, delta=1e-6)
    await _grant(client, ds, "a", 0.1, 0.0)
    await _grant(client, ds, "b", 0.2, 0.0)
    await _grant(client, ds, "c", 0.3, 0.0)

    body = (await _attribute(client, ds, ["a", "b", "c"])).json()
    shares = {item["request_id"]: item["epsilon_attributed"]
              for item in body["items"]}
    # Pure epsilon-DP composes additively: each request costs exactly its own
    # epsilon regardless of coalition, so Shapley shares are exact.
    assert shares["a"] == pytest.approx(0.1, abs=1e-12)
    assert shares["b"] == pytest.approx(0.2, abs=1e-12)
    assert shares["c"] == pytest.approx(0.3, abs=1e-12)
    assert body["epsilon_baseline"] == pytest.approx(0.0, abs=1e-12)
    assert body["epsilon_projected"] == pytest.approx(0.6, abs=1e-12)
    assert body["epsilon_increment"] == pytest.approx(0.6, abs=1e-12)
    assert body["epsilon_attributed_total"] == pytest.approx(0.6, abs=1e-12)


@pytest.mark.asyncio
async def test_baseline_excludes_selected_but_keeps_other_requests(client):
    ds = _new_dataset_id("baseline")
    await _register(client, ds, eps=2.0, delta=1e-6)
    # All pure-DP, so occupancy is a plain epsilon sum.
    await _grant(client, ds, "other1", 0.07, 0.0)
    await _grant(client, ds, "sel1", 0.1, 0.0)
    await _grant(client, ds, "sel2", 0.2, 0.0)
    await _grant(client, ds, "other2", 0.03, 0.0)

    body = (await _attribute(client, ds, ["sel2", "sel1"])).json()
    assert body["request_ids"] == ["sel2", "sel1"]
    # Baseline keeps other1 + other2; neither selected request is in it.
    assert body["epsilon_baseline"] == pytest.approx(0.10, abs=1e-12)
    assert body["epsilon_projected"] == pytest.approx(0.40, abs=1e-12)
    assert body["epsilon_increment"] == pytest.approx(0.30, abs=1e-12)
    shares = {item["request_id"]: item["epsilon_attributed"]
              for item in body["items"]}
    assert shares["sel1"] == pytest.approx(0.1, abs=1e-12)
    assert shares["sel2"] == pytest.approx(0.2, abs=1e-12)
    assert body["epsilon_attributed_total"] == pytest.approx(
        body["epsilon_increment"], abs=1e-12
    )


@pytest.mark.asyncio
async def test_identical_requests_get_equal_shapley_shares(client):
    ds = _new_dataset_id("symmetry")
    await _register(client, ds, eps=2.0, delta=1e-6)
    await _grant(client, ds, "base", 0.1, 1e-9)
    await _grant(client, ds, "x", 0.15, 1e-8)
    await _grant(client, ds, "y", 0.15, 1e-8)

    body = (await _attribute(client, ds, ["x", "y"])).json()
    shares = {item["request_id"]: item["epsilon_attributed"]
              for item in body["items"]}
    # Shapley symmetry: identical mechanisms get identical shares.
    assert shares["x"] == pytest.approx(shares["y"], abs=1e-12)
    assert body["epsilon_attributed_total"] == pytest.approx(
        body["epsilon_increment"], abs=1e-12
    )


@pytest.mark.asyncio
async def test_share_brute_force_matches_independent_accounting(client):
    ds = _new_dataset_id("brute")
    target_delta = 1e-6
    await _register(client, ds, eps=5.0, delta=target_delta)
    # Baseline occupancy mixing pure and approximate DP.
    await _grant(client, ds, "base1", 0.1, 1e-9)
    await _grant(client, ds, "base2", 0.05, 0.0)
    await _grant(client, ds, "s1", 0.2, 1e-8)
    await _grant(client, ds, "s2", 0.15, 1e-7)
    await _grant(client, ds, "s3", 0.1, 0.0)
    await _grant(client, ds, "s4", 0.3, 1e-10)

    body = (await _attribute(
        client, ds, ["s3", "s1", "s4", "s2"]
    )).json()

    baseline = [(0.1, 1e-9), (0.05, 0.0)]
    selected = [(0.1, 0.0), (0.2, 1e-8), (0.3, 1e-10), (0.15, 1e-7)]
    exp_base, exp_full, exp_shares = _brute_force(
        baseline, selected, target_delta
    )
    assert body["epsilon_baseline"] == pytest.approx(exp_base, abs=1e-12)
    assert body["epsilon_projected"] == pytest.approx(exp_full, abs=1e-12)
    assert body["epsilon_increment"] == pytest.approx(
        exp_full - exp_base, abs=1e-12
    )
    by_id = {item["request_id"]: item["epsilon_attributed"]
             for item in body["items"]}
    # Items come back in request order: s3, s1, s4, s2.
    assert [item["request_id"] for item in body["items"]] == [
        "s3", "s1", "s4", "s2"
    ]
    assert by_id["s3"] == pytest.approx(exp_shares[0], abs=1e-12)
    assert by_id["s1"] == pytest.approx(exp_shares[1], abs=1e-12)
    assert by_id["s4"] == pytest.approx(exp_shares[2], abs=1e-12)
    assert by_id["s2"] == pytest.approx(exp_shares[3], abs=1e-12)
    assert body["epsilon_attributed_total"] == pytest.approx(
        exp_full - exp_base, abs=1e-10
    )


@pytest.mark.asyncio
async def test_attribution_counts_active_reservation_in_baseline(client):
    ds = _new_dataset_id("reserved")
    await _register(client, ds, eps=2.0, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "res1", "epsilon": 0.4, "delta": 1e-8,
              "ttl_seconds": 3600},
    )
    await _grant(client, ds, "a", 0.1, 1e-8)
    await _grant(client, ds, "b", 0.2, 1e-8)

    body = (await _attribute(client, ds, ["a", "b"])).json()

    # Baseline independently = reservation alone; full group adds a and b.
    exp_base, exp_full, exp_shares = _brute_force(
        [(0.4, 1e-8)], [(0.1, 1e-8), (0.2, 1e-8)], 1e-6
    )
    assert body["epsilon_baseline"] == pytest.approx(exp_base, abs=1e-12)
    assert body["epsilon_projected"] == pytest.approx(exp_full, abs=1e-12)
    assert [i["epsilon_attributed"] for i in body["items"]] == pytest.approx(
        exp_shares, abs=1e-12
    )

    # It also matches the budget view's committed occupancy.
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    committed = budget["epsilon_consumed"] + budget["epsilon_reserved"]
    assert body["epsilon_projected"] == pytest.approx(committed, abs=1e-12)


@pytest.mark.asyncio
async def test_expired_reservation_and_reversed_request_excluded(client):
    ds = _new_dataset_id("excluded")
    await _register(client, ds, eps=2.0, delta=1e-6)
    # Expired reservation (backdated, never swept by this read-only path).
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "exp", "epsilon": 0.4, "delta": 0.0,
              "ttl_seconds": 3600},
    )
    await _backdate(
        ds, "exp", datetime.now(timezone.utc) - timedelta(seconds=5)
    )
    # Reversed granted request.
    await _grant(client, ds, "gone", 0.3, 0.0)
    rev = await client.post(
        f"/datasets/{ds}/requests/gone/reversals",
        json={"reversal_id": "rev-excluded-cleanup", "reason": "cleanup"},
    )
    assert rev.status_code == 201
    await _grant(client, ds, "keep", 0.1, 0.0)

    body = (await _attribute(client, ds, ["keep"])).json()
    assert body["epsilon_baseline"] == pytest.approx(0.0, abs=1e-12)
    assert body["epsilon_projected"] == pytest.approx(0.1, abs=1e-12)
    assert body["epsilon_increment"] == pytest.approx(0.1, abs=1e-12)


@pytest.mark.asyncio
async def test_ten_request_boundary(client):
    ds = _new_dataset_id("ten")
    await _register(client, ds, eps=10.0, delta=1e-6)
    ids = [f"r{i:02d}" for i in range(10)]
    for rid in ids:
        await _grant(client, ds, rid, 0.05, 1e-8)

    r = await _attribute(client, ds, ids)
    assert r.status_code == 200
    body = r.json()
    assert body["request_count"] == 10
    assert len(body["items"]) == 10
    assert body["epsilon_attributed_total"] == pytest.approx(
        body["epsilon_increment"], abs=1e-10
    )


@pytest.mark.asyncio
async def test_attribution_is_strictly_read_only(client):
    ds = _new_dataset_id("readonly")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await _grant(client, ds, "a", 0.1, 0.0)
    await _grant(client, ds, "b", 0.2, 0.0)
    # Create the reservation after the granted requests so creating it does
    # not sweep anything itself; then backdate it past the TTL.
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "exp", "epsilon": 0.3, "delta": 0.0,
              "ttl_seconds": 3600},
    )
    await _backdate(
        ds, "exp", datetime.now(timezone.utc) - timedelta(seconds=5)
    )

    first = (await _attribute(client, ds, ["a", "b"])).json()
    second = (await _attribute(client, ds, ["a", "b"])).json()
    for key in (
        "epsilon_baseline", "epsilon_projected", "epsilon_increment",
        "epsilon_attributed_total",
    ):
        assert first[key] == second[key]

    # No new records of any kind...
    assert (await client.get(f"/datasets/{ds}/requests")).json()["count"] == 2
    assert (
        await client.get(f"/datasets/{ds}/reversals")
    ).json()["count"] == 0
    # ...and the lazy expiry sweep must not have run: the past-TTL
    # reservation is still stored as granted with no expired_at.
    async with db.connect() as conn:
        row = await db.get_reservation(conn, ds, "exp")
        assert row["status"] == "granted"
        assert row["expired_at"] is None


@pytest.mark.asyncio
async def test_attribution_recomputes_against_live_ledger(client):
    ds = _new_dataset_id("live")
    await _register(client, ds, eps=2.0, delta=1e-6)
    await _grant(client, ds, "a", 0.1, 1e-8)
    await _grant(client, ds, "b", 0.2, 1e-8)

    before = (await _attribute(client, ds, ["a", "b"])).json()
    assert before["epsilon_baseline"] == pytest.approx(0.0, abs=1e-12)

    # A new granted request enlarges the baseline on the next call.
    await _grant(client, ds, "later", 0.15, 1e-8)
    after = (await _attribute(client, ds, ["a", "b"])).json()
    assert after["epsilon_baseline"] > before["epsilon_baseline"]
    exp_base = max(0.0, rdp.evaluate_entries([(0.15, 1e-8)], 1e-6)[1])
    assert after["epsilon_baseline"] == pytest.approx(exp_base, abs=1e-12)

    # Reversing a baseline request shrinks it again.
    await client.post(
        f"/datasets/{ds}/requests/later/reversals",
        json={"reversal_id": "rev-later", "reason": "redo"},
    )
    again = (await _attribute(client, ds, ["a", "b"])).json()
    assert again["epsilon_baseline"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.asyncio
async def test_shares_sum_to_the_two_occupancies_difference(client):
    ds = _new_dataset_id("sumidentity")
    await _register(client, ds, eps=3.0, delta=1e-6)
    await _grant(client, ds, "base", 0.2, 1e-9)
    for rid, eps, delta in [
        ("a", 0.1, 1e-8), ("b", 0.15, 0.0), ("c", 0.08, 1e-7),
        ("d", 0.12, 1e-8), ("e", 0.05, 0.0),
    ]:
        await _grant(client, ds, rid, eps, delta)

    body = (await _attribute(
        client, ds, ["a", "b", "c", "d", "e"]
    )).json()
    total = sum(item["epsilon_attributed"] for item in body["items"])
    assert total == pytest.approx(
        body["epsilon_projected"] - body["epsilon_baseline"], abs=1e-10
    )
    assert body["epsilon_attributed_total"] == pytest.approx(total, abs=1e-12)
    assert body["epsilon_increment"] == pytest.approx(
        body["epsilon_projected"] - body["epsilon_baseline"], abs=1e-12
    )


# ---------------------------------------------------------------- validation --


@pytest.mark.asyncio
async def test_missing_dataset_404(client):
    r = await _attribute(client, "no-such-dataset", ["a"])
    assert r.status_code == 404
    assert r.json()["error"] == "dataset_not_found"


@pytest.mark.asyncio
async def test_missing_request_404(client):
    ds = _new_dataset_id("missingreq")
    await _register(client, ds)
    r = await _attribute(client, ds, ["ghost"])
    assert r.status_code == 404
    assert r.json()["error"] == "request_not_found"
    assert r.json()["request_id"] == "ghost"


@pytest.mark.asyncio
async def test_one_missing_among_existing_still_404(client):
    ds = _new_dataset_id("onemissing")
    await _register(client, ds)
    await _grant(client, ds, "a", 0.1, 0.0)
    r = await _attribute(client, ds, ["a", "ghost"])
    assert r.status_code == 404
    assert r.json()["request_id"] == "ghost"


@pytest.mark.asyncio
async def test_rejected_request_conflicts_409(client):
    ds = _new_dataset_id("rejected")
    await _register(client, ds, eps=0.01, delta=1e-6)
    r = await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "big", "query_type": "q",
              "epsilon": 0.5, "delta": 1e-7},
    )
    assert r.json()["allowed"] is False

    conflict = await _attribute(client, ds, ["big"])
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "request_not_granted"


@pytest.mark.asyncio
async def test_reversed_request_conflicts_409(client):
    ds = _new_dataset_id("reversed")
    await _register(client, ds)
    await _grant(client, ds, "a", 0.1, 0.0)
    await client.post(
        f"/datasets/{ds}/requests/a/reversals",
        json={"reversal_id": "rev-reversed-conflict", "reason": "cleanup"},
    )
    conflict = await _attribute(client, ds, ["a"])
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "request_already_reversed"


@pytest.mark.asyncio
async def test_duplicate_request_ids_422(client):
    ds = _new_dataset_id("dup")
    await _register(client, ds)
    await _grant(client, ds, "a", 0.1, 0.0)
    r = await _attribute(client, ds, ["a", "a"])
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_empty_and_too_many_request_ids_422(client):
    ds = _new_dataset_id("bounds")
    await _register(client, ds)
    assert (await _attribute(client, ds, [])).status_code == 422
    ids = [f"r{i}" for i in range(11)]
    for rid in ids:
        await _grant(client, ds, rid, 0.001, 0.0)
    assert (await _attribute(client, ds, ids)).status_code == 422


@pytest.mark.asyncio
async def test_invalid_request_id_characters_422(client):
    ds = _new_dataset_id("badid")
    await _register(client, ds)
    r = await _attribute(client, ds, ["bad id!"])
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_missing_request_ids_field_422(client):
    ds = _new_dataset_id("nofield")
    await _register(client, ds)
    r = await client.post(
        f"/datasets/{ds}/cost-attributions", json={}
    )
    assert r.status_code == 422
