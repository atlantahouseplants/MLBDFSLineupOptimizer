"""Tests for the official BallparkPal API ingestion module.

All HTTP is faked — no network, no API key needed. Payloads mirror the
openapi.json schemas.
"""
from __future__ import annotations

import pandas as pd
import pytest

from slate_optimizer.ingestion import bpp_official_api as api


# ── Fixtures: payloads matching the OpenAPI schemas ──────────────────────────

TEAM_MAP = {1: "NYY", 2: "BOS", 3: "LAD", 4: "SF"}

GAMES = [
    {
        "gameId": 100,
        "gameDate": "2026-07-17",
        "gameTime": "7:07",
        "teamAwayId": 1,
        "teamHomeId": 2,
        "venueId": 3,
    },
    {
        "gameId": 200,
        "gameDate": "2026-07-17",
        "gameTime": "9:38",
        "teamAwayId": 4,
        "teamHomeId": 3,
        "venueId": 5,
    },
]

AVERAGES = {
    100: {
        "batters": [
            {
                "playerId": 501,
                "playerName": "Aaron Judge",
                "teamId": 1,
                "team": "NYY",
                "battingPosition": 2,
                "plateAppearances": 4.3,
                "atBats": 3.8,
                "singles": 0.6,
                "doubles": 0.2,
                "triples": 0.01,
                "homeRuns": 0.31,
                "hits": 1.12,
                "totalBases": 2.2,
                "rbis": 0.9,
                "runs": 0.85,
                "walks": 0.5,
                "strikeouts": 1.1,
                "stolenBaseSuccesses": 0.02,
                "fantasyPointsDK": 10.1,
                "fantasyPointsFD": 12.6,
            },
            {
                "playerId": 502,
                "playerName": "Rafael Devers",
                "teamId": 2,
                "team": "BOS",
                "battingPosition": 3,
                "plateAppearances": 4.2,
                "atBats": 3.7,
                "singles": 0.7,
                "doubles": 0.25,
                "triples": 0.0,
                "homeRuns": 0.22,
                "hits": 1.0,
                "totalBases": 1.9,
                "rbis": 0.7,
                "runs": 0.7,
                "walks": 0.4,
                "strikeouts": 1.0,
                "stolenBaseSuccesses": 0.01,
                "fantasyPointsDK": 9.0,
                "fantasyPointsFD": 11.2,
            },
        ],
        "pitchers": [
            {
                "playerId": 601,
                "playerName": "Gerrit Cole",
                "teamId": 1,
                "team": "NYY",
                "isStarter": True,
                "innings": 5.8,
                "battersFaced": 22.5,
                "strikeouts": 6.5,
                "walks": 1.8,
                "hitsAllowed": 5.5,
                "runsAllowed": 2.4,
                "winProbability": 0.55,
                "lossProbability": 0.30,
                "qualityStartProbability": 0.62,
                "fantasyPointsDK": 21.0,
                "fantasyPointsFD": 35.5,
            },
            {
                "playerId": 602,
                "playerName": "Bullpen Guy",
                "teamId": 2,
                "team": "BOS",
                "isStarter": False,  # must be filtered out
                "innings": 1.0,
                "battersFaced": 4.0,
                "strikeouts": 1.0,
                "walks": 0.5,
                "hitsAllowed": 1.0,
                "runsAllowed": 0.5,
                "winProbability": 0.02,
                "lossProbability": 0.03,
                "qualityStartProbability": 0.0,
                "fantasyPointsDK": 2.0,
                "fantasyPointsFD": 3.0,
            },
        ],
        "teams": [
            {"teamId": 1, "team": "NYY", "runs": 4.6},
            {"teamId": 2, "team": "BOS", "runs": 3.9},
        ],
    },
    200: {"batters": [], "pitchers": [], "teams": [
        {"teamId": 4, "team": "SF", "runs": 3.2},
        {"teamId": 3, "team": "LAD", "runs": 4.9},
    ]},
}

PROBABILITIES = {
    100: [
        {
            "marketType": "moneyline",
            "marketKey": "ml",
            "displayName": "Moneyline",
            "line": None,
            "side": None,
            "odds": -130,
            "probability": 0.565,
            "average": None,
            "subject": {"type": "team", "id": 1},
            "teamId": 1,
        },
        {
            "marketType": "moneyline",
            "marketKey": "ml",
            "displayName": "Moneyline",
            "line": None,
            "side": None,
            "odds": 110,
            "probability": 0.435,
            "average": None,
            "subject": {"type": "team", "id": 2},
            "teamId": 2,
        },
        {
            "marketType": "batter_prop",
            "marketKey": "home_runs",
            "displayName": "Aaron Judge Home Runs",
            "line": 0.5,
            "side": "over",
            "odds": 250,
            "probability": 0.29,
            "average": 0.31,
            "subject": {"type": "batter", "id": 501},
            "teamId": 1,
        },
        {
            "marketType": "batter_prop",
            "marketKey": "hits",
            "displayName": "Aaron Judge Hits",
            "line": 0.5,
            "side": "over",
            "odds": -160,
            "probability": 0.68,
            "average": 1.12,
            "subject": {"type": "batter", "id": 501},
            "teamId": 1,
        },
        {
            # Under side — must NOT be used as a "to record" probability
            "marketType": "batter_prop",
            "marketKey": "stolen_bases",
            "displayName": "Aaron Judge Stolen Bases",
            "line": 0.5,
            "side": "under",
            "odds": -500,
            "probability": 0.98,
            "average": 0.02,
            "subject": {"type": "batter", "id": 501},
            "teamId": 1,
        },
    ],
    200: [],
}

MLB_HAND = pd.DataFrame(
    [
        {"player_name": "Aaron Judge", "mlb_team": "NYY", "bats": "R", "throws": "R"},
        {"player_name": "Rafael Devers", "mlb_team": "BOS", "bats": "L", "throws": "R"},
        {"player_name": "Gerrit Cole", "mlb_team": "NYY", "bats": "R", "throws": "R"},
    ]
)


# ── Unit tests: helpers ───────────────────────────────────────────────────────


class TestHelpers:
    def test_as_fraction_passthrough(self):
        assert api._as_fraction(0.42) == pytest.approx(0.42)

    def test_as_fraction_percentage(self):
        assert api._as_fraction(42.0) == pytest.approx(0.42)

    def test_as_fraction_garbage(self):
        assert api._as_fraction(None) != api._as_fraction(None)  # NaN

    def test_poisson_prob(self):
        assert api._poisson_prob(0.0) == 0.0
        assert api._poisson_prob(0.31) == pytest.approx(1 - 2.718281828 ** -0.31, rel=1e-3)

    def test_last_name_suffix(self):
        assert api._last_name("Cal Ripken Jr.") == "Ripken"
        assert api._last_name("Aaron Judge") == "Judge"

    def test_is_over05(self):
        assert api._is_over05({"line": 0.5, "side": "over"})
        assert not api._is_over05({"line": 0.5, "side": "under"})
        assert not api._is_over05({"line": 1.5, "side": "over"})
        assert api._is_over05({"line": None, "side": None})


class TestProbabilityParsing:
    def test_moneyline_extraction(self):
        probs = api.parse_win_probabilities(PROBABILITIES[100], TEAM_MAP)
        assert probs == {1: pytest.approx(0.565), 2: pytest.approx(0.435)}

    def test_batter_props_extraction(self):
        props = api.parse_batter_prop_probabilities(PROBABILITIES[100])
        assert props[501]["hr"] == pytest.approx(0.29)
        assert props[501]["hit"] == pytest.approx(0.68)
        # Under-side SB market must be ignored → no sb entry
        assert "sb" not in props[501]


# ── Frame builders ────────────────────────────────────────────────────────────


def _hand_lookup():
    return api._HandednessLookup(MLB_HAND)


class TestFrameBuilders:
    def test_batters_legacy_contract(self):
        df = api.build_batters_frame(
            GAMES, AVERAGES, TEAM_MAP, _hand_lookup(),
            api.parse_batter_prop_probabilities(PROBABILITIES[100]),
        )
        assert len(df) == 2
        judge = df[df["FullName"] == "Aaron Judge"].iloc[0]
        # Legacy column contract
        for col in [
            "GamePk", "GameDate", "GameTime", "PlayerId", "FullName", "LastName",
            "BatterStand", "Side", "Team", "Opponent", "BattingPosition",
            "PlateAppearances", "AtBats", "Hits", "Bases", "Strikeouts", "Walks",
            "Singles", "Doubles", "Triples", "HomeRuns", "RBIs", "Runs",
            "StolenBaseAttempts", "StolenBaseSuccesses", "PointsDK", "PointsFD",
            "HomeRunProbability", "HitProbability", "StolenBaseProbability",
        ]:
            assert col in df.columns, col
        assert judge["Team"] == "NYY"
        assert judge["Opponent"] == "BOS"
        assert judge["Side"] == "A"
        assert judge["BattingPosition"] == 2
        assert judge["PointsFD"] == pytest.approx(12.6)
        assert judge["Bases"] == pytest.approx(2.2)  # totalBases → Bases
        assert judge["BatterStand"] == "R"  # from MLB handedness
        # Market probabilities win over Poisson fallback
        assert judge["HomeRunProbability"] == pytest.approx(0.29)
        assert judge["HitProbability"] == pytest.approx(0.68)
        # No SB market (under ignored) → Poisson fallback from successes
        assert judge["StolenBaseProbability"] == pytest.approx(
            api._poisson_prob(0.02)
        )

    def test_batter_poisson_fallback_when_no_markets(self):
        df = api.build_batters_frame(GAMES, AVERAGES, TEAM_MAP, _hand_lookup(), {})
        judge = df[df["FullName"] == "Aaron Judge"].iloc[0]
        assert judge["HomeRunProbability"] == pytest.approx(api._poisson_prob(0.31))
        assert 0 < judge["HomeRunProbability"] < 1

    def test_pitchers_starters_only(self):
        df = api.build_pitchers_frame(GAMES, AVERAGES, TEAM_MAP, _hand_lookup())
        assert len(df) == 1  # Bullpen Guy filtered out
        cole = df.iloc[0]
        assert cole["FullName"] == "Gerrit Cole"
        assert cole["Team"] == "NYY"
        assert cole["Opponent"] == "BOS"
        assert cole["WinPct"] == pytest.approx(0.55)
        assert cole["LossPct"] == pytest.approx(0.30)
        assert cole["NdPct"] == pytest.approx(0.15)
        assert cole["QualityStart"] == pytest.approx(0.62)
        assert cole["PointsFD"] == pytest.approx(35.5)
        assert cole["PitcherHand"] == "R"

    def test_games_frame(self):
        win_probs = {100: {1: 0.565, 2: 0.435}, 200: {4: 0.35, 3: 0.65}}
        df = api.build_games_frame(GAMES, AVERAGES, TEAM_MAP, win_probs)
        g100 = df[df["GamePk"] == 100].iloc[0]
        assert g100["AwayTeam"] == "NYY"
        assert g100["HomeTeam"] == "BOS"
        assert g100["RunsAway"] == pytest.approx(4.6)
        assert g100["RunsHome"] == pytest.approx(3.9)
        assert g100["AwayWinPct"] == pytest.approx(0.565)
        assert g100["HomeWinPct"] == pytest.approx(0.435)

    def test_teams_frame_sums_batters(self):
        batters = api.build_batters_frame(GAMES, AVERAGES, TEAM_MAP, _hand_lookup(), {})
        win_probs = {100: {1: 0.565, 2: 0.435}, 200: {4: 0.35, 3: 0.65}}
        df = api.build_teams_frame(GAMES, AVERAGES, TEAM_MAP, win_probs, batters)
        assert len(df) == 4  # 2 teams × 2 games
        nyy = df[(df["GamePk"] == 100) & (df["Team"] == "NYY")].iloc[0]
        assert nyy["Runs"] == pytest.approx(4.6)
        assert nyy["WinPercent"] == pytest.approx(0.565)
        assert nyy["Side"] == "A"
        assert nyy["HomeRuns"] == pytest.approx(0.31)  # Judge's HR sum
        lad = df[(df["GamePk"] == 200) & (df["Team"] == "LAD")].iloc[0]
        assert lad["Side"] == "H"
        assert lad["Runs"] == pytest.approx(4.9)


# ── HTTP layer ────────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    """Routes GETs by URL path to canned payloads."""

    def __init__(self, routes, headers=None):
        self.routes = routes
        self.headers = headers or {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        for path, responder in self.routes.items():
            if path in url:
                if isinstance(responder, FakeResponse):
                    return responder
                if callable(responder):
                    return responder(params)
                return FakeResponse(200, responder)
        return FakeResponse(404, {"error": {"code": "not_found", "message": url}})


class TestHttpLayer:
    def _session(self):
        return FakeSession({"/health": {"meta": {}, "data": {"status": "ok"}}})

    def test_401_raises_auth(self):
        s = FakeSession({"/games": FakeResponse(401, {"error": {"code": "unauthorized", "message": "bad key"}})})
        with pytest.raises(api.BppAuthError):
            api._get(s, "/games", backoff=0)

    def test_403_raises_auth(self):
        s = FakeSession({"/games": FakeResponse(403, {"error": {"code": "forbidden", "message": "lapsed"}})})
        with pytest.raises(api.BppAuthError):
            api._get(s, "/games", backoff=0)

    def test_429_retries_then_raises(self, monkeypatch):
        monkeypatch.setattr(api.time, "sleep", lambda *_: None)
        s = FakeSession({"/games": FakeResponse(429, {"error": {"code": "rate_limited", "message": "slow down"}})})
        with pytest.raises(api.BppRateLimitError):
            api._get(s, "/games", max_retries=2, backoff=0)
        assert len(s.calls) == 3  # initial + 2 retries

    def test_date_out_of_range(self):
        s = FakeSession({"/games": FakeResponse(400, {"error": {"code": "date_out_of_range", "message": "past date"}})})
        with pytest.raises(api.BppDateRangeError):
            api._get(s, "/games", backoff=0)


# ── End-to-end: fetch_bpp_api_data with a fully faked session ────────────────


@pytest.fixture
def fake_full_session():
    return FakeSession(
        {
            "/teams": {
                "meta": {},
                "data": [
                    {"teamId": 1, "abv": "NYY", "city": "New York", "nickname": "Yankees"},
                    {"teamId": 2, "abv": "BOS", "city": "Boston", "nickname": "Red Sox"},
                    {"teamId": 3, "abv": "LAD", "city": "Los Angeles", "nickname": "Dodgers"},
                    {"teamId": 4, "abv": "SFG", "city": "San Francisco", "nickname": "Giants"},
                ],
            },
            "/projections/averages": lambda params: FakeResponse(
                200, {"meta": {}, "data": AVERAGES[int(params["gameId"])]}
            ),
            "/projections/probabilities": lambda params: FakeResponse(
                200, {"meta": {}, "data": PROBABILITIES[int(params["gameId"])]}
            ),
            "/parkfactors/hitters": {
                "meta": {},
                "data": [
                    {
                        "gameId": 100, "gameTime": "7:07", "teamAway": "NYY",
                        "teamHome": "BOS", "playerId": 501, "playerName": "Aaron Judge",
                        "team": "NYY", "homeRuns": 1.1, "doublesTriples": 1.0,
                        "singles": 0.98, "homeRunsStadium": 1.05,
                        "doublesTriplesStadium": 1.0, "singlesStadium": 1.0,
                        "homeRunsWeather": 1.05, "doublesTriplesWeather": 1.0,
                        "singlesWeather": 0.98,
                    }
                ],
            },
            "/parkfactors": {
                "meta": {},
                "data": [
                    {
                        "gameId": 100, "gameTime": "7:07", "teamAway": "NYY",
                        "teamHome": "BOS", "runsPercent": 104, "homeRunsPercent": 110,
                        "doublesTriplesPercent": 100, "singlesPercent": 98,
                        "runsAmount": 0.2, "homeRunsAmount": 0.05,
                        "doublesTriplesAmount": 0.0, "singlesAmount": -0.02,
                    }
                ],
            },
            "/games": lambda params: FakeResponse(
                200, {"meta": {}, "data": GAMES}
            ),
        }
    )


class TestFetchEndToEnd:
    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("BPP_API_KEY", raising=False)
        with pytest.warns(UserWarning, match="BPP_API_KEY not set"):
            assert api.fetch_bpp_api_data() is None

    def test_full_bundle(self, monkeypatch, fake_full_session):
        monkeypatch.setattr(api, "_make_session", lambda key: fake_full_session)
        monkeypatch.setattr(api, "fetch_mlb_handedness", lambda season=None: MLB_HAND)
        bundle = api.fetch_bpp_api_data(
            api_key="TESTKEY", date_str="2026-07-17", sleep_seconds=0
        )
        assert bundle is not None
        s = bundle.summary()
        assert s["batters"] == 2
        assert s["pitchers"] == 1
        assert s["games"] == 2
        assert s["teams"] == 4
        assert s["park_factors"] == 1
        assert s["hitter_park_factors"] == 1
        assert s["probabilities"] == 5

        # Batting orders derive from BattingPosition
        bo = bundle.batting_orders()
        assert set(bo.columns) == {"team", "order_position", "player_name"}
        assert bo.loc[bo["player_name"] == "Aaron Judge", "order_position"].iloc[0] == 2

        # Handedness merges batters + pitchers
        hand = bundle.handedness()
        assert set(hand.columns) == {"player_name", "team", "bats", "throws"}
        assert hand.loc[hand["player_name"] == "Rafael Devers", "bats"].iloc[0] == "L"
        assert hand.loc[hand["player_name"] == "Gerrit Cole", "throws"].iloc[0] == "R"

        # SF normalized from SFG alias
        assert "SF" in set(bundle.teams["Team"])

        # Files written with the legacy names the pipeline expects
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            csvs = bundle.to_csvs(tmp)
            excels = bundle.to_excels(tmp)
            assert os.path.exists(f"{tmp}/bpp_batters_2026-07-17.csv")
            assert os.path.exists(f"{tmp}/bpp_probabilities_2026-07-17.csv")
            assert os.path.exists(f"{tmp}/bpp_hitter_park_factors_2026-07-17.csv")
            assert os.path.exists(f"{tmp}/batting_orders_2026-07-17.csv")
            assert os.path.exists(f"{tmp}/handedness_2026-07-17.csv")
            for stem in ["Batters", "Pitchers", "Games", "Teams"]:
                assert os.path.exists(f"{tmp}/BallparkPal_{stem}_2026-07-17.xlsx")
            assert "batters" in csvs and "batters" in excels

            # The written Excels must round-trip through the EXISTING loader —
            # this is the contract the whole downstream pipeline depends on.
            from slate_optimizer.ingestion.ballparkpal import BallparkPalLoader

            loader = BallparkPalLoader(tmp)
            loaded = loader.load_bundle()
            assert len(loaded.batters) == 2
            # snake_case normalization: PointsFD → points_fd etc.
            assert "points_fd" in loaded.batters.columns
            assert "batting_position" in loaded.batters.columns
            assert "win_pct" in loaded.pitchers.columns
            assert "runs_away" in loaded.games.columns
            assert "win_percent" in loaded.teams.columns

    def test_auth_failure_returns_none(self, monkeypatch):
        bad = FakeSession({"/teams": FakeResponse(401, {"error": {"code": "unauthorized", "message": "bad key"}})})
        monkeypatch.setattr(api, "_make_session", lambda key: bad)
        with pytest.warns(UserWarning):
            assert api.fetch_bpp_api_data(api_key="BAD", sleep_seconds=0) is None
