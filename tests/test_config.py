from personica.config import Settings

ALL_VARS = [
    "PERSONICA_DATA_DIR", "ASSISTANT_DATA_DIR",
    "PERSONICA_EMBED_MODEL", "ASSISTANT_EMBED_MODEL",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "OPENROUTER_UTILITY_MODEL", "OPENROUTER_BASE_URL",
    "PERSONICA_KEEP_LAST_TURNS", "ASSISTANT_KEEP_LAST_TURNS",
    "PERSONICA_RETRIEVAL_TOP_K", "ASSISTANT_RETRIEVAL_TOP_K",
    "PERSONICA_RETRIEVAL_MIN_SCORE", "ASSISTANT_RETRIEVAL_MIN_SCORE",
    "PERSONICA_RELEVANCE_WEIGHT", "PERSONICA_RECENCY_HALF_LIFE_DAYS",
    "PERSONICA_TIMEZONE", "ASSISTANT_TIMEZONE",
    "PERSONICA_LOG_LEVEL", "ASSISTANT_LOG_LEVEL",
]


def clear_env(monkeypatch):
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults(monkeypatch):
    clear_env(monkeypatch)
    s = Settings.from_env()
    assert s.data_dir == "./personica_data"
    assert s.api_key == ""
    assert s.chat_model == "openrouter/openai/gpt-4o"
    assert s.utility_model == "openrouter/openai/gpt-4o-mini"
    assert s.base_url == "https://openrouter.ai/api/v1"
    assert s.keep_last_turns == 5
    assert s.retrieval_top_k == 5
    assert s.retrieval_min_score == 0.20
    assert s.relevance_weight == 0.7
    assert s.recency_half_life_days == 30.0
    assert s.timezone == "Asia/Kolkata"
    assert s.log_level == "INFO"


def test_hybrid_ranking_overrides(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("PERSONICA_RELEVANCE_WEIGHT", "0.9")
    monkeypatch.setenv("PERSONICA_RECENCY_HALF_LIFE_DAYS", "7")
    s = Settings.from_env()
    assert s.relevance_weight == 0.9
    assert s.recency_half_life_days == 7.0


def test_env_overrides(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("PERSONICA_DATA_DIR", "/tmp/pd")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("PERSONICA_KEEP_LAST_TURNS", "3")
    monkeypatch.setenv("PERSONICA_RETRIEVAL_MIN_SCORE", "0.5")
    s = Settings.from_env()
    assert s.data_dir == "/tmp/pd"
    assert s.api_key == "sk-test"
    assert s.keep_last_turns == 3
    assert s.retrieval_min_score == 0.5


def test_legacy_assistant_vars_still_work(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("ASSISTANT_DATA_DIR", "/tmp/legacy")
    monkeypatch.setenv("ASSISTANT_TIMEZONE", "Europe/Paris")
    s = Settings.from_env()
    assert s.data_dir == "/tmp/legacy"
    assert s.timezone == "Europe/Paris"


def test_personica_vars_win_over_legacy(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("ASSISTANT_DATA_DIR", "/tmp/legacy")
    monkeypatch.setenv("PERSONICA_DATA_DIR", "/tmp/new")
    s = Settings.from_env()
    assert s.data_dir == "/tmp/new"


def test_invalid_numbers_fall_back_to_defaults(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("PERSONICA_KEEP_LAST_TURNS", "not-a-number")
    monkeypatch.setenv("PERSONICA_RETRIEVAL_MIN_SCORE", "abc")
    s = Settings.from_env()
    assert s.keep_last_turns == 5
    assert s.retrieval_min_score == 0.20


def test_base_url_trailing_slash_stripped(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.com/api/")
    s = Settings.from_env()
    assert s.base_url == "https://example.com/api"
