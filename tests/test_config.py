"""配置系统测试：.env + yaml 合并、路由解析。"""

from fastvideo3.core.config import AppConfig, load_config


def test_appconfig_defaults_and_db_url(outdir):
    cfg = AppConfig(fake=True)
    assert cfg.db_url.startswith("sqlite:///")
    assert cfg.model_name("deepseek_text", "x") == "x"


def test_appconfig_postgres_url():
    cfg = AppConfig(profile="prod-postgres", pg_password="pw")
    assert cfg.db_url.startswith("postgresql+psycopg://fv3:pw@localhost:5432/fastvideo3")


def test_load_config_merges_env_and_yaml(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "DEEPSEEK_API_KEY=sk-test-123\nDASHSCOPE_API_KEY=sk-ds-456\nFV3_FAKE=1\n",
        encoding="utf-8",
    )
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        """
providers:
  text: {default: deepseek, fallback: dashscope_qwen}
generation:
  talk_mode: audio_overlay
  resolution: [320, 568]
limits:
  max_parallel_jobs: 2
models:
  deepseek_text: deepseek-chat
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path=yml, dotenv_path=env)
    assert cfg.deepseek_api_key == "sk-test-123"
    assert cfg.fake is True
    assert cfg.talk_mode == "audio_overlay"
    assert cfg.resolution == (320, 568)
    assert cfg.max_parallel_jobs == 2
    assert cfg.routing["text"].default == "deepseek"
    assert cfg.routing["text"].fallback == "dashscope_qwen"
