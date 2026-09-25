"""CLI 冒烟（Typer CliRunner，离线）。"""

from typer.testing import CliRunner

from fastvideo3.app.cli import app

runner = CliRunner()


def test_doctor_runs(outdir):
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output


def test_ls_runs_empty(outdir):
    result = runner.invoke(app, ["ls-runs"])
    assert result.exit_code == 0


def test_stage_script_fake(outdir):
    result = runner.invoke(app, ["stage", "script", "--topic", "CLI测试", "--fake"])
    assert result.exit_code == 0
