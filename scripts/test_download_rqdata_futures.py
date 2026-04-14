from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import download_rqdata_futures as downloader


class DummyRQData:
    def __init__(self) -> None:
        self.init_calls = []
        self.get_price_calls = []

    def init(self, user: str, password: str) -> None:
        self.init_calls.append((user, password))

    def get_price(self, **kwargs):
        self.get_price_calls.append(kwargs)
        index = pd.to_datetime(["2025-01-02", "2025-01-03"])
        return pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.1, 2.1],
                "low": [0.9, 1.9],
                "close": [1.05, 2.05],
                "volume": [10, 20],
                "open_interest": [100, 200],
            },
            index=index,
        )


class DummyFutures:
    def __init__(self) -> None:
        self.calls = []

    def get_dominant_price(self, **kwargs):
        self.calls.append(kwargs)
        index = pd.to_datetime(["2025-01-02", "2025-01-03"])
        return pd.DataFrame(
            {
                "open": [3.0, 4.0],
                "high": [3.1, 4.1],
                "low": [2.9, 3.9],
                "close": [3.05, 4.05],
                "volume": [30, 40],
                "open_interest": [300, 400],
            },
            index=index,
        )


def write_cfg(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def make_cfg(tmp_path: Path) -> str:
    out_dir = tmp_path / "raw"
    normalized_base = tmp_path / "normalized" / "hab_bars"
    return f"""
[download]
start_date = 2018-01-01
end_date = 2026-01-01
frequency = 1d
fields = open, high, low, close, volume, open_interest
raw_output_dir = {out_dir}
normalized_output_base = {normalized_base}
write_csv = true
write_parquet = true
overwrite = false

[adapter]
default_slippage_override =
drop_zero_volume_rows = false

[job:contract]
kind = contract
order_book_id = CU2506
underlying_symbol = CU
strategy_symbol = CU2506
normalize = false

[job:dominant]
kind = dominant
underlying_symbol = CU
strategy_symbol = CU
variants = none, pre
normalize = true
normalize_variant = pre
""".strip()


def test_init_rqdata_from_env_requires_credentials(monkeypatch) -> None:
    dummy = DummyRQData()
    monkeypatch.setattr(downloader, "rqdatac", dummy)
    monkeypatch.delenv("RQDATAC_USER", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="Missing RQData credentials"):
        downloader.init_rqdata_from_env()


def test_load_env_file_sets_missing_values(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\nRQDATAC_USER=test_user\nRQDATAC_PASSWORD='test_pass'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RQDATAC_USER", raising=False)
    monkeypatch.delenv("RQDATAC_PASSWORD", raising=False)

    loaded = downloader.load_env_file(env_path)

    assert loaded is True
    assert downloader.os.environ["RQDATAC_USER"] == "test_user"
    assert downloader.os.environ["RQDATAC_PASSWORD"] == "test_pass"


def test_load_env_file_does_not_override_existing_env_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("RQDATAC_USER=file_user\n", encoding="utf-8")
    monkeypatch.setenv("RQDATAC_USER", "existing_user")

    downloader.load_env_file(env_path)

    assert downloader.os.environ["RQDATAC_USER"] == "existing_user"


def test_load_config_reads_cfg(tmp_path: Path) -> None:
    config_path = tmp_path / "config.cfg"
    write_cfg(config_path, make_cfg(tmp_path))

    config = downloader.load_config(config_path)

    assert config["start_date"] == "2018-01-01"
    assert config["raw_output_dir"] == str(tmp_path / "raw")
    assert config["normalized_output_base"] == str(tmp_path / "normalized" / "hab_bars")
    assert len(config["jobs"]) == 2


def test_load_config_rejects_missing_download_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.cfg"
    write_cfg(config_path, "[adapter]\ndrop_zero_volume_rows = false\n")

    with pytest.raises(ValueError, match="required \\[download\\] section"):
        downloader.load_config(config_path)


def test_load_config_rejects_invalid_dominant_variant(tmp_path: Path) -> None:
    config_path = tmp_path / "config.cfg"
    write_cfg(
        config_path,
        """
[download]
start_date = 2018-01-01
end_date = 2026-01-01
frequency = 1d
fields = open, high, low, close, volume, open_interest

[job:dominant]
kind = dominant
underlying_symbol = CU
variants = none, bad
""".strip(),
    )

    with pytest.raises(ValueError, match="unsupported dominant variants"):
        downloader.load_config(config_path)


def test_download_contract_job_calls_get_price_with_expected_params(
    monkeypatch, tmp_path: Path
) -> None:
    dummy = DummyRQData()
    monkeypatch.setattr(downloader, "rqdatac", dummy)

    result, normalization_input = downloader.download_contract_job(
        {"kind": "contract", "order_book_id": "CU2506", "underlying_symbol": "CU"},
        start_date="2018-01-01",
        end_date="2026-01-01",
        frequency="1d",
        fields=downloader.DEFAULT_FIELDS,
        raw_output_dir=tmp_path,
        overwrite=True,
        write_csv=True,
        write_parquet=True,
    )

    assert result.status == "ok"
    assert normalization_input is not None
    assert dummy.get_price_calls == [
        {
            "order_book_ids": "CU2506",
            "start_date": "2018-01-01",
            "end_date": "2026-01-01",
            "frequency": "1d",
            "fields": downloader.DEFAULT_FIELDS,
            "adjust_type": "none",
        }
    ]
    assert (tmp_path / "contracts" / "CU2506.csv").exists()
    assert (tmp_path / "contracts" / "CU2506.parquet").exists()


def test_download_dominant_job_calls_expected_variants(monkeypatch, tmp_path: Path) -> None:
    dummy = DummyFutures()
    monkeypatch.setattr(downloader, "rqfutures", dummy)

    result, normalization_input = downloader.download_dominant_job(
        {"kind": "dominant", "underlying_symbol": "CU", "variants": ["none", "pre"]},
        start_date="2018-01-01",
        end_date="2026-01-01",
        frequency="1d",
        fields=downloader.DEFAULT_FIELDS,
        raw_output_dir=tmp_path,
        overwrite=True,
        write_csv=True,
        write_parquet=True,
    )

    assert result.status == "ok"
    assert normalization_input is not None
    assert dummy.calls == [
        {
            "underlying_symbols": "CU",
            "start_date": "2018-01-01",
            "end_date": "2026-01-01",
            "frequency": "1d",
            "fields": downloader.DEFAULT_FIELDS,
            "adjust_type": "none",
        },
        {
            "underlying_symbols": "CU",
            "start_date": "2018-01-01",
            "end_date": "2026-01-01",
            "frequency": "1d",
            "fields": downloader.DEFAULT_FIELDS,
            "adjust_type": "pre",
            "adjust_method": "prev_close_spread",
        },
    ]
    assert (tmp_path / "dominant_none" / "CU.csv").exists()
    assert (tmp_path / "dominant_none" / "CU.parquet").exists()
    assert (tmp_path / "dominant_pre" / "CU.csv").exists()
    assert (tmp_path / "dominant_pre" / "CU.parquet").exists()


def test_write_outputs_skips_existing_files_without_overwrite(tmp_path: Path) -> None:
    base_path = tmp_path / "contracts" / "CU2506"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.with_suffix(".csv").write_text("existing", encoding="utf-8")
    base_path.with_suffix(".parquet").write_text("existing", encoding="utf-8")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [10],
            "open_interest": [100],
        }
    )

    details = downloader.write_outputs(
        df=df,
        base_path=base_path,
        overwrite=False,
        write_csv=True,
        write_parquet=True,
    )

    assert "skip existing" in details[0]
    assert base_path.with_suffix(".csv").read_text(encoding="utf-8") == "existing"


def test_normalize_output_frame_resets_datetime_index() -> None:
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [10],
            "open_interest": [100],
        },
        index=pd.to_datetime(["2025-01-02"]),
    )

    normalized = downloader.normalize_output_frame(df)

    assert "date" in normalized.columns
    assert normalized.loc[0, "date"] == pd.Timestamp("2025-01-02")


def test_build_normalized_dataset_uses_adapter_shape(tmp_path: Path) -> None:
    normalization_input = downloader.NormalizationInput(
        rq_df=pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02"]),
                "open": [3.0],
                "high": [3.1],
                "low": [2.9],
                "close": [3.05],
                "volume": [30],
                "open_interest": [300],
            }
        ),
        symbol_spec=downloader.RQSymbolSpec(
            rq_symbol="CU_dominant_pre",
            underlying_symbol="CU",
            strategy_symbol="CU",
        ),
    )

    details = downloader.build_normalized_dataset(
        [normalization_input],
        normalized_output_base=tmp_path / "normalized" / "hab_bars",
        adapter_config={"drop_zero_volume_rows": False},
        overwrite=True,
        write_csv=True,
        write_parquet=True,
    )

    csv_path = tmp_path / "normalized" / "hab_bars.csv"
    parquet_path = tmp_path / "normalized" / "hab_bars.parquet"
    normalized = pd.read_parquet(parquet_path)

    assert "wrote" in details[0]
    assert csv_path.exists()
    assert parquet_path.exists()
    assert list(normalized.columns) == downloader.RQDataFuturesResearchAdapter.OUTPUT_COLUMNS
    assert normalized.loc[0, "symbol"] == "CU"


def test_run_jobs_continues_after_failures(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.cfg"
    write_cfg(config_path, make_cfg(tmp_path))
    config = downloader.load_config(config_path)

    def failing_contract_job(*args, **kwargs):
        raise RuntimeError("contract failed")

    def ok_dominant_job(*args, **kwargs):
        return (
            downloader.JobResult(job_label="dominant:CU", status="ok", details=["done"]),
            downloader.NormalizationInput(
                rq_df=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2025-01-02"]),
                        "open": [3.0],
                        "high": [3.1],
                        "low": [2.9],
                        "close": [3.05],
                        "volume": [30],
                        "open_interest": [300],
                    }
                ),
                symbol_spec=downloader.RQSymbolSpec(
                    rq_symbol="CU_dominant_pre",
                    underlying_symbol="CU",
                    strategy_symbol="CU",
                ),
            ),
        )

    monkeypatch.setattr(downloader, "download_contract_job", failing_contract_job)
    monkeypatch.setattr(downloader, "download_dominant_job", ok_dominant_job)

    outcome = downloader.run_jobs(config=config, overwrite=False)

    assert len(outcome.results) == 2
    assert outcome.results[0].status == "error"
    assert outcome.results[1].status == "ok"
    assert len(outcome.normalization_inputs) == 1


def test_main_builds_normalized_dataset_and_returns_zero(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config_path = tmp_path / "config.cfg"
    write_cfg(config_path, make_cfg(tmp_path))

    dummy = DummyRQData()
    dummy_futures = DummyFutures()
    monkeypatch.setattr(downloader, "rqdatac", dummy)
    monkeypatch.setattr(downloader, "rqfutures", dummy_futures)
    monkeypatch.setenv("RQDATAC_USER", "user")
    monkeypatch.setenv("RQDATAC_PASSWORD", "pass")

    exit_code = downloader.main(["--config", str(config_path)])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "[OK] normalized:hab_bars" in stdout
    assert (tmp_path / "normalized" / "hab_bars.parquet").exists()
