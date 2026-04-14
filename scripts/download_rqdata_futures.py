"""Download RQData futures history and build HAB-ready daily bars.

This script has two distinct outputs:
- raw cache files for audit/debugging
- a normalized bars file that can be fed directly into the HAB backtester

Recommended use in this repo:
- use `contract` jobs when you need a specific deliverable contract
- use `dominant` jobs for symbol-level research
- for `dominant`, keep both `none` and `pre` raw caches, but usually feed `pre`
  into the backtest dataset because it reduces roll-gap noise
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from data.adapters.rqdata_futures_adatpter import (
    RQDataFuturesResearchAdapter,
    RQSymbolSpec,
)

try:
    import rqdatac
    from rqdatac import futures as rqfutures
except ImportError:  # pragma: no cover - exercised via runtime guard
    rqdatac = None
    rqfutures = None


DEFAULT_FIELDS = ["open", "high", "low", "close", "volume", "open_interest"]
DEFAULT_FREQUENCY = "1d"
DEFAULT_VARIANTS = ["none", "pre"]
VALID_KINDS = {"contract", "dominant"}
VALID_VARIANTS = {"none", "pre"}
PRE_ADJUST_METHOD = "prev_close_spread"
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


@dataclass(frozen=True)
class JobResult:
    job_label: str
    status: str
    details: List[str]


@dataclass(frozen=True)
class NormalizationInput:
    rq_df: pd.DataFrame
    symbol_spec: RQSymbolSpec


@dataclass(frozen=True)
class RunJobsOutcome:
    results: List[JobResult]
    normalization_inputs: List[NormalizationInput]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw RQData futures history and build HAB-ready bars."
    )
    parser.add_argument("--config", required=True, help="Path to .cfg or .json config file.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_args(argv)


def parse_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def load_env_file(path: str | Path = DEFAULT_ENV_PATH, *, overwrite: bool = False) -> bool:
    env_path = Path(path)
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if overwrite or key not in os.environ:
            os.environ[key] = value

    return True


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".cfg":
        return load_cfg_config(config_path)
    if suffix == ".json":
        return load_json_config(config_path)
    raise ValueError("Unsupported config format. Use .cfg or .json.")


def load_json_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    return normalize_loaded_config(config)


def load_cfg_config(path: Path) -> Dict[str, Any]:
    parser = configparser.ConfigParser()
    read_files = parser.read(path, encoding="utf-8")
    if not read_files:
        raise ValueError(f"Could not read config file: {path}")

    if not parser.has_section("download"):
        raise ValueError("Config missing required [download] section.")

    download = parser["download"]
    adapter = parser["adapter"] if parser.has_section("adapter") else {}

    jobs: List[Dict[str, Any]] = []
    for section_name in parser.sections():
        if not section_name.startswith("job:"):
            continue
        section = parser[section_name]
        job: Dict[str, Any] = {
            "name": section_name.split("job:", 1)[1],
            "kind": section.get("kind"),
            "underlying_symbol": section.get("underlying_symbol"),
            "strategy_symbol": section.get("strategy_symbol"),
            "normalize": section.getboolean("normalize", fallback=True),
        }
        if section.get("order_book_id"):
            job["order_book_id"] = section.get("order_book_id")
        variants = parse_list(section.get("variants"))
        if variants:
            job["variants"] = variants
        normalize_variant = section.get("normalize_variant")
        if normalize_variant:
            job["normalize_variant"] = normalize_variant.strip()
        jobs.append(job)

    config = {
        "start_date": download.get("start_date"),
        "end_date": download.get("end_date"),
        "frequency": download.get("frequency", fallback=DEFAULT_FREQUENCY),
        "fields": parse_list(download.get("fields")) or list(DEFAULT_FIELDS),
        "raw_output_dir": download.get("raw_output_dir", fallback="data/cache/raw_rqdata"),
        "normalized_output_base": download.get(
            "normalized_output_base",
            fallback="data/cache/normalized/hab_bars",
        ),
        "write_csv": download.getboolean("write_csv", fallback=True),
        "write_parquet": download.getboolean("write_parquet", fallback=True),
        "overwrite": download.getboolean("overwrite", fallback=False),
        "adapter": {
            "default_slippage_override": parse_optional_float(
                adapter.get("default_slippage_override")
                if hasattr(adapter, "get")
                else None
            ),
            "drop_zero_volume_rows": (
                adapter.getboolean("drop_zero_volume_rows", fallback=False)
                if hasattr(adapter, "getboolean")
                else False
            ),
        },
        "jobs": jobs,
    }
    return normalize_loaded_config(config)


def normalize_loaded_config(config: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = {
        "start_date",
        "end_date",
        "frequency",
        "fields",
        "jobs",
    }
    missing = sorted(key for key in required_keys if not config.get(key))
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    if config["frequency"] != DEFAULT_FREQUENCY:
        raise ValueError("Only daily frequency '1d' is supported in this downloader.")

    fields = config["fields"]
    if not isinstance(fields, list) or not fields:
        raise ValueError("'fields' must be a non-empty list.")

    jobs = config["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("'jobs' must be a non-empty list.")

    normalized = {
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "frequency": config["frequency"],
        "fields": list(fields),
        "raw_output_dir": config.get("raw_output_dir") or config.get("output_dir") or "data/cache/raw_rqdata",
        "normalized_output_base": config.get("normalized_output_base", "data/cache/normalized/hab_bars"),
        "write_csv": bool(config.get("write_csv", True)),
        "write_parquet": bool(config.get("write_parquet", True)),
        "overwrite": bool(config.get("overwrite", False)),
        "adapter": config.get("adapter", {}) or {},
        "jobs": jobs,
    }

    for idx, job in enumerate(normalized["jobs"]):
        validate_job(job=job, index=idx)

    return normalized


def validate_job(job: Dict[str, Any], index: int) -> None:
    kind = job.get("kind")
    if kind not in VALID_KINDS:
        raise ValueError(f"Job #{index} has invalid kind: {kind!r}")

    if not job.get("underlying_symbol"):
        raise ValueError(f"Job #{index} requires 'underlying_symbol'.")

    if kind == "contract":
        if not job.get("order_book_id"):
            raise ValueError(f"Job #{index} kind='contract' requires 'order_book_id'.")
        return

    variants = job.get("variants", DEFAULT_VARIANTS)
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"Job #{index} dominant variants must be a non-empty list.")

    invalid_variants = sorted(set(variants) - VALID_VARIANTS)
    if invalid_variants:
        raise ValueError(
            f"Job #{index} has unsupported dominant variants: {invalid_variants}"
        )

    normalize_variant = job.get("normalize_variant")
    if normalize_variant and normalize_variant not in variants:
        raise ValueError(
            f"Job #{index} normalize_variant={normalize_variant!r} is not in variants={variants}"
        )


def init_rqdata_from_env() -> None:
    if rqdatac is None:
        raise RuntimeError("rqdatac is not installed. Install it before running this script.")

    user = os.getenv("RQDATAC_USER")
    password = os.getenv("RQDATAC_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "Missing RQData credentials. Set RQDATAC_USER and RQDATAC_PASSWORD."
        )

    rqdatac.init(user, password)


def normalize_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        first_col = out.columns[0]
        out = out.rename(columns={first_col: "date"})
    elif "date" not in out.columns and "datetime" in out.columns:
        out = out.rename(columns={"datetime": "date"})

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    return out


def write_outputs(
    df: pd.DataFrame,
    base_path: Path,
    *,
    overwrite: bool,
    write_csv: bool,
    write_parquet: bool,
) -> List[str]:
    if not write_csv and not write_parquet:
        raise ValueError("At least one output format must be enabled.")

    csv_path = base_path.with_suffix(".csv")
    parquet_path = base_path.with_suffix(".parquet")
    requested_paths = []
    if write_csv:
        requested_paths.append(csv_path)
    if write_parquet:
        requested_paths.append(parquet_path)

    if not overwrite and any(path.exists() for path in requested_paths):
        return [f"skip existing {path}" for path in requested_paths]

    normalized = normalize_output_frame(df)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    details: List[str] = []

    if write_csv:
        normalized.to_csv(csv_path, index=False)
        details.append(f"wrote {csv_path}")
    if write_parquet:
        normalized.to_parquet(parquet_path, index=False)
        details.append(f"wrote {parquet_path}")

    return details


def build_symbol_spec(job: Dict[str, Any], rq_symbol: str) -> RQSymbolSpec:
    underlying_symbol = str(job["underlying_symbol"]).upper()
    strategy_symbol = str(job.get("strategy_symbol") or underlying_symbol)
    return RQSymbolSpec(
        rq_symbol=rq_symbol,
        underlying_symbol=underlying_symbol,
        strategy_symbol=strategy_symbol,
    )


def download_contract_job(
    job: Dict[str, Any],
    *,
    start_date: str,
    end_date: str,
    frequency: str,
    fields: Sequence[str],
    raw_output_dir: Path,
    overwrite: bool,
    write_csv: bool,
    write_parquet: bool,
) -> tuple[JobResult, NormalizationInput | None]:
    if rqdatac is None:
        raise RuntimeError("rqdatac is not available.")

    order_book_id = str(job["order_book_id"])
    df = rqdatac.get_price(
        order_book_ids=order_book_id,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        fields=list(fields),
        adjust_type="none",
    )
    details = write_outputs(
        df=df,
        base_path=raw_output_dir / "contracts" / order_book_id,
        overwrite=overwrite,
        write_csv=write_csv,
        write_parquet=write_parquet,
    )
    normalization_input = None
    if job.get("normalize", True):
        normalization_input = NormalizationInput(
            rq_df=normalize_output_frame(df),
            symbol_spec=build_symbol_spec(job=job, rq_symbol=order_book_id),
        )

    return (
        JobResult(job_label=f"contract:{order_book_id}", status="ok", details=details),
        normalization_input,
    )


def download_dominant_job(
    job: Dict[str, Any],
    *,
    start_date: str,
    end_date: str,
    frequency: str,
    fields: Sequence[str],
    raw_output_dir: Path,
    overwrite: bool,
    write_csv: bool,
    write_parquet: bool,
) -> tuple[JobResult, NormalizationInput | None]:
    if rqfutures is None:
        raise RuntimeError("rqdatac.futures is not available.")

    underlying_symbol = str(job["underlying_symbol"]).upper()
    variants = list(job.get("variants", DEFAULT_VARIANTS))
    normalize_variant = str(
        job.get("normalize_variant") or ("pre" if "pre" in variants else variants[0])
    )

    details: List[str] = []
    selected_df: pd.DataFrame | None = None

    for variant in variants:
        kwargs: Dict[str, Any] = {
            "underlying_symbols": underlying_symbol,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency,
            "fields": list(fields),
            "adjust_type": variant,
        }
        subdir = "dominant_none"
        if variant == "pre":
            kwargs["adjust_method"] = PRE_ADJUST_METHOD
            subdir = "dominant_pre"

        df = rqfutures.get_dominant_price(**kwargs)
        details.extend(
            write_outputs(
                df=df,
                base_path=raw_output_dir / subdir / underlying_symbol,
                overwrite=overwrite,
                write_csv=write_csv,
                write_parquet=write_parquet,
            )
        )
        if variant == normalize_variant:
            selected_df = normalize_output_frame(df)

    normalization_input = None
    if job.get("normalize", True):
        if selected_df is None:
            raise RuntimeError(
                f"No dominant dataframe selected for normalization: {underlying_symbol}"
            )
        normalization_input = NormalizationInput(
            rq_df=selected_df,
            symbol_spec=build_symbol_spec(
                job=job,
                rq_symbol=f"{underlying_symbol}_dominant_{normalize_variant}",
            ),
        )

    return (
        JobResult(job_label=f"dominant:{underlying_symbol}", status="ok", details=details),
        normalization_input,
    )


def build_normalized_dataset(
    normalization_inputs: Sequence[NormalizationInput],
    *,
    normalized_output_base: str | Path,
    adapter_config: Dict[str, Any],
    overwrite: bool,
    write_csv: bool,
    write_parquet: bool,
) -> List[str]:
    if not normalization_inputs:
        return ["skip normalized dataset: no jobs selected for normalization"]

    adapter = RQDataFuturesResearchAdapter(
        default_slippage_override=adapter_config.get("default_slippage_override"),
        drop_zero_volume_rows=bool(adapter_config.get("drop_zero_volume_rows", False)),
    )
    normalized = adapter.normalize_many(
        rq_dfs=[item.rq_df for item in normalization_inputs],
        symbol_specs=[item.symbol_spec for item in normalization_inputs],
    )
    return write_outputs(
        df=normalized,
        base_path=Path(normalized_output_base),
        overwrite=overwrite,
        write_csv=write_csv,
        write_parquet=write_parquet,
    )


def run_jobs(config: Dict[str, Any], overwrite: bool) -> RunJobsOutcome:
    raw_output_dir = Path(config["raw_output_dir"])
    results: List[JobResult] = []
    normalization_inputs: List[NormalizationInput] = []

    for job in config["jobs"]:
        try:
            if job["kind"] == "contract":
                result, normalization_input = download_contract_job(
                    job,
                    start_date=config["start_date"],
                    end_date=config["end_date"],
                    frequency=config["frequency"],
                    fields=config["fields"],
                    raw_output_dir=raw_output_dir,
                    overwrite=overwrite,
                    write_csv=config["write_csv"],
                    write_parquet=config["write_parquet"],
                )
            else:
                result, normalization_input = download_dominant_job(
                    job,
                    start_date=config["start_date"],
                    end_date=config["end_date"],
                    frequency=config["frequency"],
                    fields=config["fields"],
                    raw_output_dir=raw_output_dir,
                    overwrite=overwrite,
                    write_csv=config["write_csv"],
                    write_parquet=config["write_parquet"],
                )
        except Exception as exc:
            label = f"{job['kind']}:{job.get('order_book_id') or job.get('underlying_symbol')}"
            result = JobResult(job_label=label, status="error", details=[str(exc)])
            normalization_input = None

        results.append(result)
        if normalization_input is not None:
            normalization_inputs.append(normalization_input)

    return RunJobsOutcome(results=results, normalization_inputs=normalization_inputs)


def print_summary(results: Iterable[JobResult]) -> None:
    for result in results:
        print(f"[{result.status.upper()}] {result.job_label}")
        for detail in result.details:
            print(f"  - {detail}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    overwrite = args.overwrite or config["overwrite"]

    load_env_file()
    init_rqdata_from_env()
    outcome = run_jobs(config=config, overwrite=overwrite)

    try:
        normalized_details = build_normalized_dataset(
            outcome.normalization_inputs,
            normalized_output_base=config["normalized_output_base"],
            adapter_config=config["adapter"],
            overwrite=overwrite,
            write_csv=config["write_csv"],
            write_parquet=config["write_parquet"],
        )
        outcome.results.append(
            JobResult(
                job_label="normalized:hab_bars",
                status="ok",
                details=normalized_details,
            )
        )
    except Exception as exc:
        outcome.results.append(
            JobResult(
                job_label="normalized:hab_bars",
                status="error",
                details=[str(exc)],
            )
        )

    print_summary(outcome.results)
    return 1 if any(result.status == "error" for result in outcome.results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
