"""
Full ETL pipeline orchestrator.
Runs all four stages in sequence: collect → bronze → silver → gold.

Usage:
    python -m etl.run_pipeline --start 2018-01-01 --end 2024-12-31
    python -m etl.run_pipeline --start 2025-01-01 --end 2025-12-31 --force
    python -m etl.run_pipeline --skip-collect  # skip ingestion, rebuild layers only
"""

import argparse
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run full ETL pipeline")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--force", "-f", action="store_true", help="Re-download existing partitions")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection step")
    args = parser.parse_args()

    print("=" * 60)
    print("Energy Price Forecast — Full ETL Pipeline")
    print("=" * 60)

    if not args.skip_collect:
        print("\nStep 1/4: Collecting data...")
        from etl.collect import CollectPrices, CollectLoad, CollectWeather

        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        api_key = os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise EnvironmentError("ENTSOE_API_KEY not set. Check your .env file.")

        CollectPrices(api_key).process(start, end, force=args.force)
        CollectLoad(api_key).process(start, end, force=args.force)
        CollectWeather().process(start, end, force=args.force)
    else:
        print("\nStep 1/4: Skipping data collection (--skip-collect).")

    print("\nStep 2/4: Building bronze layer...")
    from etl.bronze import build_bronze
    build_bronze()

    print("\nStep 3/4: Building silver layer (features)...")
    from etl.silver import build_silver
    build_silver()

    print("\nStep 4/4: Building gold layer (ABT + Feast prep)...")
    from etl.gold import build_gold
    build_gold()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    print("\nNext steps:")
    print("  cd feature_store && feast apply")
    print("  cd feature_store && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)")
    print("  zenml init && zenml up")
    print("  python -c \"from pipelines.training_pipeline import training_pipeline; training_pipeline()\"")


if __name__ == "__main__":
    main()
