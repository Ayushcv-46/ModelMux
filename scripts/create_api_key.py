"""
CLI Script: Generate and register a ModelMux API Key (Day 31).

Usage:
    python scripts/create_api_key.py --name "acme-corp" --rate-limit 100 --tenant "tenant_acme"
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so imports work cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.auth import create_api_key
from app.db import SessionLocal, init_db


def main():
    parser = argparse.ArgumentParser(description="Generate and register a ModelMux API Key")
    parser.add_argument("--name", required=True, help="Descriptive name/label for this key (e.g. 'dev-app')")
    parser.add_argument("--rate-limit", type=int, default=60, help="Rate limit in requests per minute (default: 60)")
    parser.add_argument("--tenant", default=None, help="Tenant or organization identifier (default: matches name)")

    args = parser.parse_args()

    # Ensure tables exist
    init_db()

    db = SessionLocal()
    try:
        record, plain_key = create_api_key(
            db=db,
            name=args.name,
            rate_limit=args.rate_limit,
            tenant_id=args.tenant,
        )

        print("\n" + "=" * 68)
        print(" [ModelMux] API KEY CREATED SUCCESSFULLY")
        print("=" * 68)
        print(f" ID:          {record.id}")
        print(f" Name:        {record.name}")
        print(f" Tenant ID:   {record.tenant_id}")
        print(f" Rate Limit:  {record.rate_limit} req/min")
        print(f" Active:      {record.active}")
        print(f" Hash (DB):   {record.key_hash[:16]}...{record.key_hash[-8:]} (SHA-256)")
        print("-" * 68)
        print(" YOUR SECRET API KEY:")
        print(f"   {plain_key}")
        print("-" * 68)
        print(" IMPORTANT: Copy this key now! It is displayed ONCE and cannot")
        print(" be recovered later from the database.")
        print("=" * 68 + "\n")

    finally:
        db.close()


if __name__ == "__main__":
    main()
