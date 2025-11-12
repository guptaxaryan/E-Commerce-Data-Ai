from __future__ import annotations

"""Data loading utilities for Excel workbooks into pandas DataFrames.

Functions here are side-effect free except for reading from disk.
"""
from pathlib import Path
from typing import Dict
import pandas as pd

__all__ = ["load_excel_data"]


def load_excel_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load tabular data under data_dir.

    Supports:
    - Excel: .xls, .xlsx (each sheet becomes <workbook>__<sheet>)
    - CSV: .csv (stored as <filename_stem>)

    Errors are swallowed; only successfully parsed tables are returned.
    """
    tables: Dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        return tables

    # CSV files
    for csv_file in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        tables[csv_file.stem] = df

    # Excel workbooks
    for workbook in sorted(data_dir.glob("*.xls*")):
        try:
            xls = pd.ExcelFile(workbook)
        except Exception:
            continue
        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet)
            except Exception:
                continue
            tables[f"{workbook.stem}__{sheet}"] = df
    return tables
