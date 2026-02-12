"""Parquet Adapter — load and transform timeseries data for the API.

Provides column-selective loading, downsampling, and format conversion
(JSON payload or Arrow IPC) for efficient API responses.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_timeseries(
    path: Path,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load a timeseries parquet file, optionally selecting columns.

    Always includes 'timestamp' if available and not explicitly excluded.

    Args:
        path: path to the .parquet file.
        columns: optional list of column names to load.

    Returns:
        pandas DataFrame.
    """
    if columns is not None:
        # Always ensure timestamp is included
        if "timestamp" not in columns:
            columns = ["timestamp"] + columns
        # Only request columns that exist in the file
        schema = pq.read_schema(str(path))
        available = set(schema.names)
        columns = [c for c in columns if c in available]

    df = pd.read_parquet(str(path), columns=columns)
    return df


def read_parquet_metadata(path: Path) -> dict:
    """Read schema-level metadata from a parquet file."""
    schema = pq.read_schema(str(path))
    meta = schema.metadata or {}
    return {
        k.decode() if isinstance(k, bytes) else k:
        v.decode() if isinstance(v, bytes) else v
        for k, v in meta.items()
        if not (isinstance(k, bytes) and k.startswith(b"pandas"))
    }


def downsample(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Downsample a DataFrame to at most max_points rows.

    Uses uniform stride sampling to preserve temporal distribution.
    Always keeps the first and last rows.

    Args:
        df: input DataFrame with a 'timestamp' column.
        max_points: maximum number of rows in the output.

    Returns:
        Downsampled DataFrame.
    """
    if len(df) <= max_points:
        return df

    stride = max(1, len(df) // max_points)
    indices = list(range(0, len(df), stride))
    # Ensure last row is included
    if indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)
    return df.iloc[indices].reset_index(drop=True)


def to_json_payload(df: pd.DataFrame) -> dict:
    """Convert a DataFrame to a JSON-serializable payload.

    Returns:
        dict with keys: columns (list[str]), rows (list[list]).
        NaN/inf values are converted to None for JSON safety.
    """
    columns = df.columns.tolist()
    # Replace NaN/inf with None for JSON serialization
    rows = []
    for _, row in df.iterrows():
        json_row = []
        for val in row:
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                json_row.append(None)
            elif isinstance(val, (np.integer,)):
                json_row.append(int(val))
            elif isinstance(val, (np.floating,)):
                json_row.append(float(val))
            elif isinstance(val, np.bool_):
                json_row.append(bool(val))
            else:
                json_row.append(val)
        rows.append(json_row)
    return {"columns": columns, "rows": rows}


def to_arrow_ipc(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to Arrow IPC stream bytes.

    Args:
        df: input DataFrame.

    Returns:
        bytes of the Arrow IPC stream.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    sink = io.BytesIO()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue()
