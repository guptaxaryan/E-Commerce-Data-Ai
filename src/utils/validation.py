"""Data validation utilities for uploaded datasets."""
from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class ValidationIssue:
    """Represents a data quality issue."""
    severity: str  # "error", "warning", "info"
    message: str
    table_name: str
    column: str | None = None


class DataValidator:
    """Validates uploaded data quality and structure."""
    
    @staticmethod
    def validate_dataframe(name: str, df: pd.DataFrame) -> List[ValidationIssue]:
        """Run validation checks on a single dataframe."""
        issues: List[ValidationIssue] = []
        
        # Check if dataframe is empty
        if df.empty:
            issues.append(ValidationIssue(
                severity="warning",
                message="Table is empty (no rows)",
                table_name=name
            ))
            return issues
        
        # Check for completely null columns
        for col in df.columns:
            if df[col].isna().all():
                issues.append(ValidationIssue(
                    severity="warning",
                    message="Column contains only null values",
                    table_name=name,
                    column=str(col)
                ))
        
        # Check for high null percentage
        for col in df.columns:
            null_pct = (df[col].isna().sum() / len(df)) * 100
            if null_pct > 50 and null_pct < 100:
                issues.append(ValidationIssue(
                    severity="info",
                    message=f"Column has {null_pct:.1f}% null values",
                    table_name=name,
                    column=str(col)
                ))
        
        # Check for duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            dup_pct = (dup_count / len(df)) * 100
            issues.append(ValidationIssue(
                severity="info",
                message=f"Found {dup_count} duplicate rows ({dup_pct:.1f}%)",
                table_name=name
            ))
        
        # Check for suspiciously wide tables
        if len(df.columns) > 100:
            issues.append(ValidationIssue(
                severity="warning",
                message=f"Table has {len(df.columns)} columns - consider splitting",
                table_name=name
            ))
        
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
        if unnamed_cols:
            issues.append(ValidationIssue(
                severity="warning",
                message=f"Found {len(unnamed_cols)} unnamed columns (possibly from Excel formatting)",
                table_name=name
            ))
        
        return issues
    
    @staticmethod
    def validate_all(dataframes: Dict[str, pd.DataFrame]) -> Tuple[List[ValidationIssue], Dict[str, int]]:
        """Validate all dataframes and return issues + summary stats.
        
        Returns:
            Tuple of (issues_list, stats_dict)
        """
        all_issues: List[ValidationIssue] = []
        stats = {
            "total_tables": len(dataframes),
            "total_rows": 0,
            "total_columns": 0,
            "errors": 0,
            "warnings": 0,
            "infos": 0,
        }
        
        for name, df in dataframes.items():
            issues = DataValidator.validate_dataframe(name, df)
            all_issues.extend(issues)
            
            stats["total_rows"] += len(df)
            stats["total_columns"] += len(df.columns)
        
        # Count severity levels
        for issue in all_issues:
            stats[issue.severity + "s"] += 1
        
        return all_issues, stats
    
    @staticmethod
    def format_issues_report(issues: List[ValidationIssue], limit: int = 10) -> str:
        """Format validation issues as a readable report."""
        if not issues:
            return "✅ No data quality issues detected"
        
        report_lines = [f"Found {len(issues)} data quality issues:\n"]
        
        # Group by severity
        by_severity = {"error": [], "warning": [], "info": []}
        for issue in issues:
            by_severity[issue.severity].append(issue)
        
        # Display errors first, then warnings, then info
        for severity in ["error", "warning", "info"]:
            items = by_severity[severity]
            if not items:
                continue
            
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[severity]
            report_lines.append(f"\n{icon} {severity.upper()} ({len(items)}):")
            
            for issue in items[:limit]:
                location = f"[{issue.table_name}"
                if issue.column:
                    location += f".{issue.column}"
                location += "]"
                report_lines.append(f"  • {location} {issue.message}")
        
        if len(issues) > limit:
            report_lines.append(f"\n... and {len(issues) - limit} more issues")
        
        return "\n".join(report_lines)


__all__ = ["DataValidator", "ValidationIssue"]
