"""Schema Explorer: inferred schema, column details, primary keys and contract validation."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.components.dataset import AnalysisState
from app.components.ui import card, Kpi, empty_state, error_state, fmt_int, kpi_row, page_header, section
from datasentinel.pipeline import AnalysisResult
from datasentinel.quality.models import CheckStatus
from datasentinel.schema.contract import SchemaContract
from datasentinel.schema.models import SemanticType

_CONTRACT_CHECKS = {"required_column", "column_contract", "unexpected_columns"}


def contract_results(result: AnalysisResult) -> list:
    """Schema-rule results produced in contract mode (empty when no contract was supplied)."""
    return [r for r in result.quality.results
            if r.rule_id == "consistency.schema" and r.details.get("check") in _CONTRACT_CHECKS]


def contract_status(result: AnalysisResult) -> str:
    """'No contract', 'Validated' or 'N violation(s)' — from the engine's schema-rule results."""
    results = contract_results(result)
    if not results:
        return "No contract"
    failed = sum(1 for r in results if r.status == CheckStatus.FAILED)
    return "Validated" if failed == 0 else f"{failed} violation(s)"


def column_contract_map(result: AnalysisResult) -> dict[str, str]:
    """Per-column contract outcome (passed / failed / skipped) where a contract was supplied."""
    return {r.column: r.status.value for r in contract_results(result) if r.column}


def schema_kpis(result: AnalysisResult) -> list[Kpi]:
    schema = result.schema
    return [
        Kpi("Columns", fmt_int(schema.n_columns), caption=f"{fmt_int(schema.n_rows)} rows"),
        Kpi("Inferred types", fmt_int(len(schema.type_summary())),
            caption=", ".join(f"{v} {k}" for k, v in sorted(schema.type_summary().items(), key=lambda kv: -kv[1]))),
        Kpi("Nullable columns", fmt_int(sum(1 for c in schema.columns if c.nullable)), caption="contain missing values"),
        Kpi("Primary key", schema.primary_key.label if schema.primary_key else "None",
            caption=f"confidence {schema.primary_key.confidence:.2f}" if schema.primary_key
            else "no unique column or pair"),
        Kpi("Contract", contract_status(result),
            caption="from the uploaded baseline" if contract_results(result) else "upload one in the sidebar"),
    ]


def columns_table(result: AnalysisResult) -> pd.DataFrame:
    """Column explorer rows built from the schema report and profile."""
    schema, profile = result.schema, result.profile
    keys = {c for pk in schema.primary_key_candidates if not pk.is_composite for c in pk.columns}
    contract = column_contract_map(result)
    return pd.DataFrame([{
        "column": c.name, "type": c.semantic_type.value, "storage": c.storage_type, "nullable": c.nullable,
        "null_pct": round(profile.columns[c.name].null_pct, 2), "distinct": c.unique_count,
        "pk_candidate": c.name in keys, "contract": contract.get(c.name, "—"), "notes": "; ".join(c.notes),
    } for c in schema.columns])


def render(state: AnalysisState) -> None:
    """Render the Schema Explorer section."""
    result = state.result
    page_header("Schema Explorer", "Inferred semantic types, keys and contract validation for the active dataset.",
                eyebrow="Analysis", context=[f"Dataset · {state.dataset.filename}"] if state.dataset else [])
    if state.error_title:
        error_state(state.error_title, state.error_detail or "")
        return
    if result is None:
        empty_state("No schema to explore yet", "Load a dataset from the sidebar to infer its schema.",
                    why="Types, keys and contract validation are inferred from the data itself.", icon="search")
        return

    schema = result.schema
    kpi_row(schema_kpis(result))

    with card("sch_1"):
        section("Columns", "Inferred from the data; values that fail to parse are reported as quality issues.")
        types = [t.value for t in SemanticType if t.value in schema.type_summary()]
        selected = st.multiselect("Filter by type", types, default=types, key="schema_types")
        table = columns_table(result)
        st.dataframe(table[table["type"].isin(selected)], hide_index=True, column_config={
            "column": st.column_config.TextColumn("Column", width="medium"),
            "null_pct": st.column_config.NumberColumn("Null %", format="%.2f"),
            "distinct": st.column_config.NumberColumn("Distinct", format="%d"),
            "pk_candidate": st.column_config.CheckboxColumn("PK?"),
            "nullable": st.column_config.CheckboxColumn("Nullable"),
            "contract": st.column_config.TextColumn("Contract"),
            "notes": st.column_config.TextColumn("Notes", width="large"),
        })

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with card("sch_2"):
            section("Column details")
            name = st.selectbox("Column", [c.name for c in schema.columns], key="schema_col")
            column, prof = schema.column(name), result.profile.columns[name]
            c1, c2, c3 = st.columns(3)
            c1.metric("Inferred type", column.semantic_type.value, help=f"Storage: {column.storage_type}")
            c2.metric("Null count", fmt_int(column.null_count), help=f"{prof.null_pct:.2f}% of rows")
            c3.metric("Distinct values", fmt_int(column.unique_count))
            if column.parse_success_ratio is not None:
                st.caption(f"Parsed from text: {column.parse_success_ratio * 100:.1f}% of values matched "
                           f"{column.semantic_type.value}.")
            if column.sample_values:
                st.markdown("**Sample values:** " + ", ".join(f"`{v}`" for v in column.sample_values))
            for note in column.notes:
                st.caption(f"Note: {note}")
            if prof.numeric:
                st.caption(f"Range {prof.numeric.min:g} – {prof.numeric.max:g} · median {prof.numeric.median:g}")
            elif prof.categorical:
                st.caption(f"Cardinality {prof.categorical.cardinality:,} · mode share {prof.categorical.mode_pct:.1f}%")
            elif prof.datetime:
                st.caption(f"Range {prof.datetime.min} – {prof.datetime.max} ({prof.datetime.span_days:g} days)")
    with right:
        with card("sch_3"):
            section("Primary key candidates")
            if schema.primary_key_candidates:
                st.dataframe(pd.DataFrame([{"key": pk.label, "type": "composite" if pk.is_composite else "single",
                                            "confidence": pk.confidence, "evidence": "; ".join(pk.reasons)}
                                           for pk in schema.primary_key_candidates]), hide_index=True,
                             column_config={"confidence": st.column_config.ProgressColumn(
                                 min_value=0, max_value=1, format="%.2f")})
            else:
                st.info("No column or column pair uniquely identifies every row (duplicates or nulls prevent it).")

    _render_contract(result)


def _render_contract(result: AnalysisResult) -> None:
    with card("sch_4"):
        section("Schema contract", "Validate future loads against an expected schema.")
        results = contract_results(result)
        if not results:
            st.info("No contract was supplied. Download the inferred contract below, edit it if needed, and upload it "
                    "under “Schema baseline & quality settings” in the sidebar.")
        else:
            status = contract_status(result)
            (st.success if status == "Validated" else st.warning)(f"Contract status: {status}")
            missing = [r.column for r in results if r.details.get("check") == "required_column"
                       and r.status == CheckStatus.FAILED]
            mismatches = [r for r in results if r.details.get("check") == "column_contract"
                          and r.status == CheckStatus.FAILED and "type is" in r.message.lower()]
            other = [r for r in results if r.status == CheckStatus.FAILED and r.column not in missing
                     and r not in mismatches]
            if missing:
                st.markdown("**Missing required columns:** " + ", ".join(f"`{c}`" for c in missing))
            if mismatches:
                st.markdown("**Type mismatches**")
                st.dataframe(pd.DataFrame([{"column": r.column, "expected": r.details["expected"]["type"],
                                            "actual": r.details["actual"]["type"], "message": r.message}
                                           for r in mismatches]), hide_index=True)
            if other:
                st.markdown("**Other contract violations**")
                st.dataframe(pd.DataFrame([{"target": r.target, "check": r.details.get("check"),
                                            "message": r.message} for r in other]), hide_index=True)
        with st.container(horizontal=True, gap="small"):
            st.download_button("Download schema report (JSON)", json.dumps(result.schema.to_dict(), indent=2, default=str),
                               icon=":material/download:", file_name=f"{result.dataset.name}.schema.json",
                               mime="application/json")
            st.download_button("Download as schema contract (JSON)", SchemaContract.from_schema_report(result.schema).to_json(),
                               icon=":material/download:", file_name=f"{result.dataset.name}.contract.json",
                               mime="application/json")
