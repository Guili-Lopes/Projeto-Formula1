"""
src/data/validate_datasets.py
==============================
Validação de dados (Etapa 3 da reestruturação).

Executa as verificações previstas no plano:
    - quantidade de corridas por temporada e fonte;
    - temporadas parciais;
    - pilotos e equipes por corrida;
    - corridas duplicadas;
    - posições válidas;
    - DNF, DNS e DSQ;
    - comparação entre dados históricos e OpenF1 (2023+).

Uso:
    python -m src.data.validate_datasets [--report docs/relatorio_validacao_dados.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_pipeline import load_seasons
from src.data.repository import (
    OPENF1_FIRST_YEAR,
    _openf1_records_for_year,
    load_openf1_race_results,
)
from src.data_openf1.race_mapping import canonical_race_key

DATA_DIR = _PROJECT_ROOT / "data"
MANIFESTS_DIR = DATA_DIR / "openf1" / "manifests"
OVERLAP_YEARS = [2023, 2024, 2025]


def _records_by_key(records):
    return {(r.season, canonical_race_key(r.race)): r for r in records}


def validate(top_k: int = 10) -> dict:
    """Roda todas as verificações e devolve um dicionário estruturado."""
    out: dict = {"years": {}, "comparison": {}, "issues": [], "summary": {}}

    # ── 1. Integridade por temporada e fonte ─────────────────────────────
    for year in range(2019, 2026):
        info: dict = {}
        legacy = load_seasons(str(DATA_DIR), [year], top_k=top_k)
        info["legacy_races"] = len(legacy)
        dup = [k for k, v in Counter(r.race for r in legacy).items() if v > 1]
        if dup:
            out["issues"].append(f"{year}: corridas duplicadas no legado: {dup}")

        if year >= OPENF1_FIRST_YEAR:
            df = load_openf1_race_results(year)
            recs = _openf1_records_for_year(year, top_k, DATA_DIR)
            info["openf1_races"] = len(recs)
            manifest_path = MANIFESTS_DIR / f"sync_{year}.json"
            if manifest_path.exists():
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                info["openf1_status"] = m.get("status")
            if not df.empty:
                per_race = df.groupby("session_key").size()
                info["drivers_per_race_min"] = int(per_race.min())
                info["drivers_per_race_max"] = int(per_race.max())
                short = per_race[per_race < 20]
                if len(short):
                    names = (
                        df[df["session_key"].isin(short.index)]
                        .groupby("session_key")["meeting_name"].first().tolist()
                    )
                    out["issues"].append(
                        f"{year}: corridas com menos de 20 pilotos na OpenF1: "
                        f"{sorted(set(names))}"
                    )
                # posições válidas: 1..n únicas entre classificados
                for sk, g in df.groupby("session_key"):
                    pos = g.loc[g["classified"], "position_num"].dropna()
                    if pos.duplicated().any():
                        out["issues"].append(
                            f"{year}/{g['meeting_name'].iloc[0]}: posição "
                            "duplicada entre classificados (OpenF1)."
                        )
                info["dnf"] = int(df["dnf"].sum())
                info["dns"] = int(df["dns"].sum())
                info["dsq"] = int(df["dsq"].sum())
        out["years"][year] = info

    # ── 2. Comparação legado × OpenF1 nas temporadas de transição ────────
    total_races = matched = 0
    ranking_equal = classified_equal = tail_set_equal = 0
    diffs: list[str] = []

    for year in OVERLAP_YEARS:
        legacy = _records_by_key(load_seasons(str(DATA_DIR), [year], top_k=top_k))
        openf1 = _records_by_key(_openf1_records_for_year(year, top_k, DATA_DIR))

        only_legacy = sorted(k[1] for k in legacy.keys() - openf1.keys())
        only_openf1 = sorted(k[1] for k in openf1.keys() - legacy.keys())
        if only_legacy:
            diffs.append(f"{year}: apenas no legado → {only_legacy}")
        if only_openf1:
            diffs.append(f"{year}: apenas na OpenF1 → {only_openf1}")

        for key in sorted(legacy.keys() & openf1.keys()):
            total_races += 1
            lr, orr = legacy[key], openf1[key]
            matched += 1
            l_cls = lr.ranking[: lr.n_classified]
            o_cls = orr.ranking[: orr.n_classified]
            l_tail = lr.ranking[lr.n_classified:]
            o_tail = orr.ranking[orr.n_classified:]

            if l_cls == o_cls:
                classified_equal += 1
            else:
                diffs.append(
                    f"{key[0]} {key[1]}: classificados divergem\n"
                    f"      legado: {l_cls}\n      openf1: {o_cls}"
                )
            if set(l_tail) == set(o_tail):
                tail_set_equal += 1
            else:
                diffs.append(
                    f"{key[0]} {key[1]}: conjunto de DNFs diverge "
                    f"(legado={sorted(l_tail)}, openf1={sorted(o_tail)})"
                )
            if lr.ranking == orr.ranking:
                ranking_equal += 1

    out["comparison"] = {
        "years": OVERLAP_YEARS,
        "races_compared": total_races,
        "classified_identical": classified_equal,
        "dnf_set_identical": tail_set_equal,
        "full_ranking_identical": ranking_equal,
        "differences": diffs,
    }

    out["summary"] = {
        "ok": not out["issues"] and classified_equal == total_races
        and tail_set_equal == total_races,
        "n_issues": len(out["issues"]),
        "n_differences": len(diffs),
    }
    return out


def render_report(result: dict) -> str:
    lines = [
        "# Relatório de Validação de Dados — Etapa 3",
        "",
        "Comparação entre o dataset histórico e os dados OpenF1 processados,",
        "conforme a Etapa 3 do plano de reestruturação.",
        "",
        "## Corridas por temporada e fonte",
        "",
        "| Ano | Legado | OpenF1 | Status OpenF1 | Pilotos/corrida | DNF | DNS | DSQ |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for year, info in result["years"].items():
        pil = ""
        if "drivers_per_race_min" in info:
            pil = f"{info['drivers_per_race_min']}–{info['drivers_per_race_max']}"
        lines.append(
            f"| {year} | {info.get('legacy_races', '')} "
            f"| {info.get('openf1_races', '—')} "
            f"| {info.get('openf1_status', '—')} | {pil} "
            f"| {info.get('dnf', '—')} | {info.get('dns', '—')} "
            f"| {info.get('dsq', '—')} |"
        )

    c = result["comparison"]
    lines += [
        "",
        "## Comparação legado × OpenF1 (2023–2025)",
        "",
        f"- Corridas comparadas: **{c['races_compared']}**",
        f"- Ordem dos classificados idêntica: **{c['classified_identical']}/{c['races_compared']}**",
        f"- Conjunto de DNFs idêntico: **{c['dnf_set_identical']}/{c['races_compared']}**",
        f"- Ranking completo (incl. ordem da cauda de DNFs) idêntico: "
        f"**{c['full_ranking_identical']}/{c['races_compared']}**",
        "",
    ]
    if c["differences"]:
        lines += ["### Diferenças encontradas", ""]
        for d in c["differences"]:
            lines.append(f"- {d}")
        lines.append("")
    if result["issues"]:
        lines += ["## Ocorrências de qualidade", ""]
        for issue in result["issues"]:
            lines.append(f"- {issue}")
        lines.append("")
    lines += [
        "## Conclusão",
        "",
        "✅ Fontes equivalentes para migração." if result["summary"]["ok"]
        else "⚠️ Há diferenças documentadas acima; avaliar antes de remover o legado.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="docs/relatorio_validacao_dados.md")
    args = parser.parse_args(argv)

    result = validate()
    report = render_report(result)
    path = _PROJECT_ROOT / args.report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    c = result["comparison"]
    print(f"Corridas comparadas: {c['races_compared']}")
    print(f"Classificados idênticos: {c['classified_identical']}")
    print(f"DNF set idêntico: {c['dnf_set_identical']}")
    print(f"Ranking completo idêntico: {c['full_ranking_identical']}")
    print(f"Diferenças: {len(c['differences'])} | Issues: {len(result['issues'])}")
    print(f"Relatório: {path}")
    return 0 if result["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
