import csv
import json
from html import escape
from pathlib import Path

from .benchmark import Example
from .media import render_preview


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _asset(name: str) -> str:
    return Path(__file__).with_name(name).read_text(encoding="utf-8")


def _fmt(value: float | None, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if value is not None else "—"


def _percent(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "—"


def _table(headers: tuple[str, ...], rows: list[tuple]) -> str:
    head = "".join(f"<th scope='col'>{escape(name)}</th>" for name in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _document(title: str, body: str, script: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>{_asset("report.css")}</style>
</head>
<body><main>{body}</main>{script}</body>
</html>
"""


def _examples(output: Path, examples: list[Example]) -> str:
    cards = []
    for number, example in enumerate(examples, start=1):
        filename = f"clip-{number:02d}.gif"
        sampled = example.indices
        render_preview(
            [example.clip.images[index] for index in sampled],
            [index / example.clip.fps for index in sampled],
            example.result, output / "examples" / filename,
            tuple(example.clip.truth[index] for index in sampled),
        )
        score = example.metrics
        before = abs(example.baseline - score["reference_count"])
        after = score["absolute_error"]
        outcome = "Improved count" if after < before else "Worse count" if after > before else "Same count error"
        cards.append(f"""
        <figure class="clip">
          <a href="examples/{filename}"><img src="examples/{filename}" alt="Tracking replay with colored boxes and trails" loading="lazy"></a>
          <figcaption>
            <div class="clip-title"><strong>{escape(example.clip.id)}</strong><span>{outcome}</span></div>
            <dl class="counts">
              <div><dt>Reference</dt><dd>{score['reference_count']}</dd></div>
              <div><dt>Max frame</dt><dd>{example.baseline}</dd></div>
              <div><dt>{escape(score['method'].capitalize())}</dt><dd>{example.result.count}</dd></div>
            </dl>
            <small>{example.fps:g} FPS · {score['sampling_misses']} absent from sampled frames ·
            {score['fragmented_ids']} fragmented IDs · {score['merged_tracks']} merged tracks</small>
          </figcaption>
        </figure>""")
    return "".join(cards)


def write_benchmark(
    output: Path, metadata: dict, rows: list[dict], summary: list[dict],
    intervals: list[dict], slices: list[dict], examples: list[Example],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_clip.csv", rows)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "paired.csv", intervals)
    _write_csv(output / "slices.csv", slices)
    write_json(output / "run.json", {"metadata": metadata, "summary": summary, "paired": intervals, "slices": slices})
    detector = [row for row in summary if row["source"] == "detector"]
    best = min(detector, key=lambda row: (row["mae"], row["p95_ms"]))
    source = escape(metadata["dataset"])
    method_table = _table(
        ("Boxes", "Method", "FPS", "MAE", "Under", "Over", "Bias", "IDF1", "p95 ms", "Missed by sampling"),
        [
            (
                row["source"], row["method"], f'{row["fps"]:g}', _fmt(row["mae"]),
                _percent(row["under_rate"]), _percent(row["over_rate"]), _fmt(row["bias"]),
                _percent(row["idf1"]),
                _fmt(row["p95_ms"], 1) if row["source"] == "detector" else "—",
                _fmt(row["mean_sampling_misses"]),
            )
            for row in summary
        ],
    )
    intervals_table = _table(
        ("Boxes", "Method", "FPS", "Δ MAE", "95% CI", "Sites"),
        [
            (
                row["source"], row["method"], f'{row["fps"]:g}', _fmt(row["delta_mae"]),
                f'{_fmt(row["low"])} to {_fmt(row["high"])}' if row["low"] is not None else "—",
                row["sites"],
            )
            for row in intervals
        ],
    )
    chart_data = json.dumps(detector, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    label = "Synthetic verification" if metadata["dataset"] == "synthetic fixture" else "SA-FARI benchmark"
    caveat = (
        "Generated scenes and a color-marker detector. These scores verify the pipeline on controlled inputs."
        if metadata["dataset"] == "synthetic fixture"
        else "Counts refer to annotated appearances within a clip. SA-FARI contains clips with animals."
    )
    timing = (
        "Timing includes JSON lookup, frame decode, optional features and association."
        if metadata["detector_mode"] == "cached"
        else "Timing sums frame decode, detector, optional features and association."
    )
    clips = _examples(output, examples)
    evidence = f"""
      <section><h2>Inspect the tracks</h2>
        <p class="muted">Colored boxes and trails show predicted identities. White outlines show annotated boxes.</p>
        <div class="examples">{clips}</div>
      </section>""" if clips else ""
    body = f"""
      <header>
        <div class="topline"><span class="wordmark">TrapCount</span><span class="status">{label}</span></div>
        <h1>Wildlife counting benchmark</h1>
        <p class="lede">How much counting accuracy survives when we process fewer frames?</p>
        <p class="notice">{caveat}</p>
        <p class="muted">{source} · {escape(metadata['split'])} split · {escape(metadata['detector'])} · {escape(metadata['device'])}</p>
      </header>
      <div class="stats">
        <div><span>Clips evaluated</span><strong>{best['clips']}</strong></div>
        <div><span>Camera locations</span><strong>{best['sites']}</strong></div>
        <div><span>Lowest observed MAE</span><strong>{_fmt(best['mae'])}</strong></div>
        <div><span>Method at lowest MAE</span><strong>{escape(best['method'])} · {best['fps']:g} FPS</strong></div>
      </div>
      <section>
        <h2>Counting error and processing time</h2>
        <p class="muted">One point per method and frame rate. Lower and farther left is better.
        {timing} Model load and report rendering are excluded.</p>
        <div class="legend" aria-label="Toggle methods on the plot">
          <button type="button" data-method="max-frame" aria-pressed="true">Max frame</button>
          <button type="button" data-method="geometry" aria-pressed="true">Geometry</button>
          <button type="button" data-method="appearance" aria-pressed="true">Appearance</button>
        </div>
        <canvas id="tradeoff" height="310" aria-label="Scatter plot of p95 processing time versus count MAE"></canvas>
      </section>
      <section><h2>Results by frame rate</h2>{method_table}
        <p class="muted">Oracle rows reuse annotated boxes from the selected frames. The counting target remains the full clip at every frame rate.</p>
      </section>
      <section><h2>Paired change from max frame</h2>{intervals_table}
        <p class="muted">Negative values mean lower error. Confidence intervals resample camera locations and require at least five locations.</p>
      </section>
      {evidence}
      <footer>
        <strong>Run artifacts</strong>
        <div class="links"><a href="per_clip.csv">Per clip</a><a href="summary.csv">Summary</a>
        <a href="slices.csv">Evaluation slices</a><a href="paired.csv">Paired comparisons</a>
        <a href="manifest.csv">Data manifest</a><a href="run.json">Configuration and metrics</a></div>
      </footer>
    """
    script = f'<script id="chart-data" type="application/json">{chart_data}</script><script>{_asset("chart.js")}</script>'
    (output / "index.html").write_text(_document("TrapCount · Benchmark", body, script), encoding="utf-8")


def write_inference(output: Path, payload: dict) -> None:
    write_json(output / "tracks.json", payload)
    body = f"""
      <header>
        <div class="topline"><span class="wordmark">TrapCount</span><span class="status">Clip analysis</span></div>
        <h1>Wildlife tracks</h1>
        <p class="lede">{escape(payload['input'])}</p>
        <p class="muted">{escape(payload['detector'])} · {escape(payload['method'])} · {payload['fps']:g} FPS</p>
      </header>
      <div class="stats">
        <div><span>Estimated animals</span><strong>{payload['count']}</strong></div>
        <div><span>Frames processed</span><strong>{payload['sampled_frames']}</strong></div>
      </div>
      <section class="single-preview"><h2>Clip replay</h2>
      <img src="tracks.gif" alt="Animated replay of predicted tracks">
      <p class="muted">Replay shows {payload['preview_frames']} of {payload['sampled_frames']} processed frames.</p>
      </section>
      <footer><div class="links"><a href="tracks.json">Track observations</a><a href="tracks.gif">Animated replay</a></div></footer>
    """
    (output / "index.html").write_text(_document("TrapCount · Clip", body), encoding="utf-8")
