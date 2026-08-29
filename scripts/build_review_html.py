"""生成离线人工复核页面，并导出兼容决策工具的 JSONL。"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Any

try:
    from scripts.build_label_review_sheets import _resolve_image
except ImportError:  # direct execution: python scripts/build_review_html.py
    from build_label_review_sheets import _resolve_image


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("error review must contain a non-empty items array")
    return items


def _image_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _checks(options: list[str], selected: list[str], field: str) -> str:
    selected_values = set(selected)
    return "".join(
        '<label><input type="checkbox" data-field="{}" value="{}"{}> {}</label>'.format(
            _escape(field),
            _escape(option),
            " checked" if option in selected_values else "",
            _escape(option),
        )
        for option in options
    )


def build_review_html(
    *,
    error_review_path: Path,
    image_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build a self-contained offline HTML review page."""
    raw_items = _load_items(error_review_path)
    case_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    cards: list[str] = []
    events = [
        "vru_near_maneuver_path",
        "vehicle_near_maneuver_path",
        "fixed_obstacle_near_path",
        "narrow_passage",
        "visibility_occlusion",
        "parking_space_conflict",
    ]
    advice = [
        "maintain_observation",
        "slow_down",
        "yield",
        "prepare_to_stop",
        "change_maneuver_when_safe",
    ]
    risks = ["low", "medium", "high"]

    for index, item in enumerate(raw_items, start=1):
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("error review case_id must be a non-blank string")
        if case_id in case_ids:
            raise ValueError(f"duplicate error review case_id: {case_id}")
        case_ids.add(case_id)
        image_path = _resolve_image({"image_ref": item.get("image_ref")}, image_root)
        candidate = item.get("candidate_assessment")
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate assessment must be an object: {case_id}")
        cases.append({"case_id": case_id, "candidate_assessment": candidate})
        model = item.get("model_assessment")
        model_text = (
            "model: FAIL | " + str((item.get("failure") or {}).get("category", "unknown"))
            if model is None
            else "model: {} | {}".format(
                model.get("risk_level", "unknown"),
                ",".join(model.get("events", [])) or "no_event",
            )
        )
        candidate_text = "candidate: {} | {}".format(
            candidate.get("risk_level", "unknown"),
            ",".join(candidate.get("events", [])) or "no_event",
        )
        risk_options = "".join(
            '<option value="{}"{}>{}</option>'.format(
                _escape(risk),
                " selected" if risk == candidate.get("risk_level") else "",
                risk,
            )
            for risk in risks
        )
        raw_output = item.get("raw_output") or ""
        cards.append(
            _CARD_TEMPLATE.format(
                index=index,
                case_id=_escape(case_id),
                priority=_escape(item.get("review_priority", "unknown")),
                priority_class=_escape(item.get("review_priority", "unknown")),
                image_uri=_escape(_image_data_uri(image_path)),
                split=_escape(item.get("split")),
                source_group=_escape(item.get("source_group_id")),
                candidate_text=_escape(candidate_text),
                model_text=_escape(model_text),
                risk_options=risk_options,
                event_checks=_checks(events, candidate.get("events", []), "events"),
                advice_checks=_checks(advice, candidate.get("driver_advice", []), "advice"),
                evidence=_escape("\n".join(candidate.get("evidence", []))),
                raw_output=_escape(raw_output),
            )
        )

    document = _PAGE_TEMPLATE.replace("__CARDS__", "\n".join(cards)).replace(
        "__CASES__", _json_for_script(cases)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "sample_count": len(cases),
        "embedded_image_count": len(cases),
        "ready_for_review": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error-review", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_review_html(
                error_review_path=args.error_review,
                image_root=args.image_root,
                output_path=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


_CARD_TEMPLATE = """<article class=\"card {priority_class}\" data-case-id=\"{case_id}\" data-priority=\"{priority}\">
<div class=\"card-head\"><strong>{index}. {case_id}</strong><span>priority: {priority}</span></div>
<div class=\"card-body\">
<div><img class=\"scene\" src=\"{image_uri}\" alt=\"{case_id}\"></div>
<div>
<div class=\"context\">split: {split}
source_group: {source_group}
{candidate_text}
{model_text}</div>
<label>复核状态 <select data-field=\"status\"><option value=\"pending\">pending</option><option value=\"confirmed\">confirmed</option><option value=\"corrected\">corrected</option></select></label>
<fieldset><legend>人工确认风险等级</legend><select data-field=\"risk\">{risk_options}</select></fieldset>
<fieldset><legend>人工确认事件</legend><div class=\"checks\">{event_checks}</div></fieldset>
<fieldset><legend>人工确认驾驶建议</legend><div class=\"checks\">{advice_checks}</div></fieldset>
<label>可见证据（每行一条）<textarea data-field=\"evidence\">{evidence}</textarea></label>
<label>复核说明<textarea data-field=\"note\" placeholder=\"corrected 必填；confirmed 可填写确认依据\"></textarea></label>
<details><summary>查看模型原始输出</summary><pre>{raw_output}</pre></details>
</div>
</div>
</article>"""


_PAGE_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ParkSight-VLM 人工复核</title>
<style>
body { margin: 0; background: #f3f4f6; color: #17202a; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
header { position: sticky; top: 0; z-index: 2; padding: 12px 18px; background: #17202a; color: white; box-shadow: 0 2px 8px #0003; }
h1 { margin: 0 0 8px; font-size: 20px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
button, select { border: 1px solid #aab4bd; border-radius: 5px; padding: 6px 9px; background: white; font: inherit; }
button { cursor: pointer; }
button.primary { background: #2e75b6; color: white; border-color: #2e75b6; }
#summary { margin-left: auto; font-size: 14px; }
main { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; padding: 16px; }
.card { background: white; border: 1px solid #d5dbe0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px #0001; }
.card.high { border: 2px solid #c0392b; }
.card-head { padding: 9px 12px; display: flex; justify-content: space-between; gap: 10px; background: #eef2f5; }
.card-head strong { overflow-wrap: anywhere; }
.card-head span { font-weight: 700; }
.card.high .card-head span { color: #b42318; }
.card-body { display: grid; grid-template-columns: minmax(230px, 42%) 1fr; gap: 12px; padding: 12px; }
.scene { width: 100%; max-height: 330px; object-fit: contain; background: #111; }
.context { font-size: 13px; color: #4d5963; margin: 4px 0 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
fieldset { border: 1px solid #d5dbe0; border-radius: 5px; margin: 7px 0; padding: 7px; }
legend { font-size: 12px; color: #53606b; }
label { display: block; font-size: 13px; margin: 5px 0; }
.checks { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
textarea { box-sizing: border-box; width: 100%; min-height: 62px; resize: vertical; font: 13px/1.35 "Segoe UI", "Microsoft YaHei", sans-serif; }
details { margin-top: 7px; font-size: 12px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 180px; overflow: auto; background: #f5f6f7; padding: 6px; }
.hidden { display: none; }
@media (max-width: 700px) { main { grid-template-columns: 1fr; padding: 8px; } .card-body { grid-template-columns: 1fr; } #summary { margin-left: 0; } }
</style>
</head>
<body>
<header>
<h1>ParkSight-VLM 80 条候选标注人工复核</h1>
<div class="toolbar">
<label>筛选优先级 <select id="priority-filter"><option value="all">全部</option><option value="high">high</option><option value="medium">medium</option><option value="low">low</option></select></label>
<button id="confirm-visible">将当前显示项标为 confirmed</button>
<button class="primary" id="download">下载已完成决策 JSONL</button>
<span id="summary"></span>
</div>
</header>
<main id="cases">__CARDS__</main>
<script type="application/json" id="review-data">__CASES__</script>
<script>
const CASES = JSON.parse(document.getElementById("review-data").textContent);
function visibleCards() { return [...document.querySelectorAll(".card:not(.hidden)")]; }
function updateSummary() {
  const counts = {pending: 0, confirmed: 0, corrected: 0};
  document.querySelectorAll(".card").forEach(card => counts[card.querySelector('[data-field="status"]').value]++);
  document.getElementById("summary").textContent = `共 ${CASES.length} 条 | pending ${counts.pending} | confirmed ${counts.confirmed} | corrected ${counts.corrected}`;
}
document.getElementById("priority-filter").addEventListener("change", event => {
  document.querySelectorAll(".card").forEach(card => card.classList.toggle("hidden", event.target.value !== "all" && card.dataset.priority !== event.target.value));
});
document.getElementById("confirm-visible").addEventListener("click", () => {
  visibleCards().forEach(card => card.querySelector('[data-field="status"]').value = "confirmed");
  updateSummary();
});
document.querySelectorAll('[data-field="status"]').forEach(select => select.addEventListener("change", updateSummary));
function readAssessment(card) {
  return {
    schema_version: "parking_risk_v1",
    risk_level: card.querySelector('[data-field="risk"]').value,
    events: [...card.querySelectorAll('[data-field="events"]:checked')].map(input => input.value),
    evidence: card.querySelector('[data-field="evidence"]').value.split("\n").map(value => value.trim()).filter(Boolean),
    driver_advice: [...card.querySelectorAll('[data-field="advice"]:checked')].map(input => input.value)
  };
}
document.getElementById("download").addEventListener("click", () => {
  const output = [...document.querySelectorAll(".card")].filter(card => {
    const status = card.querySelector('[data-field="status"]').value;
    return status === "confirmed" || status === "corrected";
  }).map(card => ({
    case_id: card.dataset.caseId,
    review_status: card.querySelector('[data-field="status"]').value,
    human_assessment: readAssessment(card),
    review_note: card.querySelector('[data-field="note"]').value
  }));
  if (!output.length) { alert("还没有已完成的复核决策。"); return; }
  const blob = new Blob([output.map(row => JSON.stringify(row)).join("\n") + "\n"], {type: "application/x-ndjson;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "review_decisions_batch.jsonl";
  link.click();
  URL.revokeObjectURL(link.href);
});
updateSummary();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
