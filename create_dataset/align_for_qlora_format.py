import json

src = "/home/data_seung/qlora/response_10k.jsonl"
dst = "/home/data_seung/qlora/response_10k_noattr.jsonl"

with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        for r in obj.get("restaurants", []):
            if isinstance(r, dict) and r.get("attributes") is None:
                r.pop("attributes")
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


import json
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = (
        "You are a restaurant recommender.\n"
        "Rules:\n"
        "1) You MUST recommend 3 restaurants, chosen ONLY from the provided restaurants list.\n"
        "2) NEVER invent a restaurant or modify restaurant names.\n"
        "3) Output MUST be valid JSON only (no markdown, no extra text).\n"
        "4) JSON schema:\n"
        "{\n"
        '  "output": [\n'
        '    {"name": "<exact restaurant name from restaurants>", "reason": "<explanation>"},\n'
        '    {"name": "...", "reason": "..."},\n'
        '    {"name": "...", "reason": "..."}\n'
        "  ]\n"
        "}\n"
        "5) Reasons should primarily reflect the user query, and reference fields like distance_km, stars, attributes, review_count, and keywords when available.\n"
        "6) Write each reason as if a kind and real person is speaking naturally in conversation.\n"
    )


def _json_dumps(obj: Any) -> str:
    """Stable JSON string for prompts/outputs (no ASCII escaping)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_input_text(user_query: str, restaurants: List[Dict[str, Any]]) -> str:
    """
    QLoRA용 input 하나의 string으로 합치기:
    - system prompt를 포함
    - user payload는 JSON으로 포함
    """
    user_payload = {
        "query": user_query,
        "restaurants": restaurants,
    }

    # 학습 포맷은 자유지만, 보통 아래처럼 role을 명확히 넣는 게 안정적임
    # (학습 시 inference에서도 동일한 포맷으로 넣어주면 됨)
    text = (
        "### System:\n"
        + SYSTEM_PROMPT.strip()
        + "\n\n### User:\n"
        + _json_dumps(user_payload)
        + "\n"
    )
    return text


def build_output_text(output_list: Any) -> str:
    """
    output은 반드시 JSON only:
      {"output":[{"name":...,"reason":...}, ...]}
    """
    # 이미 list[dict] 형태라고 가정
    out_obj = {"output": output_list}
    return _json_dumps(out_obj)


def convert_jsonl(
    input_jsonl_path: str,
    output_jsonl_path: str,
    keep_metadata: bool = False,
    max_records: Optional[int] = None,
) -> None:
    """
    입력 jsonl(네 reco_results.jsonl)을 읽어서
    QLoRA 학습용 jsonl로 변환:
      - 필수 컬럼: input, output
      - 옵션: meta(원하면 group_index/prompt_index 등 보관)
    """
    n_in, n_out, n_skip = 0, 0, 0

    with open(input_jsonl_path, "r", encoding="utf-8") as fin, open(
        output_jsonl_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if max_records is not None and n_out >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            n_in += 1
            try:
                rec = json.loads(line)

                user_query = rec.get("input", None)
                restaurants = rec.get("restaurants", None)
                model_output = rec.get("output", None)

                if not isinstance(user_query, str) or not user_query.strip():
                    n_skip += 1
                    continue
                if not isinstance(restaurants, list) or len(restaurants) == 0:
                    n_skip += 1
                    continue
                if not isinstance(model_output, list) or len(model_output) != 3:
                    # 규칙 위반 레코드면 학습 데이터로 쓰지 않는게 안전
                    n_skip += 1
                    continue
                
                # ✅ business_id 제거
                cleaned_restaurants = []
                for r in restaurants:
                    if isinstance(r, dict):
                        r_clean = {k: v for k, v in r.items() if k != "business_id"}
                        cleaned_restaurants.append(r_clean)

                out_rec: Dict[str, Any] = {
                    "input": build_input_text(user_query.strip(), cleaned_restaurants),
                    "output": build_output_text(model_output),
                }

                if keep_metadata:
                    meta = {k: rec.get(k) for k in ["group_index", "prompt_index"] if k in rec}
                    if meta:
                        out_rec["meta"] = meta

                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n_out += 1

            except Exception:
                n_skip += 1
                continue

    print(f"[done] read={n_in}, wrote={n_out}, skipped={n_skip}")
    print(f"saved -> {output_jsonl_path}")


if __name__ == "__main__":

    convert_jsonl(
        input_jsonl_path="/home/data_seung/qlora/response_10k_noattr.jsonl",
        output_jsonl_path="/home/data_seung/qlora/response_10k_noattr_modify4qlora.jsonl",
        keep_metadata=False,   # 원하면 True
        max_records=None,      # 예: 10000처럼 제한 가능
    )