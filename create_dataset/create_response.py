import os
import json
import time
from typing import Any, Dict, List, Union, Optional
from tqdm import tqdm
import torch
from openai import OpenAI

import os
os.environ["OPENAI_API_KEY"] = NULL # Your API KEy


# ---------------------------
# IO
# ---------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_prompts_from_pth(path: str) -> List[str]:
    """
    user_prompt.pth 안에 list[str]가 바로 들어있으면 그대로 사용.
    dict/tuple/기타면 내부에서 string list처럼 보이는 걸 최대한 찾아봄.
    """
    obj = torch.load(path, map_location="cpu")

    # 1) 가장 흔한 케이스: list[str]
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return obj

    # 2) dict 안에 list[str]가 있는 케이스
    if isinstance(obj, dict):
        # 흔한 키들 우선 탐색
        for k in ["prompts", "prompt", "queries", "query", "inputs", "input", "user_prompts", "user_prompt", "responses"]:
            v = obj.get(k, None)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v

        # 아니면 dict의 value들 중 list[str] 찾아보기
        for v in obj.values():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v

    # 3) tuple 안에 list[str]가 있는 케이스
    if isinstance(obj, tuple):
        for v in obj:
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v

    raise ValueError(
        "user_prompt.pth에서 list[str] 형태의 질문 리스트를 찾지 못했습니다. "
        "pth 내부 구조를 확인해 주세요."
    )


# ---------------------------
# Restaurant payload shaping
# ---------------------------
def compact_restaurant(biz: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 입력/저장용으로 'reviews', 'review_count' 등 큰 필드는 제거하고 핵심만 남김.
    business_id는 이미 제거된 파일이라고 하셨지만, 혹시 몰라서 다시 pop 처리.
    """
    out = dict(biz)
    # out.pop("business_id", None)
    # out.pop("reviews", None)
    # out.pop("review_count", None) # 전부 쓸거 (2026.02.24)
    return out


# ---------------------------
# LLM call
# ---------------------------
def build_messages(user_query: str, restaurants: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = (
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

    user = {
        "query": user_query,
        "restaurants": restaurants
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def call_recommendation(
    client: OpenAI,
    model: str,
    user_query: str,
    restaurants: List[Dict[str, Any]],
    max_retries: int = 3,
    temperature: float = 0.2,
) -> List[Dict[str, str]]:
    """
    모델 응답을 JSON으로 파싱해 output 리스트를 반환.
    파싱 실패/규칙 위반 시 재시도.
    """
    last_err: Optional[str] = None

    for attempt in range(max_retries):
        try:
            messages = build_messages(user_query, restaurants)

            resp = client.chat.completions.create(
                model=model,
                messages=messages
            )

            content = resp.choices[0].message.content
            data = json.loads(content)

            # 검증: output 존재, 길이 3
            # if "output" not in data or not isinstance(data["output"], list) or len(data["output"]) != 3:
            #     raise ValueError("Invalid JSON schema or not exactly 3 recommendations.")

            if "output" not in data or not isinstance(data["output"], list):
                raise ValueError("Invalid JSON schema")
            

            # 검증: name이 restaurants 안에 있는지
            valid_names = set(r.get("name") for r in restaurants if isinstance(r, dict))
            for item in data["output"]:
                if not isinstance(item, dict) or "name" not in item or "reason" not in item:
                    raise ValueError("Each output item must have name and reason.")
                if item["name"] not in valid_names:
                    raise ValueError(f"Invented/unknown restaurant name: {item['name']}")

            return data["output"]

        except Exception as e:
            last_err = str(e)
            # 간단한 backoff
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Failed after retries. Last error: {last_err}")


# ---------------------------
# Main pipeline
# ---------------------------
def run(
    groups_json_path: str,
    prompts_pth_path: str,
    output_jsonl_path: str,
    model: str = "gpt-5-mini",
    temperature: float = 0.2,
    compact: bool = True,
    n_prompts: Optional[int] = None,  # 추가: 몇 개 prompt만 쓸지
) -> None:
    client = OpenAI()  # OPENAI_API_KEY 환경변수 필요

    groups = load_json(groups_json_path)
    prompts = load_prompts_from_pth(prompts_pth_path)

    if not isinstance(groups, list) or (len(groups) > 0 and not isinstance(groups[0], list)):
        raise ValueError("groups json must be: [ [restaurant_dict, ...], [ ... ], ... ]")

    # ---- 여기부터 핵심: prompt를 그룹 개수만큼만 사용 ----
    # 기본 동작: 그룹 개수만큼만 prompt 사용
    k = len(groups) if n_prompts is None else min(n_prompts, len(groups))
    prompts_k = [p.strip() for p in prompts if isinstance(p, str) and p.strip()][:k]
    groups_k = groups[:len(prompts_k)]  # prompt 수에 맞춰 그룹도 자름
    records: List[Dict[str, Any]] = []

    for gi, (q, group) in tqdm(enumerate(zip(prompts_k, groups_k))):
        if not isinstance(group, list):
            continue

        restaurants = [compact_restaurant(x) for x in group if isinstance(x, dict)] if compact else group
        if len(restaurants) == 0:
            continue

        output = call_recommendation(
            client=client,
            model=model,
            user_query=q,
            restaurants=restaurants,
            temperature=temperature,
        )

        record = {
            "input": q,
            "restaurants": restaurants,
            "output": output,
            "group_index": gi,
            "prompt_index": gi,  # 1:1 매칭이라 gi로 둠
        }
        records.append(record)

        # 중간 저장
        save_jsonl(records, output_jsonl_path)

    save_jsonl(records, output_jsonl_path)

if __name__ == "__main__":
    run(
        groups_json_path="/home/data_seung/qlora/10k_groups_post_keybert.json",
        prompts_pth_path="/home/data_seung/qlora/user_prompt_10k.pth",
        output_jsonl_path="/home/data_seung/qlora/response_10k.jsonl",
        model="gpt-5-mini",      # 필요 시 변경
        temperature=0.2,
        compact=False,            # restaurants에 reviews/review_count 제거해서 저장/전송 (권장)
    )