
############################################## keybert로 review 핵심 키워드 추출 #########################################
from tqdm import tqdm
import json
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer  # (추가)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_keywords_for_groups(
    input_path: str,
    output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    top_n: int = 15,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: Optional[str] = "english",
    use_mmr: bool = True,
    diversity: float = 0.6,
    nr_candidates: int = 60,
    min_chars: int = 30,
    store_scores: bool = False,
    device: str = "cuda",  # (추가) GPU 사용
) -> None:
    """
    입력 JSON 형식:
      data = [ group1, group2, ... ]
      group = [ business1, business2, ... ]
      business = { ..., "reviews": [str, str, ...], ... }

    출력 JSON 형식(그룹 유지):
      data_out = [ group1_out, group2_out, ... ]
      group_out = [ business_out, business_out, ... ]
      business_out에 "keywords" (또는 "keywords_scored")를 추가
      + 저장 시 reviews, review_count 제거
    """
    data = load_json(input_path)

    if not isinstance(data, list) or (len(data) > 0 and not isinstance(data[0], list)):
        raise ValueError("Input JSON must be a list of groups, where each group is a list of business dicts.")

    # (수정) SentenceTransformer를 GPU로 올리고 KeyBERT에 주입
    embedder = SentenceTransformer(model_name, device=device)
    kw_model = KeyBERT(model=embedder)

    out: List[List[Dict[str, Any]]] = []

    for gi, group in tqdm(enumerate(data)):
        group_out: List[Dict[str, Any]] = []
        for bi, biz in enumerate(group):
            if not isinstance(biz, dict):
                raise ValueError(f"Business item must be dict. Found: {type(biz)} at group={gi}, index={bi}")

            biz_out = deepcopy(biz)

            # (추가) 저장 시 제거할 필드: reviews, review_count
            reviews = biz_out.pop("reviews", [])        # doc 생성에는 사용
            # biz_out.pop("review_count", None)           # 저장에서는 제거 => 02.22 : review_count도 쓰자


            # 리뷰들을 하나의 doc으로 합쳐서 business-level 키워드 뽑기
            doc = "\n".join([r.strip() for r in reviews if isinstance(r, str) and r.strip()])

            kws = kw_model.extract_keywords(
                doc,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,
                top_n=top_n,
                use_mmr=use_mmr,
                diversity=diversity,
                nr_candidates=nr_candidates,
            )

            # kws: List[Tuple[str, float]]
            if store_scores:
                biz_out["keywords_scored"] = [{"kw": k, "score": float(s)} for k, s in kws]
            biz_out["keywords"] = [k for k, _ in kws]

            group_out.append(biz_out)
        out.append(group_out)

    save_json(out, output_path)


if __name__ == "__main__":
    extract_keywords_for_groups(
        input_path="/dataset/yelp_open_dataset/50k_sikdang_to_10k_groups.json",
        output_path="10k_groups_post_keybert.json",
        model_name="all-MiniLM-L6-v2",
        top_n=20,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_mmr=True,
        diversity=0.6,
        nr_candidates=80,
        store_scores=False,   # True로 하면 keywords_scored도 저장
        device="cuda",        # (추가) GPU
    )

