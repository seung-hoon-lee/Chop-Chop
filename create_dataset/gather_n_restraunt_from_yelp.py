import json
import random
from pathlib import Path
from tqdm import tqdm 

BUSINESS_PATH = "/dataset/yelp_open_dataset/yelp_academic_dataset_business.json"
REVIEW_PATH   = "/dataset/yelp_open_dataset/yelp_academic_dataset_review.json"

OUT_JSONL = "restaurants_50k.jsonl"   
OUT_JSON  = None                    

MAX_RESTAURANTS = 50000
MAX_REVIEWS_PER_RESTAURANT = 50  
random.seed(42)  

def is_restaurant(biz: dict) -> bool:
    """
    '식당' 필터가 필요하면 여기서 조정.
    현재는 categories에 'Restaurants'가 포함된 경우만 True.
    (너무 엄격/느슨하면 조건을 바꾸면 됨)
    """
    cats = biz.get("categories") or ""
    return "Restaurants" in cats

def load_businesses(business_path: str, limit: int):
    """
    business.json에서 식당 business를 최대 limit개 뽑아 dict로 저장.
    """
    businesses = {}
    with open(business_path, "r", encoding="utf-8") as f:
        for line in f:
            biz = json.loads(line)
            if not is_restaurant(biz):
                continue

            bid = biz["business_id"]
            businesses[bid] = {
                "business_id": bid,
                "name": biz.get("name"),
                "stars": biz.get("stars"),
                "review_count": biz.get("review_count"),
                "attributes": biz.get("attributes"),
                "reviews": [],
                "distance_km": round(random.uniform(0, 10), 2),
            }
            print(len(businesses))
            if len(businesses) >= limit:
                break
    return businesses

def attach_reviews(review_path: str, businesses: dict, max_reviews_per_restaurant: int):
    """
    review.json에서 business_id가 businesses에 있는 것만 reviews에 누적.
    """
    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            r = json.loads(line)
            bid = r.get("business_id")
            if bid not in businesses:
                continue

            # 리뷰 텍스트(원하면 stars/date/user_id 등도 같이 저장 가능)
            if len(businesses[bid]["reviews"]) < max_reviews_per_restaurant:
                businesses[bid]["reviews"].append(r.get("text"))

def save_jsonl(out_path: str, businesses: dict):
    with open(out_path, "w", encoding="utf-8") as f:
        for item in businesses.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def save_json(out_path: str, businesses: dict):
    data = list(businesses.values())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("1) Loading businesses...")
    businesses = load_businesses(BUSINESS_PATH, MAX_RESTAURANTS)
    print(f"   Loaded: {len(businesses)} restaurants")

    print("2) Attaching reviews...")
    attach_reviews(REVIEW_PATH, businesses, MAX_REVIEWS_PER_RESTAURANT)

    # 간단 통계
    with_reviews = sum(1 for b in businesses.values() if b["reviews"])
    print(f"   Restaurants with >=1 review attached: {with_reviews}/{len(businesses)}")

    print("3) Saving...")
    if OUT_JSONL:
        save_jsonl(OUT_JSONL, businesses)
        print(f"   Saved JSONL to: {OUT_JSONL}")
    if OUT_JSON:
        save_json(OUT_JSON, businesses)
        print(f"   Saved JSON to: {OUT_JSON}")

if __name__ == "__main__":
    main()