import json
import random
import copy
from tqdm import tqdm
input_path = "/home/data_seung/qlora/restaurants_50k.jsonl"
output_path = "/dataset/yelp_open_dataset/50k_sikdang_to_10k_groups.json"




NUM_GROUPS = 10000 # 전체 group 개수
MIN_GROUP_SIZE = 5 
MAX_GROUP_SIZE = 20 # 주변 식당 개수 random(5~20)

random.seed(42)

# ---------------------------
# 1. 데이터 로드
# ---------------------------

# with open(input_path, "r", encoding="utf-8") as f: ## json 포멧 읽기
#     restaurants = json.load(f)

## jsonl 포멧 읽기
with open(input_path, "r", encoding="utf-8") as f:
    restaurants = [json.loads(line) for line in f]

num_restaurants = len(restaurants)
assert num_restaurants >= MAX_GROUP_SIZE, "식당 수가 너무 적습니다."

print(f"Loaded restaurants: {num_restaurants}")

# ---------------------------
# 2. 그룹 생성 (reviews 최대 5개)
# ---------------------------
groups = []

for i in tqdm(range(NUM_GROUPS)):
    group_size = random.randint(MIN_GROUP_SIZE, MAX_GROUP_SIZE)

    sampled_restaurants = random.sample(restaurants, group_size)
    group = []

    for r in sampled_restaurants:
        r_new = copy.deepcopy(r)   # 원본 보호
        # r_new["reviews"] = r_new.get("reviews", [])[:5] # 5개 리뷰만 사용
        r_new["reviews"] = r_new.get("reviews", []) # 전체 리뷰 사용

        group.append(r_new)

    groups.append(group)

print(f"Generated groups: {len(groups)}")

# ---------------------------
# 3. 저장
# ---------------------------
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        groups,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Saved → {output_path}")

