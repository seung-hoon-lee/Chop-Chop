import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import os
os.environ["OPENAI_API_KEY"] = NULL

BASE_MODEL_PATH = "/home/data_seung/llm-awq/Meta-Llama-3-8B"
ADAPTER_PATH = "/home/data_seung/qlora/output/llama3-8b-restaurant_10kfull/checkpoint-1000/adapter_model"

restaurants = "[{\"name\":\"Pizza Hut\",\"stars\":1.5,\"review_count\":45,\"attributes\":{\"Ambience\":\"{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 'casual': False}\",\"Alcohol\":\"u'none'\",\"RestaurantsTakeOut\":\"True\",\"BikeParking\":\"False\",\"RestaurantsReservations\":\"False\",\"NoiseLevel\":\"u'average'\",\"RestaurantsAttire\":\"u'casual'\",\"OutdoorSeating\":\"False\",\"BusinessAcceptsCreditCards\":\"True\",\"RestaurantsGoodForGroups\":\"True\",\"Caters\":\"False\",\"RestaurantsPriceRange2\":\"1\",\"HasTV\":\"True\",\"WiFi\":\"'no'\",\"RestaurantsDelivery\":\"True\",\"GoodForKids\":\"True\",\"BusinessParking\":\"{'garage': None, 'street': False, 'validated': None, 'lot': True, 'valet': False}\",\"GoodForMeal\":\"{'dessert': None, 'latenight': None, 'lunch': True, 'dinner': True, 'brunch': None, 'breakfast': None}\"},\"distance_km\":4.66,\"keywords\":[\"pizza hut\",\"crusts frozen\",\"undercooked downside\",\"eat waiting\",\"scene worse\",\"dropping orders\",\"shout kirkwood\",\"employees sitting\",\"work sad\",\"gave gluten\",\"reviews dodged\",\"door cheaper\",\"silverware ok\",\"just went\",\"location dump\",\"occupied wonder\",\"regarding cleanliness\",\"service slow\",\"jamar really\",\"approach table\"]},{\"name\":\"Southerner's Coffee\",\"stars\":4.5,\"review_count\":13,\"attributes\":{\"WiFi\":\"u'free'\",\"WheelchairAccessible\":\"False\",\"Caters\":\"True\",\"GoodForKids\":\"True\",\"OutdoorSeating\":\"False\",\"DriveThru\":\"True\",\"BusinessParking\":\"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}\",\"RestaurantsDelivery\":\"True\",\"RestaurantsTakeOut\":\"True\",\"BikeParking\":\"True\",\"BusinessAcceptsCreditCards\":\"True\"},\"distance_km\":8.19,\"keywords\":[\"coffee downtown\",\"welcome flavortown\",\"visiting tn\",\"experience franklin\",\"recommend place\",\"tasting quality\",\"southerner glad\",\"iced cubans\",\"amazing service\",\"bean welcomed\",\"awesome 16\",\"dunkin custom\",\"owners open\",\"fieri triple\",\"area menu\",\"far really\",\"window payment\",\"strange covid\",\"definitely try\",\"complaining ordering\"]},{\"name\":\"Local Brewing\",\"stars\":3.0,\"review_count\":254,\"attributes\":{\"RestaurantsAttire\":\"u'casual'\",\"RestaurantsTakeOut\":\"True\",\"GoodForKids\":\"True\",\"BusinessAcceptsCreditCards\":\"True\",\"RestaurantsPriceRange2\":\"2\",\"RestaurantsGoodForGroups\":\"True\",\"Caters\":\"True\",\"Smoking\":\"u'outdoor'\",\"RestaurantsTableService\":\"True\",\"CoatCheck\":\"False\",\"HasTV\":\"True\",\"BusinessParking\":\"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}\",\"Alcohol\":\"'full_bar'\",\"BikeParking\":\"True\",\"Music\":\"{'dj': False, 'background_music': False, 'no_music': False, 'jukebox': False, 'live': True, 'video': False, 'karaoke': False}\",\"GoodForMeal\":\"{'dessert': False, 'latenight': True, 'lunch': True, 'dinner': True, 'brunch': False, 'breakfast': False}\",\"GoodForDancing\":\"False\",\"WiFi\":\"'free'\",\"RestaurantsReservations\":\"True\",\"OutdoorSeating\":\"True\",\"BestNights\":\"{'monday': False, 'tuesday': False, 'friday': True, 'wednesday': False, 'thursday': True, 'sunday': False, 'saturday': False}\",\"NoiseLevel\":\"'average'\",\"HappyHour\":\"True\",\"Ambience\":\"{'touristy': False, 'hipster': False, 'romantic': False, 'divey': False, 'intimate': False, 'trendy': False, 'upscale': False, 'classy': True, 'casual': True}\",\"RestaurantsDelivery\":\"False\"},\"distance_km\":8.15,\"keywords\":[\"beer restaurant\",\"hospitality needless\",\"facebook reviews\",\"server concerned\",\"meal exceeded\",\"disrespect veteran\",\"try sitting\",\"order care\",\"leave doggy\",\"finally gave\",\"experience felicia\",\"dirty menus\",\"scf activities\",\"encapsulate rest\",\"appealing employees\",\"probably better\",\"cauliflower disappointed\",\"tables jukebox\",\"music outside\",\"balsamic reduction\"]},{\"name\":\"Le Cafe Creperie\",\"stars\":4.0,\"review_count\":33,\"attributes\":{\"RestaurantsReservations\":\"False\",\"RestaurantsDelivery\":\"True\",\"RestaurantsGoodForGroups\":\"True\",\"BusinessParking\":\"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}\",\"BikeParking\":\"True\",\"OutdoorSeating\":\"False\",\"WiFi\":\"u'no'\",\"RestaurantsAttire\":\"u'casual'\",\"NoiseLevel\":\"u'quiet'\",\"Ambience\":\"{'romantic': False, 'intimate': False, 'touristy': False, 'hipster': False, 'divey': False, 'classy': False, 'trendy': False, 'upscale': False, 'casual': True}\",\"GoodForKids\":\"True\",\"RestaurantsTakeOut\":\"True\",\"HasTV\":\"False\",\"Alcohol\":\"u'none'\",\"RestaurantsPriceRange2\":\"1\",\"Caters\":\"True\",\"BusinessAcceptsCreditCards\":\"True\"},\"distance_km\":6.6,\"keywords\":[\"crepe desserts\",\"alternative pizzas\",\"try flatbread\",\"great lunch\",\"tried naanza\",\"veggies fresh\",\"basil nanzza\",\"restaurant clean\",\"preferred raspberry\",\"nasty craving\",\"getting orders\",\"negative reviews\",\"mandarin orange\",\"concerned calories\",\"watch make\",\"definitely worth\",\"thicker broth\",\"france cheaper\",\"stopping table\",\"missing definition\"]},{\"name\":\"KFC\",\"stars\":1.5,\"review_count\":28,\"attributes\":{\"RestaurantsAttire\":\"u'casual'\",\"RestaurantsPriceRange2\":\"1\",\"WiFi\":\"'free'\",\"RestaurantsGoodForGroups\":\"True\",\"RestaurantsTakeOut\":\"True\",\"Caters\":\"True\",\"Ambience\":\"None\",\"Alcohol\":\"u'none'\",\"GoodForKids\":\"True\",\"BusinessParking\":\"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}\",\"OutdoorSeating\":\"False\",\"RestaurantsReservations\":\"False\",\"BusinessAcceptsCreditCards\":\"True\",\"RestaurantsDelivery\":\"True\",\"DriveThru\":\"True\",\"HasTV\":\"True\"},\"distance_km\":1.51,\"keywords\":[\"kfc sucks\",\"waiting order\",\"beans stopped\",\"taco management\",\"food aggravated\",\"supervisor frustrated\",\"customers wondering\",\"evening corn\",\"kitchen counting\",\"aren serving\",\"staff april\",\"getting decent\",\"fixings visiting\",\"cleaning stocking\",\"yesterday 5pm\",\"screw failed\",\"register stop\",\"sure checkyor\",\"asymptomatic people\",\"woman dressed\"]},{\"name\":\"Lucky Bamboo\",\"stars\":2.5,\"review_count\":37,\"attributes\":{\"Alcohol\":\"u'full_bar'\",\"RestaurantsDelivery\":\"False\",\"BusinessParking\":\"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}\",\"GoodForKids\":\"True\",\"RestaurantsTakeOut\":\"True\",\"RestaurantsReservations\":\"True\",\"RestaurantsAttire\":\"u'casual'\",\"WiFi\":\"u'no'\",\"OutdoorSeating\":\"False\",\"RestaurantsPriceRange2\":\"2\",\"BusinessAcceptsCreditCards\":\"True\",\"RestaurantsGoodForGroups\":\"True\",\"NoiseLevel\":\"u'average'\",\"Caters\":\"False\",\"Ambience\":\"{'romantic': False, 'intimate': False, 'touristy': False, 'hipster': False, 'divey': False, 'classy': False, 'trendy': False, 'upscale': False, 'casual': False}\",\"HasTV\":\"True\"},\"distance_km\":3.89,\"keywords\":[\"chinese restaurant\",\"provide nashville\",\"great haven\",\"bilingual addititon\",\"sum menu\",\"coast types\",\"finer aspects\",\"option dim\",\"crispy dumplings\",\"experience tables\",\"philippines couple\",\"cost plates\",\"closed renovations\",\"teach teriyaki\",\"aahh mazing\",\"tea tax\",\"pretty fun\",\"heating function\",\"average check\",\"toxins hearty\"]},{\"name\":\"McDonald's\",\"stars\":1.5,\"review_count\":20,\"attributes\":{\"BikeParking\":\"True\",\"NoiseLevel\":\"u'quiet'\",\"RestaurantsAttire\":\"u'casual'\",\"Caters\":\"False\",\"RestaurantsPriceRange2\":\"1\",\"Alcohol\":\"u'none'\",\"OutdoorSeating\":\"False\",\"BusinessAcceptsCreditCards\":\"True\",\"WiFi\":\"u'free'\",\"Ambience\":\"{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 'casual': False}\",\"RestaurantsTakeOut\":\"True\",\"RestaurantsReservations\":\"False\",\"RestaurantsGoodForGroups\":\"True\",\"RestaurantsTableService\":\"False\",\"RestaurantsDelivery\":\"True\",\"GoodForMeal\":\"{'dessert': False, 'latenight': False, 'lunch': False, 'dinner': False, 'brunch': False, 'breakfast': False}\",\"GoodForKids\":\"True\",\"HasTV\":\"True\",\"DriveThru\":\"True\",\"BusinessParking\":\"{u'valet': False, u'garage': False, u'street': None, u'lot': True, u'validated': False}\"},\"distance_km\":7.81,\"keywords\":[\"service mcdonalds\",\"ordering smells\",\"food forgetful\",\"staff inconsistent\",\"location employees\",\"dispenser hadn\",\"kiosk shut\",\"unfriendly\",\"hot clean\",\"probably slowest\",\"read menu\",\"approach doors\",\"area nice\",\"tried cover\",\"leave drive\",\"arrived 00pm\",\"kid decision\",\"reserve reviews\",\"ignored counter\",\"covid risk\"]},{\"name\":\"At The Table BYOB\",\"stars\":4.5,\"review_count\":111,\"attributes\":{\"BusinessParking\":\"{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}\",\"WheelchairAccessible\":\"True\",\"DogsAllowed\":\"False\",\"BusinessAcceptsCreditCards\":\"True\",\"WiFi\":\"u'free'\",\"RestaurantsTableService\":\"True\",\"RestaurantsGoodForGroups\":\"False\",\"BikeParking\":\"True\",\"Ambience\":\"{'touristy': False, 'hipster': False, 'romantic': True, 'divey': False, 'intimate': True, 'trendy': False, 'upscale': False, 'classy': True, 'casual': False}\",\"OutdoorSeating\":\"True\",\"NoiseLevel\":\"u'average'\",\"GoodForKids\":\"False\",\"RestaurantsReservations\":\"True\",\"Alcohol\":\"u'none'\",\"BYOB\":\"True\",\"RestaurantsAttire\":\"u'casual'\",\"RestaurantsPriceRange2\":\"3\",\"GoodForMeal\":\"{u'breakfast': False, u'brunch': False, u'lunch': False, u'dinner': True, u'latenight': False, u'dessert': None}\",\"RestaurantsTakeOut\":\"True\",\"RestaurantsDelivery\":\"True\",\"Caters\":\"True\",\"HasTV\":\"False\"},\"distance_km\":3.93,\"keywords\":[\"dining delight\",\"toast tomato\",\"salmon succulently\",\"memorable experience\",\"advice reservation\",\"sooooo good\",\"offered glass\",\"dishwasher station\",\"rushed garden\",\"courses ordered\",\"accommodating butternut\",\"seek traveled\",\"deserve pairings\",\"truffled polenta\",\"1000 calories\",\"baril stopped\",\"opentable fall\",\"contrasting textures\",\"apparently did\",\"cost savings\"]}]\n" 
# "output": "{\"output\":[{\"name\":\"KFC\",\"reason\":\"KFC is the only spot within your 2 km limit (distance_km: 1.51), so it’s the closest option to check for restroom cleanliness. It does have low overall ratings (stars: 1.5, review_count: 28) and mixed reviewer notes, though some keywords reference \\\"cleaning\\\" and \\\"stocking,\\\" so I’d recommend calling ahead to confirm their current hygiene/ restroom status before you go.\"},{\"name\":\"At The Table BYOB\",\"reason\":\"At The Table BYOB is highly rated (stars: 4.5, review_count: 111) and reviewers mention a \\\"dishwasher station\\\" in the keywords, which is a positive sign for good cleaning practices and overall hygiene. It’s a bit farther than your 2 km request (distance_km: 3.93), but its strong ratings and many reviews make it a good candidate if you’re willing to travel slightly farther for spotless restrooms.\"},{\"name\":\"Le Cafe Creperie\",\"reason\":\"Le Cafe Creperie has solid ratings (stars: 4.0, review_count: 33) and explicitly includes \\\"restaurant clean\\\" in the keywords, suggesting reviewers noticed cleanliness. It’s farther out (distance_km: 6.6) and more of a quiet, casual spot (NoiseLevel: \\\"quiet\\\", RestaurantsAttire: \\\"casual\\\"), so if strict restroom hygiene is a priority and you can travel beyond 2 km, this is a dependable choice.\"}]}"}

user_question = "query\":\"Find nearby restaurants with spotless restrooms and high hygiene ratings within 2 km.\""
prompt = "### System:\nYou are a restaurant recommender.\nRules:\n1) You MUST recommend 3 restaurants, chosen ONLY from the provided restaurants list.\n2) NEVER invent a restaurant or modify restaurant names.\n3) Output MUST be valid JSON only (no markdown, no extra text).\n4) JSON schema:\n{\n  \"output\": [\n    {\"name\": \"<exact restaurant name from restaurants>\", \"reason\": \"<explanation>\"},\n    {\"name\": \"...\", \"reason\": \"...\"},\n    {\"name\": \"...\", \"reason\": \"...\"}\n  ]\n}\n5) Reasons should primarily reflect the user query, and reference fields like distance_km, stars, attributes, review_count, and keywords when available.\n6) Write each reason as if a kind and real person is speaking naturally in conversation.\n\n### User:\n '{user_question}',\"restaurants\":'{restaurants}', \"output\":"
# prompt = (
#     "A chat between a curious human and an artificial intelligence assistant. "
#     "The assistant gives helpful, detailed, and polite answers to the user's questions. "
#     "### Human: {user_question}"
#     "### Assistant: "
# )


import json

# user_question 은 '문장'만 넣어야 함 (json 조각 X)
user_question = "Find nearby restaurants with spotless restrooms and high hygiene ratings within 2 km."

# restaurants 는 지금처럼 JSON 문자열이어도 됨. (단, 끝에 \n 있으면 제거 권장)
restaurants = restaurants.rstrip()  # 혹시 마지막에 \n 있으면 제거

prompt = (
    "### System:\n"
    "You are a restaurant recommender.\n"
    "Rules:\n"
    "1) You MUST recommend 3 restaurants, chosen ONLY from the provided restaurants list.\n"
    "2) NEVER invent a restaurant or modify restaurant names.\n"
    "3) Output MUST be valid JSON only (no markdown, no extra text).\n"
    "4) JSON schema:\n"
    "{\n"
    "  \"output\": [\n"
    "    {\"name\": \"<exact restaurant name from restaurants>\", \"reason\": \"<explanation>\"},\n"
    "    {\"name\": \"...\", \"reason\": \"...\"},\n"
    "    {\"name\": \"...\", \"reason\": \"...\"}\n"
    "  ]\n"
    "}\n"
    "5) Reasons should primarily reflect the user query, and reference fields like distance_km, stars, attributes, review_count, and keywords when available.\n"
    "6) Write each reason as if a kind and real person is speaking naturally in conversation.\n"
    "\n"
    "### User:\n"
    + json.dumps({"query": user_question, "restaurants": json.loads(restaurants)}, ensure_ascii=False)
    + "\n"
)

def load_model():
    # 학습 때와 동일하게 4bit 로드 (nf4/double_quant 등도 가능하면 맞추는 게 좋음)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=True,
        padding_side="left",  # 생성은 left padding이 보통 안전
        trust_remote_code=False,
    )

    # pad_token이 없으면 eos로 대체 (LLM 생성에서 흔히 사용)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )

    # LoRA adapter 로드 (학습된 폴더)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)
    model.eval()

    return model, tokenizer
def load_model_fp16():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=True,
        padding_side="left",
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ 4-bit 설정 제거, fp16으로 로드
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,   # 또는 torch.bfloat16 (bf16 지원 GPU면)
        low_cpu_mem_usage=True,
    )

    # LoRA adapter 로드 (그대로 OK)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)
    model.eval()

    return model, tokenizer

@torch.inference_mode()
def generate_answer(model, tokenizer, user_question: str,
                    max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True):
    # prompt = prompt_tmpl.format(user_question=user_question)
    prompt = user_question

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=6144,  # 필요시 조절
    )

    # device_map="auto"일 때는 아래처럼 to(model.device) 대신,
    # input tensor를 첫 번째 디바이스로만 옮기면 보통 잘 동작합니다.
    # (환경에 따라 그냥 .to(model.device)도 되긴 합니다.)
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # import pdb;pdb.set_trace()
    # 프롬프트를 포함해 디코딩되므로, 보통 프롬프트 이후만 잘라서 반환
    # if "output" in text:
    #     text = text.split("output", 1)[-1].strip()
    if '"output"' in text:
        try:
            data = json.loads(text.split("\n")[-1])  # 마지막 JSON만 사용
            text = data["output"]
        except Exception as e:
            print("JSON parsing failed:", e)
    return text

if __name__ == "__main__":
    model, tokenizer = load_model() # 4-bit inference
    # model, tokenizer = load_model_fp16()

    # q = "부산에서 데이트하기 좋은 이탈리안 레스토랑 추천해줘. 분위기 좋은 곳 위주로!"
    ans = generate_answer(model, tokenizer, prompt)


    ############################### Facebook/ nllb 모델 이용해서 한국어로 번역
    ## Eng -> Kor
    trans_model_name = "facebook/nllb-200-distilled-600M"

    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)

    def translate_en_to_ko(text):
        inputs = trans_tokenizer(text, return_tensors="pt")
        translated_tokens = trans_model.generate(
            **inputs,
            forced_bos_token_id=trans_tokenizer.lang_code_to_id["kor_Hang"],
            max_length=512
        )
        return trans_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    for item in ans:
        item["reason"] = translate_en_to_ko(item["reason"])

    ############################################################
    # FOR COMPARISON against API

    # from openai import OpenAI
    # client = OpenAI()

    # def translate_reason(reason_en: str) -> str:
    #     response = client.responses.create(
    #         model="gpt-5-mini",
    #         input=[
    #             {
    #                 "role": "developer",
    #                 "content": (
    #                     "You are a professional English-to-Korean translator.\n"
    #                     "Your top priority is translation accuracy.\n"
    #                     "- Preserve all numbers, ratings, distances, keywords, and proper nouns exactly.\n"
    #                     "- Do NOT add new information.\n"
    #                     "- Do NOT exaggerate or reinterpret facts.\n"
    #                     "- Maintain original meaning faithfully.\n"
    #                     "- Tone may be slightly friendly and warm.\n"
    #                     "- You may use 0-2 emojis maximum.\n"
    #                     "- Keep it natural and fluent Korean."
    #                 ),
    #             },
    #             {
    #                 "role": "user",
    #                 "content": (
    #                     "다음 영어 문장을 정확하게 한국어로 번역해주세요.\n"
    #                     "톤은 친근하게, 하지만 정보 왜곡 없이 정확하게 번역해주세요.\n\n"
    #                     f"{reason_en}\n\n"
    #                     "출력은 번역문만 주세요."
    #                 ),
    #             },
    #         ],
    #     )

    #     return response.output_text
    
    # for item in ans:
    #     item["reason"] = translate_reason(item["reason"])

    ############################################################
    

    print(ans)


