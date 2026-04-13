import os
os.environ["OPENAI_API_KEY"] = NULL # Your API Key



import openai

client = openai.OpenAI()


import torch
total_questions = 10000 # num of total questions
current_questions = 0
responses = []
while True:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that generates user prompts that real users would likely ask when searching for good nearby restaurants."},
            {"role": "user", "content": "Please generate a wide variety of user prompts that real users might ask, using very diverse combinations such as hygiene, restroom cleanliness, atmosphere, distance (within a few kilometers), purpose of visit, Wi-Fi availability, parking, quietness, value for money, menu preferences, and many other factors"}
        ]
    )
    kk = response.choices[0].message.content
    lines = kk.split('\n')
    num_of_question = len(lines)
    current_questions += num_of_question
    cleaned = [a[2:] for a in lines]
    responses.extend(cleaned)
    print(current_questions)
    if current_questions > total_questions:
        break
import pdb;pdb.set_trace()
torch.save(responses, "user_prompt_10k.pth")
