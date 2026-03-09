from openai import OpenAI

client=OpenAI(
    apikey="",
    base_url="http://192.168.2.8:8000/v1"
)


def get_ai_response(prompt):
    try:
        completion=client.chat.completions.create(
            model="Qwen2.5-14B-Instruct-AWQ",
            messages=[
                {"role":"system","content":"你是一个回答用户问题的智能助手帮助用户解决问题"},
                {"role":"user","content":prompt}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return e