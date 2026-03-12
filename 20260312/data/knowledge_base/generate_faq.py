import json
import random

def generate_mock_faq():
    categories = ["退换货政策", "物流配送", "会员权益", "产品参数", "售后维修"]
    products = ["手机X1", "平板P2", "耳机E3", "手表W4", "笔记本N5"]
    
    faqs = []
    # 1. 核心通用规则（手工编写，高质量）
    faqs.extend([
        {"question": "你们的退货政策是什么？", "answer": "购买后7天内可无理由退换货，商品需保持原包装未拆封，且不影响二次销售。退回运费由买家承担。"},
        {"question": "消费多少可以升级会员？", "answer": "累计消费满1000元可升级为白银会员（享98折），满5000元升级为黄金会员（享95折），满10000元升级为钻石会员（享9折及专属客服）。"},
        {"question": "默认发什么快递？", "answer": "我们默认使用顺丰速运或京东物流，确保商品安全快速送达。偏远地区可能转寄邮政EMS。"}
    ])

    # 2. 批量生成长尾问题（模拟 500 条数据）
    for i in range(1, 498):
        cat = random.choice(categories)
        prod = random.choice(products)
        
        if cat == "产品参数":
            faqs.append({
                "question": f"{prod}的电池容量是多少？",
                "answer": f"{prod}配备了{random.randint(3000, 6000)}mAh的大容量电池，支持{random.choice([30, 65, 120])}W快充，大约{random.randint(30, 60)}分钟即可充满。"
            })
        elif cat == "售后维修":
            faqs.append({
                "question": f"我的{prod}屏幕碎了怎么保修？",
                "answer": f"人为损坏不在{prod}的免费保修范围内。您可以联系官方售后进行付费维修，屏幕更换参考价为{random.randint(300, 1500)}元。如果您购买了碎屏险，请提供保险单号。"
            })
        else:
            faqs.append({
                "question": f"关于{prod}的{cat}规定编号#{i}",
                "answer": f"根据公司第{i}条规定，关于{prod}的{cat}，我们会提供为期{random.randint(1, 3)}年的支持服务，详情请查阅随附说明书。"
            })

    # 保存为 JSON 文件
    with open("data/knowledge_base/faq.json", "w", encoding="utf-8") as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 成功生成 {len(faqs)} 条知识库数据，已保存至 faq.json")

if __name__ == "__main__":
    generate_mock_faq()