你是角色设定卡抽取器。从下面的外貌描述中抽取结构化字段，供跨镜头一致性校验使用。

外貌描述：{appearance}

只输出 JSON：
{"face": str, "hair": str, "eyes": str, "build": str, "outfit": str, "headwear": str, "palette": [str], "distinctive": [str]}

要求：
1. 每个字段只写描述中**明确出现**的信息，不得脑补；缺失写空字符串或空数组
2. hair / outfit / headwear 必须保留原文的颜色词（"黄色头盔"不得写成"头盔"）
3. palette 为服装与配件的主色词列表（如 ["蓝色", "黄色"]）
4. distinctive 为最具辨识度的 1-3 个特征短语（如 ["脸颊冻得发红"]）
