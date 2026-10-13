"""台词情绪的**归一化**：把 LLM 写的中文自由文本收敛成供应商能消费的档位。

为什么需要这一层（2026-10-05 实测的"人物音频很奇怪"根因之一）：
`edge_tts` 的韵律表键是英文（`angry`/`happy`…），而剧本 `emotion` 字段实际写的是
「震惊转羞涩」「焦急」「戏谑」这类中文自由文本 → 查表 100% 落空 → rate/pitch 恒
`+0%`，全片每句一个语调。cosyvoice 侧同理（SDK 只认数值 `speech_rate`/`pitch_rate`）。
所以情绪必须在**进供应商之前**归一，而不是指望供应商懂中文。

一个字段可能含多种情绪（"震惊转羞涩"）：取**第一个命中**的档位，因为台词开头的
情绪决定起调，后半段的变化交给音色本身的表现力。
"""

from __future__ import annotations

# 归一后的情绪档 → (语速乘子, 音调乘子)。**数值以 1.0 为中心**，由各家 provider
# 自己换算成它要的形态（edge-tts 要 "+12%/+3Hz"，cosyvoice SDK 要 speech_rate=1.12）。
# 幅度刻意保守：情绪靠音色本身就有，韵律只加"倾向"，过头就变机器人/卡通腔。
EMOTION_PROSODY: dict[str, tuple[float, float]] = {
    "angry": (1.12, 1.06),
    "excited": (1.12, 1.08),
    "happy": (1.06, 1.04),
    "playful": (1.08, 1.06),
    "surprised": (1.08, 1.12),
    "nervous": (1.14, 1.04),
    "sad": (0.90, 0.96),
    "tired": (0.92, 0.96),
    "serious": (0.94, 0.98),
    "tender": (0.94, 1.02),
    "shy": (0.98, 1.04),
    "whisper": (0.86, 0.98),
    "neutral": (1.0, 1.0),
}

# 关键词 → 档位。顺序敏感：靠前的先命中（"震惊"要排在"惊"这类单字之前才有意义）。
_KEYWORDS: list[tuple[str, str]] = [
    ("愤怒", "angry"), ("恼火", "angry"), ("暴怒", "angry"), ("生气", "angry"),
    ("气愤", "angry"), ("怒", "angry"),
    ("激动", "excited"), ("兴奋", "excited"), ("狂喜", "excited"), ("亢奋", "excited"),
    ("戏谑", "playful"), ("调侃", "playful"), ("打趣", "playful"), ("顽皮", "playful"),
    ("贱兮兮", "playful"), ("得意", "playful"),
    ("开心", "happy"), ("高兴", "happy"), ("欣喜", "happy"), ("笑", "happy"),
    ("喜", "happy"), ("愉悦", "happy"),
    ("震惊", "surprised"), ("惊讶", "surprised"), ("吃惊", "surprised"),
    ("诧异", "surprised"), ("惊愕", "surprised"), ("愣", "surprised"), ("惊", "surprised"),
    ("焦急", "nervous"), ("着急", "nervous"), ("紧张", "nervous"), ("慌张", "nervous"),
    ("惶恐", "nervous"), ("害怕", "nervous"), ("恐惧", "nervous"), ("心虚", "nervous"),
    ("急", "nervous"),
    ("悲伤", "sad"), ("难过", "sad"), ("哀", "sad"), ("哭", "sad"), ("哽咽", "sad"),
    ("失落", "sad"), ("沮丧", "sad"), ("悲", "sad"),
    ("疲惫", "tired"), ("慵懒", "tired"), ("累", "tired"), ("虚弱", "tired"),
    ("有气无力", "tired"),
    ("严肃", "serious"), ("凝重", "serious"), ("郑重", "serious"), ("认真", "serious"),
    ("沉痛", "serious"), ("冷", "serious"), ("厉", "serious"),
    ("温柔", "tender"), ("深情", "tender"), ("宠溺", "tender"), ("关切", "tender"),
    ("撒娇", "tender"), ("柔", "tender"),
    ("羞涩", "shy"), ("害羞", "shy"), ("娇羞", "shy"), ("扭捏", "shy"), ("赧", "shy"),
    ("小声", "whisper"), ("低声", "whisper"), ("低语", "whisper"), ("耳语", "whisper"),
    ("悄悄", "whisper"), ("呢喃", "whisper"), ("喃喃", "whisper"),
    ("平静", "neutral"), ("冷淡", "neutral"), ("淡淡", "neutral"), ("默认", "neutral"),
]

# 供应商可能直接给英文档（fake 供应商与旧台账就给 "happy"/"doubt"）
_ALIASES = {
    "angry": "angry", "mad": "angry", "excited": "excited", "happy": "happy",
    "laugh": "happy", "laughing": "happy", "playful": "playful", "tease": "playful",
    "surprised": "surprised", "shock": "surprised", "shocked": "surprised",
    "nervous": "nervous", "anxious": "nervous", "panic": "nervous", "sad": "sad",
    "cry": "sad", "tired": "tired", "serious": "serious", "calm": "neutral",
    "neutral": "neutral", "doubt": "serious", "whisper": "whisper", "tender": "tender",
    "shy": "shy",
}


def normalize_emotion(raw: str | None) -> str:
    """任意情绪描述 → `EMOTION_PROSODY` 的档位名；认不出返回 `neutral`。

    返回的永远是表里存在的键，所以调用方可以放心 `.get(..., (1.0, 1.0))` 之外
    再用 `EMOTION_PROSODY[bucket]` 直接取值 —— 不会因供应商侧改表而 KeyError。
    """
    text = (raw or "").strip().lower()
    if not text:
        return "neutral"
    if text in _ALIASES:
        return _ALIASES[text]
    for zh, bucket in _KEYWORDS:
        if zh in text:
            return bucket
    return "neutral"


def prosody(raw: str | None) -> tuple[float, float]:
    """情绪 → (语速乘子, 音调乘子)，中心 1.0。"""
    return EMOTION_PROSODY.get(normalize_emotion(raw), (1.0, 1.0))
