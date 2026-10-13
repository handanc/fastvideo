"""身份一致性的**客观层**（Tier C）：人脸 embedding 余弦相似度，离线、零边际成本、可复现。

为什么需要它：现有两层都不够 ——
- Tier A（VLM 属性位）能给出"头盔颜色错了"这种可行动原因，但它慢、要钱，而且**已经多次
  解析失败**（`final_review` 落 0.6 兜底分、首帧抽查 `consistency_check_unavailable`），
  把主观模型当闸门就会把异常写成"通过"；
- Tier B（色相/直方图，`utils/quality.py`）便宜可靠，但它量的是**服装主色**，只能证伪
  "衣服换了"，回答不了"这还是不是同一张脸"。

选 SFace 而不是 insightface/ArcFace 包：本机 `cv2 5.0.0` 自带 `FaceDetectorYN`（YuNet 检测）
与 `FaceRecognizerSF`（SFace 识别，ArcFace 家族、余弦相似度），**由 cv2 的 DNN 后端直接读
ONNX** —— 不需要 onnxruntime，也不需要装 insightface（这两个包本机都没有）。权重两个文件
共约 37MB，一次下载落进 `models/`，之后完全离线。

三条硬约束（都是这个项目付过学费的）：
1. **权重缺失/检不到脸/任何异常 → 返回 `available=False` 或 `verdict="inconclusive"`，
   绝不抛错**。度量不能拦住交付；宁可明说"没量到"，也不要给一个像结论的假数字
   （`utils/lipsync.py` 的假阴性教训：那把尺子曾让人眼确认同步的片段被判不同步）。
2. **不与 Tier A/B 抢话语权**：三层各记各的数，冲突时记 `metric_conflict` 交人看，
   而不是让某一层直接改写另一层的结论。
3. 阈值写死成常量并注明出处（OpenCV 官方对 SFace 的推荐余弦阈值 0.363），
   不做"看起来更严"的私调。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fastvideo3.core.config import MODELS_DIR

#: YuNet 人脸检测 + SFace 人脸识别（OpenCV Zoo 官方权重名，与 cv2 API 配套）
DETECTOR_NAME = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_NAME = "face_recognition_sface_2021dec.onnx"

#: OpenCV 文档给 SFace 的推荐余弦阈值：≥0.363 判同一人。不私调 —— 调阈值等于自造标准，
#: 而这一层的价值恰恰在"客观、可复现、别人能拿同样的数复算"。
COSINE_THRESHOLD = 0.363

#: `FaceRecognizerSF.match` 的距离类型。cv2 5.0 里 `FR_COSINE`/`FR_NORM_L2` 这两个类属性
#: 已经不存在（实测 AttributeError），只能用字面量：0=L2、1=余弦。
_MATCH_COSINE = 1

#: 下载镜像（按顺序试）。github 在国内常被重置，所以多备几个；全部失败时把"手动放置"
#: 的路径说清楚，而不是让功能静默不可用。
_MIRRORS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/{sub}/{name}",
    "https://gitee.com/opencv/opencv_zoo/raw/main/models/{sub}/{name}",
    "https://raw.gitcode.com/gh_mirrors/op/opencv_zoo/raw/main/models/{sub}/{name}",
    "https://hf-mirror.com/opencv/opencv_zoo/resolve/main/models/{sub}/{name}",
)
_SUBDIR = {
    DETECTOR_NAME: "face_detection_yunet",
    RECOGNIZER_NAME: "face_recognition_sface",
}

_detector: Any = None
_recognizer: Any = None


def weight_paths() -> dict[str, Path]:
    return {
        DETECTOR_NAME: MODELS_DIR / DETECTOR_NAME,
        RECOGNIZER_NAME: MODELS_DIR / RECOGNIZER_NAME,
    }


def missing_weights() -> list[str]:
    """缺哪些权重文件（空列表＝齐了）。"""
    return [n for n, p in weight_paths().items() if not p.exists() or p.stat().st_size < 1024]


def available() -> bool:
    """客观身份层是否可用：权重齐 + cv2 带这两个类。"""
    if missing_weights():
        return False
    try:
        import cv2

        return hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
    except Exception:  # noqa: BLE001  # 没装 cv2 就是不可用，不是错误
        return False


def fetch_weights(force: bool = False, timeout: float = 60.0) -> tuple[list[str], str | None]:
    """下载缺失的权重到 `models/`。返回 `(已就位文件名, 错误说明或 None)`。

    逐个镜像试；任何一个成功就落盘（先写 `.part` 再改名，避免半个文件被当成权重）。
    """
    import httpx

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    last_err = ""
    for name, path in weight_paths().items():
        if path.exists() and path.stat().st_size > 1024 and not force:
            done.append(name)
            continue
        ok = False
        for tpl in _MIRRORS:
            url = tpl.format(sub=_SUBDIR[name], name=name)
            try:
                tmp = path.with_suffix(path.suffix + ".part")
                with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
                    if r.status_code != 200:
                        last_err = f"{url} → HTTP {r.status_code}"
                        continue
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                # ONNX 文件不可能只有几百字节；小于 64KB 说明拿到的是错误页/重定向 HTML
                if tmp.stat().st_size < 64 * 1024:
                    last_err = f"{url} → 只有 {tmp.stat().st_size} 字节（像是错误页而非权重）"
                    tmp.unlink(missing_ok=True)
                    continue
                tmp.replace(path)
                done.append(name)
                ok = True
                break
            except Exception as e:  # noqa: BLE001  # 换下一个镜像
                last_err = f"{url} → {type(e).__name__}: {e}"
        if not ok:
            return done, (
                f"{name} 下载失败（最后错误：{last_err}）。可手动放置到 {path} —— "
                "来源 OpenCV Zoo（models/face_detection_yunet 与 models/face_recognition_sface）。"
                "缺权重时身份一致性退回 Tier A/B（VLM 属性位 + 色相证伪），不阻断交付。"
            )
    return done, None


def _det(cv2: Any):
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            str(weight_paths()[DETECTOR_NAME]), "", (320, 320), 0.7, 0.3, 5000
        )
    return _detector


def _rec(cv2: Any):
    global _recognizer
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(
            str(weight_paths()[RECOGNIZER_NAME]), ""
        )
    return _recognizer


def _read_bgr(path: Path) -> Any:
    import cv2

    # 中文路径在 Windows 上 `cv2.imread` 会失败 → 走 fromfile + imdecode
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _faces(img: Any) -> list[Any]:
    import cv2

    det = _det(cv2)
    det.setInputSize((img.shape[1], img.shape[0]))
    out = det.detect(img)
    # cv2 4.x 返回 (retval, faces)，5.x 可能直接返回 faces；两种都接住
    faces = out[1] if isinstance(out, tuple) else out
    if faces is None or len(faces) == 0:
        return []
    return [faces[i] for i in range(faces.shape[0])]


def _feature(img: Any, face: Any) -> Any:
    import cv2

    rec = _rec(cv2)
    aligned = rec.alignCrop(img, face)
    feat = rec.feature(aligned)
    return np.ndarray.flatten(np.asarray(feat))


def _cosine(a: Any, b: Any) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def identity_similarity(ref: Path, frame: Path, *, pick: str = "best") -> dict:
    """比较两张图里的人脸是否同一人。返回诊断 dict（**永不抛错**）。

    键：`available`（客观层是否可用）、`cosine`（相似度）、`verdict`
    （same/different/inconclusive）、`reason`、`ref_faces`/`frame_faces`（各检到几张脸）。

    `pick="best"`：取**最像**的那对（多人同框时"这个人还在不在"是我们要的问题）；
    `pick="first"` 取各自第一张脸（单人镜头更快，也避免"拿 A 的脸比 B 的脸"）。
    检不到脸 → `inconclusive`：那说明这一帧没有正脸（背影/远景/空镜），
    **不是**"身份漂了"，两者必须分开说，否则会误伤空镜。
    """
    result: dict[str, Any] = {
        "available": False,
        "cosine": None,
        "verdict": "inconclusive",
        "reason": "",
        "ref_faces": 0,
        "frame_faces": 0,
    }
    if not available():
        result["reason"] = "缺 SFace/YuNet 权重（fetch_weights 或手动放入 models/）→ 客观身份层未启用"
        return result
    try:
        import cv2  # noqa: F401  # 只为确认 cv2 可用；具体调用在 _faces/_feature 内

        a_img = _read_bgr(Path(ref))
        b_img = _read_bgr(Path(frame))
        if a_img is None or b_img is None:
            result["reason"] = "图片读不出（损坏或格式不支持）"
            return result
        a_faces, b_faces = _faces(a_img), _faces(b_img)
        result["available"] = True
        result["ref_faces"] = len(a_faces)
        result["frame_faces"] = len(b_faces)
        if not a_faces or not b_faces:
            result["reason"] = (
                f"检不到人脸（参考图 {len(a_faces)} 张 / 本帧 {len(b_faces)} 张）："
                "背影、远景或空镜属正常，不判身份漂移"
            )
            return result
        pairs = (
            [(a_faces[0], b_faces[0])]
            if pick == "first"
            else [(x, y) for x in a_faces for y in b_faces]
        )
        best = max(
            (_cosine(_feature(a_img, x), _feature(b_img, y)) for x, y in pairs),
            default=0.0,
        )
        result["cosine"] = round(best, 4)
        result["verdict"] = "same" if best >= COSINE_THRESHOLD else "different"
        result["reason"] = (
            f"余弦 {best:.3f} {'≥' if best >= COSINE_THRESHOLD else '<'} 阈值 {COSINE_THRESHOLD}"
        )
        return result
    except Exception as e:  # noqa: BLE001  # 度量自身异常绝不影响交付
        result["reason"] = f"{type(e).__name__}: {e}"
        result["verdict"] = "inconclusive"
        return result


def identity_similarity_multi(refs: dict[str, Path], frame: Path) -> dict:
    """一帧对**多个候选角色**的身份判定（多人同框的镜头必须这样量）。

    为什么不能只比 `characters[0]`：本镜声明了两个角色时，采样那一帧可能正好把第一个
    人拍成侧脸/背光，余弦掉到阈值下 —— 于是"另一个角色明明还在画面上、脸也对得上"的
    合格镜头被判成身份漂移，**直接触发一次付费重拍**。2026-10-12 权重就位后第一次真跑
    就撞上了：某镜唐僧 0.306 判 different，而同帧国王 0.4286 判 same。

    判定口径（只在"该拦的时候"拦）：

    - 画面里**至少一个声明角色**对得上（余弦 ≥ 阈值）→ `same`：这一帧的身份还在；
    - 画面里有脸，但**没有一个**声明角色对得上 → `different`：整批脸都换了 / 换成别人了；
    - 画面里一张脸都检不到 → `inconclusive`：背影、远景、空镜不算身份漂移。

    返回 `{available, verdict, cosine, best_char, per_char, frame_faces, reason}`，
    `per_char[name] = {cosine, verdict, ref_faces}` 留给报告做解释。永不抛错。
    """
    result: dict[str, Any] = {
        "available": False,
        "verdict": "inconclusive",
        "cosine": None,
        "best_char": "",
        "per_char": {},
        "frame_faces": 0,
        "reason": "",
    }
    named = {str(n): Path(p) for n, p in refs.items() if p and Path(p).exists()}
    if not named:
        result["reason"] = "没有任何角色参考图 → 未度量"
        return result
    # 先借单参考图那条路确认"这层可用"（权重在不在、cv2 支不支持），失败原因照原样回
    gate = identity_similarity(next(iter(named.values())), Path(frame))
    if not gate["available"]:
        result["reason"] = gate["reason"]
        return result
    try:
        import cv2  # noqa: F401  # 复用同一套私有原语，不写第二份检测/对齐逻辑

        b_img = _read_bgr(Path(frame))
        if b_img is None:
            result["reason"] = "帧读不出（损坏或格式不支持）"
            return result
        b_faces = _faces(b_img)
        result["available"] = True
        result["frame_faces"] = len(b_faces)
        if not b_faces:
            result["reason"] = (
                f"检不到人脸（本帧 0 张 / 候选 {len(named)} 个角色）："
                "背影、远景或空镜属正常，不判身份漂移"
            )
            return result
        b_feats = [_feature(b_img, f) for f in b_faces]  # 帧上的脸只嵌一次，逐角色复用
        best_overall, best_char = -1.0, ""
        for name, ref in named.items():
            a_img = _read_bgr(ref)
            a_faces = _faces(a_img) if a_img is not None else []
            if not a_faces:
                result["per_char"][name] = {"cosine": None, "verdict": "inconclusive", "ref_faces": 0}
                continue
            cos = max(_cosine(_feature(a_img, fa), fb) for fa in a_faces for fb in b_feats)
            result["per_char"][name] = {
                "cosine": round(cos, 4),
                "verdict": "same" if cos >= COSINE_THRESHOLD else "different",
                "ref_faces": len(a_faces),
            }
            if cos > best_overall:
                best_overall, best_char = cos, name
        measured = [v for v in result["per_char"].values() if v["cosine"] is not None]
        if not measured:
            result["reason"] = "所有候选角色图都检不到脸 → 未度量"
            return result
        result["cosine"] = round(best_overall, 4)
        result["best_char"] = best_char
        if best_overall >= COSINE_THRESHOLD:
            result["verdict"] = "same"
            result["reason"] = (
                f"{best_char} 余弦 {best_overall:.3f} ≥ 阈值 {COSINE_THRESHOLD}（本帧 {len(b_faces)} 张脸）"
            )
        else:
            result["verdict"] = "different"
            result["reason"] = (
                f"本帧 {len(b_faces)} 张脸，但没有一张对得上任何声明角色"
                f"（最高 {best_overall:.3f} < 阈值 {COSINE_THRESHOLD}）"
            )
        return result
    except Exception as e:  # noqa: BLE001  # 度量自身异常绝不影响交付
        result["verdict"] = "inconclusive"
        result["reason"] = f"{type(e).__name__}: {e}"
        return result


def summarize(rows: list[dict]) -> dict:
    """把逐镜的 identity_similarity 结果汇成一段可进 consistency_report 的数字。

    只统计**量到了**的镜头：`inconclusive` 不计入达标率（把"没量到"算成不合格，
    就会像口型度量那样制造假阴性 → 触发付费重拍）。
    """
    measured = [r for r in rows if r.get("cosine") is not None]
    if not measured:
        return {
            "faceid_available": any(r.get("available") for r in rows),
            "faceid_measured": 0,
            "faceid_same_rate": None,
            "faceid_min_cosine": None,
            "faceid_note": "客观身份层没有量到任何镜头（缺权重或都检不到正脸）",
        }
    same = [r for r in measured if r["verdict"] == "same"]
    return {
        "faceid_available": True,
        "faceid_measured": len(measured),
        "faceid_same_rate": round(len(same) / len(measured), 3),
        "faceid_min_cosine": min(float(r["cosine"]) for r in measured),
        "faceid_note": f"{len(same)}/{len(measured)} 个镜头余弦 ≥ {COSINE_THRESHOLD}",
    }
