"""
two_axis_classify.py — 두 축으로 오류 '이유'까지 3분류.

오늘의 발견:
  유사도만으로는 match는 갈리지만 mismatch/hallucination이 겹친다(데이터 확인).
  → 근거 사슬 축을 더한다.

두 축:
  가로 = 기준(자료)과의 의미 유사도 (임베딩 코사인)
  세로 = 답변의 항목/근거가 자료에 실재하는가 (근거 사슬)

3분류 = 오류 이유:
  유사도 높음                     → MATCH        "자료와 일치"
  유사도 낮음 + 항목은 자료에 있음  → VALUE_ERROR  "항목은 맞으나 값이 틀림"
  유사도 낮음 + 근거 자체가 없음    → HALLUCINATION "자료에 근거 없음(지어냄)"

이유를 설명할 수 있으므로, 사용자가 취할 행동이 달라진다:
  VALUE_ERROR → "값을 확인하세요"
  HALLUCINATION → "이 내용은 자료에 없습니다"
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import re
import numpy as np


@dataclass
class Judgment:
    label: str          # MATCH | VALUE_ERROR | HALLUCINATION
    reason: str         # 사람이 읽는 오류 이유
    similarity: float   # 가로축 값
    item_grounded: bool # 세로축: 항목이 자료에 있나
    color: str          # green / orange / red


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t)


def extract_item(sentence: str) -> str:
    """문장에서 주어(항목) 대략 추출. '~은/는/이/가' 앞부분."""
    m = re.match(r"\s*([^,]+?)(은|는|이|가|의|에서|에는)\s", sentence)
    if m:
        return m.group(1).strip()
    # 폴백: 첫 명사구 대략
    return sentence[:8]


class TwoAxisClassifier:
    def __init__(self, corpus: str, embed_fn: Optional[Callable] = None,
                 sim_threshold: float = 0.6):
        """
        corpus: 자료 원문 (세로축 판정 기준)
        embed_fn: text -> vector (가로축). 없으면 유사도 축 비활성.
        sim_threshold: 이 이상이면 MATCH 후보.
        """
        self.corpus = corpus
        self.corpus_compact = _norm(corpus)
        self.embed = embed_fn
        self.sim_th = sim_threshold

    def _item_in_corpus(self, item: str) -> bool:
        """항목(주어)이 자료에 실재하나 = 세로축."""
        key = _norm(item)
        key = re.sub(r"(은|는|이|가|을|를|의|에|에서|에는)$", "", key)
        if len(key) < 2:
            return False
        # 핵심 어절이 자료에 있나 (앞 6자)
        return key[:6] in self.corpus_compact

    def classify(self, sentence: str, ref_text: str,
                 ref_vec=None, sent_vec=None) -> Judgment:
        # ── 가로축: 유사도 ──
        sim = None
        if self.embed is not None:
            rv = ref_vec if ref_vec is not None else self.embed(ref_text)
            sv = sent_vec if sent_vec is not None else self.embed(sentence)
            sim = float(np.dot(rv, sv) / ((np.linalg.norm(rv) * np.linalg.norm(sv)) + 1e-12))

        # ── 세로축: 항목이 자료에 있나 ──
        item = extract_item(sentence)
        grounded = self._item_in_corpus(item)

        # ── 결합 판정 ──
        if sim is not None and sim >= self.sim_th:
            return Judgment("MATCH", "자료와 일치", sim, grounded, "green")

        # 유사도 낮음(또는 미측정) → 세로축으로 이유 구분
        if grounded:
            return Judgment(
                "VALUE_ERROR",
                f"항목 '{item}'은 자료에 있으나 내용이 자료와 다름 — 값 오류 가능(확인 필요)",
                sim if sim is not None else -1, grounded, "orange")
        else:
            return Judgment(
                "HALLUCINATION",
                f"항목 '{item}'의 근거가 자료에 없음 — 지어낸 내용 가능",
                sim if sim is not None else -1, grounded, "red")
