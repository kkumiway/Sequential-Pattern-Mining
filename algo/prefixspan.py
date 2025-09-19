from math import ceil
from time import time

from utils.sequence_utils import SequenceDatabase, PseudoSequence

class PrefixSpan:
    def __init__(self, maximum_pattern_length: int = 1000, min_len: int = 1):
        self.total_time_ms = 0
        self.pattern_count = 0
        self.max_len = maximum_pattern_length
        self.min_len = max(1, int(min_len))

        self._minsupp_abs = 1
        self._db = None
        self._sequence_count = 0
        self._contains_itemsets_with_multiple_items = False
        self._buffer = [0] * 2000

        self._writer = None
        self._pretty_pad = 30 

    # ---------------------- Public API ----------------------
    def run(self, input_csv: str, minsup_relative: float, output_file: str):
        """
        input_csv: SPMF 토큰(-1/-2 포함)을 콤마로 구분한 CSV
        output_file: .txt 파일
        """
        if not output_file:
            raise ValueError("output_file must be provided")

        start = time()
        self.pattern_count = 0

        # load DB (CSV tokens)
        self._db = SequenceDatabase()
        self._db.load_csv_tokens(input_csv)
        self._sequence_count = self._db.size()

        # relative -> absolute minsup
        self._minsupp_abs = int(ceil(minsup_relative * self._sequence_count))
        if self._minsupp_abs <= 0:
            self._minsupp_abs = 1

        # dispatch
        self._prefixspan(output_file)

        # finalize
        self.total_time_ms = int((time() - start) * 1000)

        if self._writer:
            self._writer.close()
            self._writer = None

        self._db = None  # detach

    # ---------------------- Core driver ----------------------
    def _prefixspan(self, output_file: str):
        self._writer = open(output_file, "w", encoding="utf-8")

        # 1) count 1-length frequent items + detect multi-item itemsets
        map_item_to_sids = self._find_sequences_containing_items()

        # 2) prune infrequent items from DB (in-place), then branch
        if self._contains_itemsets_with_multiple_items:
            self._prefixspan_with_multiple_items(map_item_to_sids)
        else:
            self._prefixspan_with_single_items(map_item_to_sids)

    # ---------------------- Utilities ----------------------
    def _pattern_length_in_buffer(self, tokens_end_idx: int) -> int:
        """버퍼에 기록된 현재 패턴의 아이템 개수(양수 토큰 수)를 길이로 정의"""
        length = 0
        for i in range(tokens_end_idx + 1):
            if self._buffer[i] > 0:
                length += 1
        return length

    def _write_pattern_tokens(self, tokens_end_idx: int, pseudo_sequences: list[PseudoSequence]):
        # min_len 필터: 출력 단계에서만 적용 (탐색/재귀는 그대로)
        if self._pattern_length_in_buffer(tokens_end_idx) < self.min_len:
            return

        self.pattern_count += 1
        support = len(pseudo_sequences)

        pattern = []
        current_itemset = []

        for i in range(tokens_end_idx + 1):
            tok = self._buffer[i]
            if tok == -1:
                if current_itemset:
                    pattern.append(f"{{{' '.join(map(str, current_itemset))}}}")
                    current_itemset = []
            elif tok > 0:
                current_itemset.append(tok)

        if current_itemset:
            pattern.append(f"{{{' '.join(map(str, current_itemset))}}}")

        pattern_str = " ".join(pattern).ljust(self._pretty_pad)
        self._writer.write(f"{pattern_str} #SUP: {support}\n")

    def _write_single(self, item: int, support: int):
        # 단일 아이템 패턴은 길이 1, min_len > 1 이면 출력 생략
        if self.min_len > 1:
            return
        self.pattern_count += 1
        pattern_str = f"{{{item}}}".ljust(self._pretty_pad)
        self._writer.write(f"{pattern_str} #SUP: {support}\n")

    # ---------------------- Step 1: find 1-freq items ----------------------
    def _find_sequences_containing_items(self) -> dict[int, list[int]]:
        m: dict[int, list[int]] = {}
        self._contains_itemsets_with_multiple_items = False

        for sid in range(self._db.size()):
            seq = self._db.get(sid)
            count_in_itemset = 0
            seen_in_this_seq = set()
            for tok in seq:
                if tok > 0:
                    if tok not in seen_in_this_seq:
                        m.setdefault(tok, []).append(sid) 
                        seen_in_this_seq.add(tok)
                    count_in_itemset += 1
                    if count_in_itemset > 1:
                        self._contains_itemsets_with_multiple_items = True
                elif tok == -1:
                    count_in_itemset = 0
                elif tok == -2:
                    break
        return m

    # ---------------------- Step 2A: single-item-per-itemset path ----------------------
    def _prefixspan_with_single_items(self, map_item_to_sids: dict[int, list[int]]):
        # prune infrequent items from sequences (in-place)
        for i in range(self._db.size()):
            seq = self._db.get(i)
            if seq is None:
                continue
            write_pos = 0
            for j, tok in enumerate(seq):
                if tok > 0:
                    if len(map_item_to_sids.get(tok, ())) >= self._minsupp_abs:
                        seq[write_pos] = tok
                        write_pos += 1
                elif tok == -2:
                    if write_pos > 0:
                        seq[write_pos] = -2
                        self._db.sequences[i] = seq[: write_pos + 1]
                    else:
                        self._db.sequences[i] = None
                    break

        for item, sids in map_item_to_sids.items():
            sup = len(sids)
            if sup >= self._minsupp_abs:
                self._write_single(item, sup)

                if self.max_len > 1:
                    self._buffer[0] = item
                    projected = self._build_projected_single_items(item, sids)
                    self._recursion_single_items(projected, k=2, last_buf_pos=0)

    def _build_projected_single_items(self, item: int, sids: list[int]) -> list[PseudoSequence]:
        proj = []
        for sid in sids:
            seq = self._db.get(sid)
            if seq is None:
                continue
            j = 0
            while seq[j] != -2:
                if seq[j] == item:
                    if seq[j + 1] != -2:
                        proj.append(PseudoSequence(sid, j + 1))
                    break
                j += 1
        return proj

    def _recursion_single_items(self, database: list[PseudoSequence], k: int, last_buf_pos: int):
        items_to_pseqs: dict[int, list[PseudoSequence]] = {}
        for pseq in database:
            sid = pseq.sequence_id
            seq = self._db.get(sid)
            i = pseq.index_first_item
            while seq[i] != -2:
                tok = seq[i]
                if tok > 0:
                    lst = items_to_pseqs.setdefault(tok, [])
                    if not lst or lst[-1].sequence_id != sid:
                        lst.append(PseudoSequence(sid, i + 1))
                i += 1

        for item, pseqs in items_to_pseqs.items():
            if len(pseqs) >= self._minsupp_abs:
                self._buffer[last_buf_pos + 1] = -1
                self._buffer[last_buf_pos + 2] = item
                self._write_pattern_tokens(last_buf_pos + 2, pseqs)
                if k < self.max_len:
                    self._recursion_single_items(pseqs, k + 1, last_buf_pos + 2)

    # ---------------------- Step 2B: multi-itemset path (i- & s-extensions) ----------------------
    def _prefixspan_with_multiple_items(self, map_item_to_sids: dict[int, list[int]]):
        for i in range(self._db.size()):
            seq = self._db.get(i)
            if seq is None:
                continue
            write_pos = 0
            count_in_itemset = 0
            for j, tok in enumerate(seq):
                if tok > 0:
                    if len(map_item_to_sids.get(tok, ())) >= self._minsupp_abs:
                        seq[write_pos] = tok
                        write_pos += 1
                        count_in_itemset += 1
                elif tok == -1:
                    if count_in_itemset > 0:
                        seq[write_pos] = -1
                        write_pos += 1
                        count_in_itemset = 0
                elif tok == -2:
                    if write_pos > 0:
                        seq[write_pos] = -2
                        self._db.sequences[i] = seq[: write_pos + 1]
                    else:
                        self._db.sequences[i] = None
                    break

        for item, sids in map_item_to_sids.items():
            sup = len(sids)
            if sup >= self._minsupp_abs:
                self._write_single(item, sup)
                if self.max_len > 1:
                    self._buffer[0] = item
                    projected = self._build_projected_first_time_multi(item, sids)
                    self._recursion_multi(projected, k=2, last_buf_pos=0)

    def _build_projected_first_time_multi(self, item: int, sids: list[int]) -> list[PseudoSequence]:
        proj = []
        for sid in sids:
            seq = self._db.get(sid)
            if seq is None:
                continue
            j = 0
            while seq[j] != -2:
                if seq[j] == item:
                    end = False
                    if seq[j + 1] == -2:
                        end = True
                    elif seq[j + 1] == -1 and seq[j + 2] == -2:
                        end = True
                    if not end:
                        proj.append(PseudoSequence(sid, j + 1))
                    break
                j += 1
        return proj

    def _recursion_multi(self, database: list[PseudoSequence], k: int, last_buf_pos: int):
        first_pos_last_itemset = last_buf_pos
        while first_pos_last_itemset > 0 and self._buffer[first_pos_last_itemset - 1] != -1:
            first_pos_last_itemset -= 1

        map_postfix: dict[int, list[PseudoSequence]] = {} # i-extension 후보 (같은 아이템셋에 붙이기)
        map_normal: dict[int, list[PseudoSequence]] = {} # s-extension 후보 (새 아이템셋 시작)

        for pseq in database:
            sid = pseq.sequence_id
            seq = self._db.get(sid)

            prev_tok = seq[pseq.index_first_item - 1]
            current_itemset_is_postfix = (prev_tok != -1)
            is_first_itemset = True

            position_to_match = first_pos_last_itemset

            i = pseq.index_first_item
            while seq[i] != -2:
                tok = seq[i]
                if tok > 0:
                    target = map_postfix if current_itemset_is_postfix else map_normal
                    lst = target.setdefault(tok, [])
                    if not lst or lst[-1].sequence_id != sid:
                        lst.append(PseudoSequence(sid, i + 1))

                    if current_itemset_is_postfix and not is_first_itemset:
                        lst2 = map_normal.setdefault(tok, [])
                        if not lst2 or lst2[-1].sequence_id != sid:
                            lst2.append(PseudoSequence(sid, i + 1))

                    if (not current_itemset_is_postfix) and self._buffer[position_to_match] == tok:
                        position_to_match += 1
                        if position_to_match > last_buf_pos:
                            current_itemset_is_postfix = True

                elif tok == -1:
                    is_first_itemset = False
                    current_itemset_is_postfix = False
                    position_to_match = first_pos_last_itemset

                i += 1

        for item, pseqs in map_postfix.items():
            if len(pseqs) >= self._minsupp_abs:
                new_pos = last_buf_pos + 1
                self._buffer[new_pos] = item
                self._write_pattern_tokens(new_pos, pseqs)
                if k < self.max_len:
                    self._recursion_multi(pseqs, k + 1, new_pos)

        for item, pseqs in map_normal.items():
            if len(pseqs) >= self._minsupp_abs:
                new_pos = last_buf_pos + 1
                self._buffer[new_pos] = -1
                self._buffer[new_pos + 1] = item
                self._write_pattern_tokens(new_pos + 1, pseqs)
                if k < self.max_len:
                    self._recursion_multi(pseqs, k + 1, new_pos + 1)


if __name__ == "__main__":
    algo = PrefixSpan(maximum_pattern_length=10, min_len=1)
    # algo.run("small.csv", minsup_relative=0.5, output_file="patterns.txt")
    pass
