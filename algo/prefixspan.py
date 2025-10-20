# algo/prefixspan.py
# Parallel-friendly PrefixSpan (CPU, no GPU)
# - Multiprocessing-safe: all worker functions are module-level (no bound self)
# - Root-branch parallelization for single-item-per-itemset DBs
# - Parallel 1-frequent item counting
# - Rich debug logging (set log_level='DEBUG' for verbose)

from __future__ import annotations

import os
import tempfile
import multiprocessing as mp
from collections import defaultdict
from math import ceil
from time import time
from typing import Dict, List, Tuple, Iterable, Optional
import logging
import traceback

from utils.sequence_utils import SequenceDatabase, PseudoSequence


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("prefixspan")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.propagate = False
logger.setLevel(logging.INFO)


def _set_logger_level(level: str | int):
    try:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    except Exception:
        logger.setLevel(logging.INFO)


# ============================================================
# Small logging helpers
# ============================================================

def _preview_items_counts(d: Dict[int, List[PseudoSequence]] | Dict[int, List[int]],
                          top_k: int = 10) -> str:
    """Return a compact preview of (item: support) pairs sorted by support desc."""
    try:
        items = []
        for it, lst in d.items():
            sup = len(lst)
            items.append((it, sup))
        items.sort(key=lambda x: x[1], reverse=True)
        return ", ".join(f"{it}:{sup}" for it, sup in items[:top_k])
    except Exception:
        return "(preview-error)"


def _buffer_to_pattern_tokens(buf: List[int], end_idx: int) -> str:
    """Human-friendly pattern preview from buffer tokens."""
    try:
        pat, cur = [], []
        for i in range(end_idx + 1):
            tok = buf[i]
            if tok == -1:
                if cur:
                    pat.append("{" + " ".join(map(str, cur)) + "}")
                    cur = []
            elif tok > 0:
                cur.append(tok)
        if cur:
            pat.append("{" + " ".join(map(str, cur)) + "}")
        return " ".join(pat)
    except Exception:
        return "(buffer-preview-error)"


# ============================================================
# Module-level workers (must be picklable on Windows 'spawn')
# ============================================================

def _count_seen_and_multi_worker(args: Tuple[int, List[int]]) -> Tuple[int, set, bool]:
    """
    For one sequence: return (sid, set_of_unique_items_in_sequence, contains_multi_item_itemset)
    - Support is sequence-based: each item counted at most once per sequence.
    """
    sid, seq = args
    seen = set()
    contains_multi = False
    cnt_in_itemset = 0
    for tok in seq:
        if tok > 0:
            if tok not in seen:
                seen.add(tok)
            cnt_in_itemset += 1
            if cnt_in_itemset > 1:
                contains_multi = True
        elif tok == -1:
            cnt_in_itemset = 0
        elif tok == -2:
            break
    return sid, seen, contains_multi


def _process_root_single_worker(args: Tuple[int, List[int], int, int, int, str, int, str, int, int]) -> Tuple[str, int]:
    """
    Pure worker for a single root item branch (single-item-per-itemset DB path).
    args = (item, sids, minsupp_abs, max_len, min_len, input_csv_path, pretty_pad,
            log_level, log_every, max_preview_items)
    Writes patterns to a temp file and returns (temp_path, pattern_count_in_file).
    """
    _lvl, _log_every, _max_prev = args[-3], args[-2], args[-1]
    _set_logger_level(_lvl)
    item, sids, minsupp_abs, max_len, min_len, input_csv_path, pretty_pad = args[:7]

    logger.debug(f"[worker] start item={item}, |sids|={len(sids)}, minsupp_abs={minsupp_abs}, "
                 f"max_len={max_len}, min_len={min_len}")

    try:
        db = SequenceDatabase()
        db.load_csv_tokens(input_csv_path)

        buffer = [0] * 2000

        # Build projected DB for this root item
        proj: List[PseudoSequence] = []
        for sid in sids:
            seq = db.get(sid)
            if seq is None:
                continue
            j = 0
            n = len(seq)
            while j < n and seq[j] != -2:
                if seq[j] == item:
                    if j + 1 < n and seq[j + 1] != -2:
                        proj.append(PseudoSequence(sid, j + 1))
                    break
                j += 1

        fd, tmp_path = tempfile.mkstemp(prefix=f"ps_{item}_", suffix=".txt")
        os.close(fd)
        writer = open(tmp_path, "w", encoding="utf-8")

        pattern_count = 0
        if min_len <= 1:
            writer.write(f"{{{item}}}".ljust(pretty_pad) + f" #SUP: {len(sids)}\n")
            pattern_count += 1

        def write_pattern_tokens(end_idx: int, pseqs: List[PseudoSequence]):
            nonlocal pattern_count
            # min_len by counting positive tokens in buffer[0..end_idx]
            length = 0
            for ii in range(end_idx + 1):
                if buffer[ii] > 0:
                    length += 1
            if length < min_len:
                return

            pattern = []
            cur = []
            for ii in range(end_idx + 1):
                tok = buffer[ii]
                if tok == -1:
                    if cur:
                        pattern.append("{" + " ".join(map(str, cur)) + "}")
                        cur = []
                elif tok > 0:
                    cur.append(tok)
            if cur:
                pattern.append("{" + " ".join(map(str, cur)) + "}")
            writer.write(" ".join(pattern).ljust(pretty_pad) + f" #SUP: {len(pseqs)}\n")
            pattern_count += 1

        def recursion_single(database: List[PseudoSequence], k: int, last_pos: int):
            items_to_pseqs: Dict[int, List[PseudoSequence]] = {}
            for pseq in database:
                sid = pseq.sequence_id
                seq = db.get(sid)
                i = pseq.index_first_item
                n = len(seq)
                while i < n and seq[i] != -2:
                    tok = seq[i]
                    if tok > 0:
                        lst = items_to_pseqs.setdefault(tok, [])
                        if not lst or lst[-1].sequence_id != sid:
                            lst.append(PseudoSequence(sid, i + 1))
                    i += 1

            if k % _log_every == 0:
                logger.debug(f"[worker] item={item} depth={k} candidates={len(items_to_pseqs)} "
                             f"sample=({_preview_items_counts(items_to_pseqs, _max_prev)})")

            for it, pseqs in items_to_pseqs.items():
                if len(pseqs) >= minsupp_abs:
                    buffer[last_pos + 1] = -1
                    buffer[last_pos + 2] = it
                    write_pattern_tokens(last_pos + 2, pseqs)
                    if k < max_len:
                        recursion_single(pseqs, k + 1, last_pos + 2)

        if max_len > 1:
            buffer[0] = item
            recursion_single(proj, 2, 0)

        writer.close()

        # Count patterns in temp file
        cnt = 0
        with open(tmp_path, "r", encoding="utf-8") as f:
            for _ in f:
                cnt += 1

        logger.debug(f"[worker] done item={item}, patterns_in_file={cnt}, proj_size={len(proj)}")
        return tmp_path, cnt

    except Exception as e:
        logger.error(f"[worker] item={item} crashed: {e}\n{traceback.format_exc()}")
        fd, tmp_path = tempfile.mkstemp(prefix=f"ps_{item}_err_", suffix=".txt")
        os.close(fd)
        return tmp_path, 0


# ============================================================
# PrefixSpan class
# ============================================================

class PrefixSpan:
    def __init__(
        self,
        maximum_pattern_length: int = 1000,
        min_len: int = 1,
        n_jobs: int = 1,
        log_level: str | int = "INFO",
        log_every: int = 1,            # DEBUG: log every k-depth step (>=1)
        max_preview_items: int = 10,   # DEBUG preview top-k items
        # -------- INFO-friendly knobs --------
        info_depth_every: int = 5,     # INFO: every k-depth, print compact candidates
        info_pattern_every: int = 5000,# INFO: every N patterns, print progress
        info_preview_items: int = 5,   # INFO: preview top-k in INFO logs
    ):
        self.total_time_ms = 0
        self.pattern_count = 0
        self.max_len = maximum_pattern_length
        self.min_len = max(1, int(min_len))
        self.n_jobs = max(1, int(n_jobs))

        self._minsupp_abs = 1
        self._db: SequenceDatabase | None = None
        self._sequence_count = 0
        self._contains_itemsets_with_multiple_items = False
        self._buffer = [0] * 2000

        self._writer = None
        self._pretty_pad = 30
        self._input_csv_path: str | None = None

        # logging controls
        self._log_level = log_level
        self._log_every = max(1, int(log_every))
        self._max_preview_items = max(1, int(max_preview_items))

        # INFO knobs
        self._info_depth_every = max(1, int(info_depth_every))
        self._info_pattern_every = max(1, int(info_pattern_every))
        self._info_preview_items = max(1, int(info_preview_items))

        _set_logger_level(self._log_level)

        logger.info(f"PrefixSpan init: max_len={self.max_len}, min_len={self.min_len}, "
                    f"n_jobs={self.n_jobs}, log_level={self._log_level}, "
                    f"info_depth_every={self._info_depth_every}, "
                    f"info_pattern_every={self._info_pattern_every}")

    # ---------------------- Public API ----------------------
    def run(self, input_csv: str, minsup_relative: float, output_file: str):
        if not output_file:
            raise ValueError("output_file must be provided")

        start = time()
        self.pattern_count = 0
        self._input_csv_path = input_csv

        logger.info(f"Loading DB from: {input_csv}")
        self._db = SequenceDatabase()
        self._db.load_csv_tokens(input_csv)
        self._sequence_count = self._db.size()
        logger.info(f"DB loaded: sequences={self._sequence_count}")

        self._minsupp_abs = int(ceil(minsup_relative * self._sequence_count))
        if self._minsupp_abs <= 0:
            self._minsupp_abs = 1
        logger.info(f"minsup_relative={minsup_relative} -> minsupp_abs={self._minsupp_abs}")

        self._prefixspan(output_file)

        self.total_time_ms = int((time() - start) * 1000)

        if self._writer:
            self._writer.close()
            self._writer = None

        logger.info(f"Finished. patterns={self.pattern_count}, time_ms={self.total_time_ms}")
        self._db = None

    # ---------------------- Core driver ----------------------
    def _prefixspan(self, output_file: str):
        self._writer = open(output_file, "w", encoding="utf-8")
        logger.info("Step1: counting 1-frequent items & multi-itemset detection...")
        map_item_to_sids = self._find_sequences_containing_items()
        logger.info(f"1-frequent items: {len(map_item_to_sids)} "
                    f"(contains_multi_itemsets={self._contains_itemsets_with_multiple_items}) "
                    f"sample=({_preview_items_counts(map_item_to_sids, self._max_preview_items)})")

        logger.info("Step2: pruning infrequent items & branching...")
        if self._contains_itemsets_with_multiple_items:
            self._prefixspan_with_multiple_items(map_item_to_sids)
        else:
            self._prefixspan_with_single_items(map_item_to_sids)

    # ---------------------- Utilities ----------------------
    def _pattern_length_in_buffer(self, tokens_end_idx: int) -> int:
        length = 0
        for i in range(tokens_end_idx + 1):
            if self._buffer[i] > 0:
                length += 1
        return length

    def _write_pattern_tokens(self, tokens_end_idx: int, pseudo_sequences: List[PseudoSequence]):
        if self._pattern_length_in_buffer(tokens_end_idx) < self.min_len:
            return

        self.pattern_count += 1
        support = len(pseudo_sequences)
        pattern_str = _buffer_to_pattern_tokens(self._buffer, tokens_end_idx).ljust(self._pretty_pad)
        self._writer.write(f"{pattern_str} #SUP: {support}\n")

        # INFO: periodic progress
        if self.pattern_count % self._info_pattern_every == 0:
            logger.info(f"[write] patterns={self.pattern_count} last={pattern_str.strip()} sup={support}")

        # DEBUG: always
        logger.debug(f"[write] {pattern_str.strip()} #SUP: {support}")

    def _write_single(self, item: int, support: int):
        if self.min_len > 1:
            return
        self.pattern_count += 1
        pattern_str = f"{{{item}}}".ljust(self._pretty_pad)
        self._writer.write(f"{pattern_str} #SUP: {support}\n")
        logger.debug(f"[write-single] {{{item}}} #SUP: {support}")

    def _write_concat(self, output_file: str, tmp_files: List[str]):
        logger.info(f"Concatenating {len(tmp_files)} temp files -> {output_file}")
        with open(output_file, "w", encoding="utf-8") as out:
            for p in tmp_files:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {p}: {e}")

    # ---------------------- Step 1: find 1-freq items ----------------------
    def _find_sequences_containing_items(self) -> Dict[int, List[int]]:
        assert self._db is not None
        m: Dict[int, List[int]] = defaultdict(list)
        contains_multi = False

        tasks = [(sid, self._db.get(sid)) for sid in range(self._db.size())
                 if self._db.get(sid) is not None]

        n_jobs = min(self.n_jobs, mp.cpu_count())
        logger.info(f"Counting 1-freq with n_jobs={n_jobs}, tasks={len(tasks)}")

        if n_jobs <= 1 or len(tasks) < 2:
            for idx, (sid, seq) in enumerate(tasks, 1):
                seen = set()
                cnt_in_itemset = 0
                for tok in seq:
                    if tok > 0:
                        if tok not in seen:
                            seen.add(tok)
                        cnt_in_itemset += 1
                        if cnt_in_itemset > 1:
                            contains_multi = True
                    elif tok == -1:
                        cnt_in_itemset = 0
                    elif tok == -2:
                        break
                for it in seen:
                    m[it].append(sid)
                if idx % 100 == 0:
                    logger.info(f"[1-freq] processed {idx}/{len(tasks)} sequences")
        else:
            with mp.Pool(processes=n_jobs) as pool:
                for idx, (sid, seen, has_multi) in enumerate(
                        pool.imap_unordered(_count_seen_and_multi_worker, tasks, chunksize=64), 1):
                    if has_multi:
                        contains_multi = True
                    for it in seen:
                        m[it].append(sid)
                    if idx % 100 == 0:
                        logger.info(f"[1-freq] worker processed {idx}/{len(tasks)} sequences")

        self._contains_itemsets_with_multiple_items = contains_multi
        return m

    # ---------------------- Step 2A: single-item-per-itemset path ----------------------
    def _prefixspan_with_single_items(self, map_item_to_sids: Dict[int, List[int]]):
        assert self._db is not None

        logger.info("Pruning (single-item-per-itemset path)...")
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

        # integrity check
        for sid in range(self._db.size()):
            seq = self._db.get(sid)
            if seq is None:
                continue
            if not (len(seq) > 0 and seq[-1] == -2):
                logger.error(f"sid {sid} sequence missing -2 after pruning")
                raise ValueError(f"sid {sid} sequence missing -2 after pruning")

        roots = [(item, sids) for item, sids in map_item_to_sids.items()
                 if len(sids) >= self._minsupp_abs]
        logger.info(f"Roots: {len(roots)} sample=({_preview_items_counts(dict(roots), self._max_preview_items)})")

        # Single-thread path
        if self.n_jobs <= 1 or len(roots) < 2:
            logger.info("Root traversal in single thread.")
            for idx_root, (item, sids) in enumerate(roots, 1):
                patterns_before = self.pattern_count
                self._write_single(item, len(sids))
                if self.max_len > 1:
                    self._buffer[0] = item
                    projected = self._build_projected_single_items(item, sids)
                    logger.info(f"[root] start item={item} proj_size={len(projected)} depth_start=2 ({idx_root}/{len(roots)})")
                    self._recursion_single_items(projected, k=2, last_buf_pos=0)
                produced = self.pattern_count - patterns_before
                logger.info(f"[root] done  item={item} produced_patterns={produced} total_patterns={self.pattern_count}")
            return

        # Parallel root branches
        if not self._input_csv_path:
            raise RuntimeError("Input CSV path is missing for parallel workers.")

        logger.info(f"Root traversal in parallel: n_jobs={min(self.n_jobs, mp.cpu_count())}")
        args = [
            (item, sids, self._minsupp_abs, self.max_len, self.min_len,
             self._input_csv_path, self._pretty_pad, self._log_level, self._log_every, self._max_preview_items)
            for item, sids in roots
        ]

        tmp_files: List[str] = []
        total_patterns = 0
        with mp.Pool(processes=min(self.n_jobs, mp.cpu_count())) as pool:
            for idx, (tmp_path, cnt) in enumerate(
                    pool.imap_unordered(_process_root_single_worker, args, chunksize=1), 1):
                tmp_files.append(tmp_path)
                total_patterns += cnt
                if idx % 10 == 0 or idx == len(args):
                    logger.info(f"[roots-parallel] collected {idx}/{len(args)} temp files, "
                                f"patterns_so_far(+files)={total_patterns}")

        self._writer.close()
        self._write_concat(self._writer.name, tmp_files)
        self.pattern_count += total_patterns
        self._writer = open(self._writer.name, "a", encoding="utf-8")

    def _build_projected_single_items(self, item: int, sids: List[int]) -> List[PseudoSequence]:
        assert self._db is not None
        proj: List[PseudoSequence] = []
        for sid in sids:
            seq = self._db.get(sid)
            if seq is None:
                continue
            j = 0
            n = len(seq)
            while j < n and seq[j] != -2:
                if seq[j] == item:
                    if j + 1 < n and seq[j + 1] != -2:
                        proj.append(PseudoSequence(sid, j + 1))
                    break
                j += 1
        return proj

    def _recursion_single_items(self, database: List[PseudoSequence], k: int, last_buf_pos: int):
        assert self._db is not None
        items_to_pseqs: Dict[int, List[PseudoSequence]] = {}
        for pseq in database:
            sid = pseq.sequence_id
            seq = self._db.get(sid)
            i = pseq.index_first_item
            n = len(seq)
            while i < n and seq[i] != -2:
                tok = seq[i]
                if tok > 0:
                    lst = items_to_pseqs.setdefault(tok, [])
                    if not lst or lst[-1].sequence_id != sid:
                        lst.append(PseudoSequence(sid, i + 1))
                i += 1

        # INFO: compact progress each few depths
        if k % self._info_depth_every == 0:
            logger.info(f"[single] depth={k} candidates={len(items_to_pseqs)} "
                        f"sample=({_preview_items_counts(items_to_pseqs, self._info_preview_items)}) "
                        f"prefix=({_buffer_to_pattern_tokens(self._buffer, last_buf_pos)})")

        # DEBUG: detailed
        if k % self._log_every == 0:
            logger.debug(f"[single] depth={k} candidates={len(items_to_pseqs)} "
                         f"sample=({_preview_items_counts(items_to_pseqs, self._max_preview_items)}) "
                         f"prefix=({_buffer_to_pattern_tokens(self._buffer, last_buf_pos)})")

        for item, pseqs in items_to_pseqs.items():
            if len(pseqs) >= self._minsupp_abs:
                self._buffer[last_buf_pos + 1] = -1
                self._buffer[last_buf_pos + 2] = item
                self._write_pattern_tokens(last_buf_pos + 2, pseqs)
                if k < self.max_len:
                    self._recursion_single_items(pseqs, k + 1, last_buf_pos + 2)

    # ---------------------- Step 2B: multi-itemset path (i- & s-extensions) ----------------------
    def _prefixspan_with_multiple_items(self, map_item_to_sids: Dict[int, List[int]]):
        assert self._db is not None

        logger.info("Pruning (multi-itemset path)...")
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

        roots = [(item, sids) for item, sids in map_item_to_sids.items()
                 if len(sids) >= self._minsupp_abs]
        logger.info(f"Roots (multi): {len(roots)} sample=({_preview_items_counts(dict(roots), self._max_preview_items)})")

        for idx_root, (item, sids) in enumerate(roots, 1):
            patterns_before = self.pattern_count
            sup = len(sids)
            self._write_single(item, sup)
            if self.max_len > 1:
                self._buffer[0] = item
                projected = self._build_projected_first_time_multi(item, sids)
                logger.info(f"[root-multi] start item={item} proj_size={len(projected)} depth_start=2 ({idx_root}/{len(roots)})")
                self._recursion_multi(projected, k=2, last_buf_pos=0)
            produced = self.pattern_count - patterns_before
            logger.info(f"[root-multi] done  item={item} produced_patterns={produced} total_patterns={self.pattern_count}")

    def _build_projected_first_time_multi(self, item: int, sids: List[int]) -> List[PseudoSequence]:
        assert self._db is not None
        proj: List[PseudoSequence] = []
        for sid in sids:
            seq = self._db.get(sid)
            if seq is None:
                continue
            j = 0
            n = len(seq)
            while j < n and seq[j] != -2:
                if seq[j] == item:
                    end = False
                    if j + 1 >= n or (seq[j + 1] == -2) or (j + 2 < n and seq[j + 1] == -1 and seq[j + 2] == -2):
                        end = True
                    if not end and j + 1 < n:
                        proj.append(PseudoSequence(sid, j + 1))
                    break
                j += 1
        return proj

    def _recursion_multi(self, database: List[PseudoSequence], k: int, last_buf_pos: int):
        assert self._db is not None

        first_pos_last_itemset = last_buf_pos
        while first_pos_last_itemset > 0 and self._buffer[first_pos_last_itemset - 1] != -1:
            first_pos_last_itemset -= 1

        map_postfix: Dict[int, List[PseudoSequence]] = {}  # i-extension
        map_normal: Dict[int, List[PseudoSequence]] = {}   # s-extension

        for pseq in database:
            sid = pseq.sequence_id
            seq = self._db.get(sid)

            prev_tok = seq[pseq.index_first_item - 1]
            current_itemset_is_postfix = (prev_tok != -1)
            is_first_itemset = True

            position_to_match = first_pos_last_itemset

            i = pseq.index_first_item
            n = len(seq)
            while i < n and seq[i] != -2:
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

        # INFO: compact progress each few depths
        if k % self._info_depth_every == 0:
            logger.info(f"[multi] depth={k} i-ext={len(map_postfix)} s-ext={len(map_normal)} "
                        f"i-sample=({_preview_items_counts(map_postfix, self._info_preview_items)}) "
                        f"s-sample=({_preview_items_counts(map_normal, self._info_preview_items)}) "
                        f"prefix=({_buffer_to_pattern_tokens(self._buffer, last_buf_pos)})")

        # DEBUG: detailed
        if k % self._log_every == 0:
            logger.debug(f"[multi] depth={k} i-ext={len(map_postfix)} s-ext={len(map_normal)} "
                         f"i-sample=({_preview_items_counts(map_postfix, self._max_preview_items)}) "
                         f"s-sample=({_preview_items_counts(map_normal, self._max_preview_items)}) "
                         f"prefix=({_buffer_to_pattern_tokens(self._buffer, last_buf_pos)})")

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
    # Example (CLI): adjust paths as needed
    algo = PrefixSpan(
        maximum_pattern_length=10,
        min_len=1,
        n_jobs=4,
        log_level="INFO",        # "DEBUG" for very verbose logging
        log_every=2,             # DEBUG depth cadence
        max_preview_items=5,     # DEBUG preview size
        info_depth_every=5,      # INFO depth cadence
        info_pattern_every=2000, # INFO pattern cadence
        info_preview_items=5,    # INFO preview size
    )
    # algo.run("database/small.csv", minsup_relative=0.5, output_file="patterns.txt")
    pass
