class SequenceDatabase:
    def __init__(self):
        self.sequences = []  # list[list[int]]

    def load_csv_tokens(self, input_csv: str):
        """
        CSV 포맷 (SPMF 토큰 그대로):
        - 각 줄이 하나의 시퀀스
        - 아이템·구분자·끝표시를 콤마로 구분: 예) 3,-1,1,-1,2,-1,-2
        - -1: itemset 구분자, -2: 시퀀스 종료
        """
        with open(input_csv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = [int(x.strip()) for x in line.split(",") if x.strip() != ""]
                # 끝이 -2로 끝나지 않으면 보정
                if toks and toks[-1] != -2:
                    toks.append(-2)
                self.sequences.append(toks)

    def size(self) -> int:
        return len(self.sequences)

    def get(self, idx: int):
        return self.sequences[idx]


class PseudoSequence:
    __slots__ = ("sequence_id", "index_first_item")
    def __init__(self, sequence_id: int, index_first_item: int):
        self.sequence_id = sequence_id
        self.index_first_item = index_first_item
