def get_stats(ids: list) -> dict:
    stats = dict()
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def merge(ids: list, target_pair: tuple, new_token: int) -> list:
    new_ids, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == target_pair:
            new_ids.append(new_token)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BasicTokenizer:
    
    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> byteobject

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """ Trains Tokenizer on given text filling up self.merges and self.vocab. """
        num_merges = vocab_size - 256
        ids = list(text.encode('utf-8'))

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose:
                print(f'pair {pair} gets merged to --> {idx}')
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        # Save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    def encode(self, text: str) -> list[int]:
        """ Encodes text into integer sequence using self.merges dictionary. """
        ids = list(text.encode('utf-8'))

        while len(ids) >= 2:
            pairs = get_stats(ids)
            pair = min(pairs, key=lambda k: self.merges.get(k, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """ Decodes given integer sequence back to original text. """ 
        byte_tokens = b"".join(self.vocab[idx] for idx in ids)
        text = byte_tokens.decode('utf-8', errors='replace')
        return text
