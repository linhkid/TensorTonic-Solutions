from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.id_to_word = {i: t for t, i in self.word_to_id.items()}

        next_id = 4
        for text in texts:
            for token in text.split():
                if token not in self.word_to_id:
                    self.word_to_id[token] = next_id
                    self.id_to_word[next_id] = token
                    next_id += 1

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.word_to_id.get(token, self.word_to_id[self.unk_token]) for token in text.split()]

    def decode(self, ids: List[int]) -> str:
        # Usually tests expect raw text, not special tokens
        
        final = [self.id_to_word.get(i, self.unk_token) for i in ids]
        return " ".join(final)
