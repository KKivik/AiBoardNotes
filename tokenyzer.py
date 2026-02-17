import pickle

class Tokenyzer:
    def __init__(self):
        with open('tokenyzer/byte_to_idx.pkl', 'rb') as f:
            self.merges = pickle.load(f)

        with open('tokenyzer/idx_to_byte.pkl', 'rb') as f:
            self.vocab = pickle.load(f)

    def decode(self, ids):
        # ids - list of integers [1, 265, 3]
        tokens = b"".join(self.vocab[x] for x in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        tokens = list(text.encode("utf-8"))  # [5, 167, 258]
        while len(tokens) >= 2:
            # run and count all consecutive pairs in bt
            stats = []
            for i, j in zip(tokens, tokens[1:]):
                stats.append((i, j))
            # next, we find pair with minimum idx in merges, to merge from up to down
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(idx, pair, tokens)
        return tokens

    def merge(self, idx, focus, txt_copy):
        first, second = focus
        repl_txt = []
        txt_copy = [txt_copy]
        for label in txt_copy:
            temp = []
            i = 0
            while i < len(label):
                if i + 1 < len(label) and label[i] == first and label[i + 1] == second:
                    temp.append(idx)
                    i += 2
                else:
                    temp.append(label[i])
                    i += 1
            repl_txt.append(temp)

        return repl_txt[0]

if __name__ == "__main__":
    Tkn = Tokenyzer()
    print(Tkn.decode(Tkn.encode("lalala")))

