import random
import json


class DeterministicLabeler:
    def __init__(self, label_length: int = 2, max_length: int = 7, min_length: int = 3):
        self.label_length = label_length
        with open("words.txt", "r") as infile:
            self.wordlist = [
                w
                for w in infile.read().splitlines()
                if min_length <= len(w) <= max_length
            ]

    def label(self, model) -> str:
        json_repr = json.loads(model.to_json())
        _remove_key(json_repr, "name")

        random.seed(int.from_bytes(str(json_repr).encode(), "big"))

        words = random.sample(self.wordlist, self.label_length)
        return "-".join((word.replace("-", "") for word in words)).lower()

    def __call__(self, *args, **kwargs):
        return self.label(*args)


def _remove_key(d, remove_key_):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key_:
                del d[key]
            else:
                _remove_key(d[key], remove_key_)
    if isinstance(d, list):
        for i in range(len(d)):
            _remove_key(d[i], remove_key_)
