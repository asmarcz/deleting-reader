"""Word-level diff that ignores case and punctuation."""
from difflib import SequenceMatcher

_PUNCTUATION = str.maketrans("", "", ",.;:\"'-()[]{}?!“”‘’")


def clean(word):
    """A word as the diff compares it: lowercase, without punctuation."""
    return word.lower().translate(_PUNCTUATION)


def opcodes(a, b):
    """difflib opcodes between the words of texts a and b."""
    return SequenceMatcher(
        None, [clean(w) for w in a.split()], [clean(w) for w in b.split()], autojunk=False
    ).get_opcodes()
