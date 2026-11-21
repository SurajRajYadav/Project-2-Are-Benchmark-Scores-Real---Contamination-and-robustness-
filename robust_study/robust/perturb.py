"""Answer-preserving perturbations.

MMLU: reorder the options.  perm[i] = index (in the ORIGINAL list) of the option shown at position i.
GSM8K: rename people, or insert an irrelevant sentence. The correct answer does not change.
"""
import random
import re

LETTERS = ["A", "B", "C", "D"]


#  MMLU
def make_perm(cond, item_id, seed=0):
    """Conditions: 'orig', 'shiftK' (cyclic shift by K), 'randK' / 'sigK' (seeded random shuffle)."""
    if cond == "orig":
        return [0, 1, 2, 3]
    if cond.startswith("shift"):
        s = int(cond[len("shift"):])
        return [(i + s) % 4 for i in range(4)]
    if cond.startswith("rand") or cond.startswith("sig"):
        rng = random.Random(f"{seed}-{item_id}-{cond}")
        while True:
            p = [0, 1, 2, 3]
            rng.shuffle(p)
            if p != [0, 1, 2, 3]:
                return p
    raise ValueError(cond)


def apply_perm(choices, answer, perm):
    """Returns (reordered choices, new index of the correct answer)."""
    return [choices[j] for j in perm], perm.index(answer)


def mmlu_body(question, choices):
    return question.strip() + "\n" + "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, choices))


def mmlu_prompt(subject, question, choices):
    s = subject.replace("_", " ")
    return (f"The following are multiple choice questions (with answers) about {s}.\n\n"
            f"{mmlu_body(question, choices)}\nAnswer:")


#  GSM8K
NAMES = ["Janet", "Natalia", "Weng", "Betty", "Julie", "James", "Albert", "Ken", "Mark", "John",
         "Sarah", "Tom", "Mary", "Emily", "Jack", "Lisa", "Anna", "David", "Sam", "Peter", "Alice",
         "Bob", "Tim", "Jane", "Carla", "Jenna", "Wendy", "Mike", "Lucy", "Kelly", "Tina", "Ryan",
         "Linda", "Paul", "Susan", "Ben", "Emma", "Nina", "Oscar", "Maria", "Joe", "Kate", "Dan",
         "Rachel", "Steve", "Laura", "Eric", "Amy", "Chris", "Megan"]
REPLACEMENTS = ["Priya", "Kofi", "Mei", "Lars", "Aiko", "Mateo", "Fatima", "Ivan", "Zara", "Hugo",
                "Noor", "Tariq", "Sofia", "Dmitri", "Amara", "Kenji", "Leila", "Pablo", "Ingrid",
                "Rafael", "Yuki", "Omar", "Elena", "Tomas", "Nadia", "Felix", "Sana", "Anders"]
_NAME_RE = re.compile(r"\b(" + "|".join(NAMES) + r")\b")

DISTRACTORS = [
    "The town library was founded 14 years ago and now holds 3,200 books.",
    "It was a sunny day, and the temperature outside was 27 degrees.",
    "A nearby bakery sells 85 loaves of bread every morning.",
    "The school across the street has 640 students and 32 teachers.",
    "Last year the city planted 1,500 trees along the main road.",
    "A local museum opens at 9 o'clock and closes at 5 o'clock.",
    "The park's walking trail is 12 kilometers long.",
    "A train passes the station every 45 minutes.",
]


def rename_people(question, rng):
    """Consistently swap first names for different ones. Returns (text, changed)."""
    found = []
    for m in _NAME_RE.findall(question):
        if m not in found:
            found.append(m)
    if not found:
        return question, False
    pool = [r for r in REPLACEMENTS if r not in question]
    rng.shuffle(pool)
    mapping = {old: pool[i] for i, old in enumerate(found)}
    return _NAME_RE.sub(lambda m: mapping[m.group(1)], question), True


def add_distractor(question, rng):
    """Insert one irrelevant sentence before the final sentence (the actual question)."""
    sentence = rng.choice(DISTRACTORS)
    parts = re.split(r"(?<=[.?!])\s+", question.strip())
    if len(parts) == 1:
        return sentence + " " + question, True
    return " ".join(parts[:-1] + [sentence, parts[-1]]), True


def gsm_variant(cond, item_id, question, seed=0):
    rng = random.Random(f"{seed}-{item_id}-{cond}")
    if cond == "orig":
        return question, False
    if cond == "rename":
        return rename_people(question, rng)
    if cond == "distractor":
        return add_distractor(question, rng)
    raise ValueError(cond)


def gsm_prompt_text(question):
    return ("Solve the math problem. Think step by step, then write the final answer on the last "
            "line in the form 'Answer: <number>'.\n\nProblem: " + question)
