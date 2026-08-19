"""Build the text->regex spec pilot and self-verify every gold answer.

A spec is the unit of ground truth: an instruction, a small VISIBLE set of
examples the model is shown, and a larger HIDDEN set it is scored on. The
split is the point -- a pattern that reproduces the visible examples without
honouring the instruction fails the hidden set, which gives us a directly
measurable overfitting signal.

Nothing ships unverified: build_all() runs every gold answer through the
verifier and refuses to write a spec whose own gold fails.
"""

from __future__ import annotations

import json
from pathlib import Path

from verifier import verify

OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# --------------------------------------------------------------------------
# Single-turn specs
#
# `traps` names the specific mistakes the hidden set is built to catch. It is
# the difficulty dial: tier is derived from trap count, so the training mix can
# be stratified without hand-labelling difficulty.
# --------------------------------------------------------------------------

VALIDATE = [
    {
        "concept": "semver",
        "traps": ["leading-zero rejection", "component count", "optional suffix"],
        "instruction": (
            "Match a semantic version number: exactly three dot-separated numeric "
            "components, optionally followed by a pre-release suffix starting with a "
            "hyphen. No numeric component may have a leading zero (a component that is "
            "just '0' is fine). The whole string must be the version."
        ),
        "visible": {
            "positives": ["1.2.3", "0.0.1", "10.20.30", "1.0.0-beta.1"],
            "negatives": ["1.2", "v1.2.3", "1.2.3.4", "hello"],
        },
        "hidden": {
            "positives": ["2.3.4-rc1", "0.1.0", "99.0.12", "1.0.0-alpha.beta.2", "7.7.7"],
            "negatives": ["01.2.3", "1.2.03", "1.02.3", "1.2.3-", "", "1.2.3 ", " 1.2.3"],
        },
        "gold": {"pattern": r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]*[0-9A-Za-z])?", "flags": ""},
    },
    {
        "concept": "ipv4",
        "traps": ["octet range 0-255", "leading zeros", "escaping the dot"],
        "instruction": (
            "Match a dotted IPv4 address: four decimal octets separated by dots, each "
            "in the range 0-255, with no leading zeros, and nothing else in the string."
        ),
        "visible": {
            "positives": ["0.0.0.0", "192.168.1.1", "255.255.255.255"],
            "negatives": ["1.2.3", "256.1.1.1", "1.2.3.4.5"],
        },
        "hidden": {
            "positives": ["8.8.8.8", "10.0.0.254", "172.16.254.1", "127.0.0.1"],
            "negatives": ["192.168.01.1", "999.1.1.1", "1.2.3.", ".1.2.3", "1.2.3.4a", "1.2.3.-1"],
        },
        "gold": {"pattern": r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)){3}", "flags": ""},
    },
    {
        "concept": "hex_color",
        "traps": ["two valid lengths", "hex alphabet only"],
        "instruction": (
            "Match a CSS hex colour: a '#' followed by either exactly three or exactly "
            "six hexadecimal digits. Upper or lower case letters are both allowed."
        ),
        "visible": {
            "positives": ["#fff", "#FFFFFF", "#a1b2c3"],
            "negatives": ["#ffff", "fff", "#gg0000"],
        },
        "hidden": {
            "positives": ["#000", "#AbCdEf", "#123", "#0f0F0f"],
            "negatives": ["#12345", "#1234567", "#", "##fff", "#ff f", "0xfff"],
        },
        "gold": {"pattern": r"#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})", "flags": ""},
    },
    {
        "concept": "time24",
        "traps": ["hour range 00-23", "minute range 00-59", "zero padding required"],
        "instruction": (
            "Match a 24-hour time in HH:MM form. Hours run 00-23 and minutes 00-59. "
            "Both parts must be zero-padded to two digits."
        ),
        "visible": {
            "positives": ["00:00", "23:59", "09:30"],
            "negatives": ["24:00", "12:60", "9:30"],
        },
        "hidden": {
            "positives": ["01:05", "19:45", "12:00", "20:59"],
            "negatives": ["23:5", "1:05", "25:00", "12:99", "0000", "12:0"],
        },
        "gold": {"pattern": r"(?:[01]\d|2[0-3]):[0-5]\d", "flags": ""},
    },
    {
        "concept": "identifier",
        "traps": ["first-character class differs from the rest"],
        "instruction": (
            "Match a Python identifier: it must start with a letter or underscore, "
            "and may then contain letters, digits or underscores."
        ),
        "visible": {
            "positives": ["foo", "_bar", "a1"],
            "negatives": ["1foo", "foo-bar", "foo bar"],
        },
        "hidden": {
            "positives": ["__init__", "x", "_", "camelCase9"],
            "negatives": ["9lives", "", "foo.bar", "foo!", "-x"],
        },
        "gold": {"pattern": r"[A-Za-z_][A-Za-z0-9_]*", "flags": ""},
    },
    {
        "concept": "iso_date",
        "traps": ["month range 01-12", "day range 01-31", "fixed-width fields"],
        "instruction": (
            "Match an ISO date in YYYY-MM-DD form. The year is four digits, the month "
            "is 01-12, and the day is 01-31. All fields are zero-padded."
        ),
        "visible": {
            "positives": ["2024-01-15", "1999-12-31"],
            "negatives": ["2024-13-01", "2024-1-15"],
        },
        "hidden": {
            "positives": ["2000-02-29", "2026-08-19", "1900-01-01", "2024-10-05"],
            "negatives": ["2024-00-10", "2024-01-32", "24-01-15", "2024-01-1", "2024/01/15"],
        },
        "gold": {"pattern": r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])", "flags": ""},
    },
    {
        "concept": "uk_postcode",
        "traps": ["variable-length outward code", "case sensitivity", "mandatory space"],
        "instruction": (
            "Match a UK postcode in uppercase: one or two letters, a digit, an optional "
            "extra letter or digit, then a space, then a digit and two letters."
        ),
        "visible": {
            "positives": ["SW1A 1AA", "M1 1AE", "B33 8TH"],
            "negatives": ["sw1a 1aa", "SW1A1AA", "12345"],
        },
        "hidden": {
            "positives": ["CR2 6XH", "DN55 1PT", "EC1A 1BB", "W1A 0AX"],
            "negatives": ["SW1A  1AA", "S1A 1AA1", "1W1 1AA", "SW1A 1A", ""],
        },
        "gold": {"pattern": r"[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}", "flags": ""},
    },
]

EXTRACT = [
    {
        "concept": "first_quoted",
        "traps": ["greedy vs lazy"],
        "instruction": (
            "Capture the contents of the FIRST double-quoted string in the text, "
            "without the surrounding quotes. Quotes are never escaped."
        ),
        "visible": [
            ['he said "hello" then "goodbye"', "hello"],
            ['log: "started" ok', "started"],
            ["no quotes at all", None],
        ],
        "hidden": [
            ['a "" b "x"', ""],
            ['"first" "second" "third"', "first"],
            ['prefix "with spaces and, punctuation!" suffix', "with spaces and, punctuation!"],
            ['mismatched " unterminated', None],
            ['"a"+"b"', "a"],
        ],
        "gold": {"pattern": r'"([^"]*)"', "flags": ""},
    },
    {
        "concept": "email_domain",
        "traps": ["stop at the domain boundary", "dot in character class"],
        "instruction": (
            "Capture the domain part of the first email address in the text -- "
            "everything after the '@' up to but not including any trailing whitespace "
            "or punctuation that is not part of the domain."
        ),
        "visible": [
            ["contact bob@example.com now", "example.com"],
            ["ana@sub.domain.co.uk wrote", "sub.domain.co.uk"],
        ],
        "hidden": [
            ["x@a.io", "a.io"],
            ["first@one.com and second@two.com", "one.com"],
            ["no address here", None],
            ["user@mail.example.org.", "mail.example.org"],
        ],
        "gold": {"pattern": r"@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})", "flags": ""},
    },
    {
        "concept": "http_status",
        "traps": ["anchor to the right field", "avoid matching byte counts"],
        "instruction": (
            'Capture the three-digit HTTP status code from an access log line. It '
            'appears immediately after the quoted request, e.g. \'"GET / HTTP/1.1" 404 512\' '
            "-> 404. Do not capture the byte count that follows it."
        ),
        "visible": [
            ['127.0.0.1 - "GET / HTTP/1.1" 404 512', "404"],
            ['10.0.0.1 - "POST /a HTTP/1.0" 200 1234', "200"],
        ],
        "hidden": [
            ['1.2.3.4 - "GET /x HTTP/2.0" 301 99', "301"],
            ['1.2.3.4 - "GET /500 HTTP/1.1" 200 500', "200"],
            ['1.2.3.4 - "DELETE /y HTTP/1.1" 503 0', "503"],
            ["malformed line without request", None],
        ],
        "gold": {"pattern": r'HTTP/\d\.\d"\s+(\d{3})', "flags": ""},
    },
    {
        "concept": "file_extension",
        "traps": ["last dot not first", "end anchoring"],
        "instruction": (
            "Capture the file extension of a filename -- the characters after the FINAL "
            "dot, with no dot included. There is no trailing whitespace."
        ),
        "visible": [
            ["report.pdf", "pdf"],
            ["archive.tar.gz", "gz"],
            ["noextension", None],
        ],
        "hidden": [
            ["a.b.c.d.txt", "txt"],
            ["photo.JPEG", "JPEG"],
            [".hidden", "hidden"],
            ["path/to/file.py", "py"],
        ],
        "gold": {"pattern": r"\.([^.]+)$", "flags": ""},
    },
    {
        "concept": "first_paren",
        "traps": ["escaping parentheses", "greedy vs lazy"],
        "instruction": (
            "Capture the contents of the FIRST parenthesised group in the text, without "
            "the parentheses. Groups are never nested."
        ),
        "visible": [
            ["total (net) and (gross)", "net"],
            ["fn(x) = y", "x"],
            ["no parens", None],
        ],
        "hidden": [
            ["()", ""],
            ["a (one) b (two) c (three)", "one"],
            ["(with spaces, and punctuation!)", "with spaces, and punctuation!"],
            ["unclosed (group", None],
        ],
        "gold": {"pattern": r"\(([^)]*)\)", "flags": ""},
    },
    {
        "concept": "year_from_date",
        "traps": ["capture a sub-field of a larger match"],
        "instruction": (
            "Capture just the four-digit year from the first ISO date (YYYY-MM-DD) "
            "appearing in the text."
        ),
        "visible": [
            ["due 2024-01-15 sharp", "2024"],
            ["2019-12-31 was the deadline", "2019"],
        ],
        "hidden": [
            ["from 2020-01-01 to 2021-01-01", "2020"],
            ["ref 12345 on 1999-06-30", "1999"],
            ["no date here 2024", None],
            ["2026-08-19", "2026"],
        ],
        "gold": {"pattern": r"(\d{4})-\d{2}-\d{2}", "flags": ""},
    },
]

SUBSTITUTE = [
    {
        "concept": "comma_decimal",
        "traps": ["capture groups required for rebuild"],
        "instruction": (
            "Replace the decimal comma with a dot in every number that uses a comma as "
            "its decimal point. Leave the digits themselves unchanged."
        ),
        "visible": [
            ["Prices: 1,50 and 2,75 EUR", "Prices: 1.50 and 2.75 EUR"],
        ],
        "hidden": [
            ["weight 10,5 kg", "weight 10.5 kg"],
            ["no decimals here 42", "no decimals here 42"],
            ["3,141592 approx", "3.141592 approx"],
            ["a 0,99 b 12,00 c", "a 0.99 b 12.00 c"],
        ],
        "gold": {"pattern": r"(\d+),(\d+)", "replacement": r"\1.\2", "flags": ""},
    },
    {
        "concept": "collapse_whitespace",
        "traps": ["quantifier on whitespace class"],
        "instruction": (
            "Collapse every run of one or more whitespace characters into a single "
            "space character."
        ),
        "visible": [
            ["a   b", "a b"],
            ["too    many     spaces", "too many spaces"],
        ],
        "hidden": [
            ["a\tb", "a b"],
            ["line\nbreak", "line break"],
            ["  leading and trailing  ", " leading and trailing "],
            ["single space", "single space"],
        ],
        "gold": {"pattern": r"\s+", "replacement": " ", "flags": ""},
    },
    {
        "concept": "strip_trailing_ws",
        "traps": ["MULTILINE flag required", "end-of-line anchoring"],
        "instruction": (
            "Remove trailing spaces and tabs at the end of every line, without touching "
            "the line breaks themselves."
        ),
        "visible": [
            ["a   \nb\n", "a\nb\n"],
            ["x\t\ny  ", "x\ny"],
        ],
        "hidden": [
            ["one \ntwo  \nthree\t", "one\ntwo\nthree"],
            ["clean\nlines\n", "clean\nlines\n"],
            ["  keep leading  \n", "  keep leading\n"],
        ],
        "gold": {"pattern": r"[ \t]+$", "replacement": "", "flags": "m"},
    },
    {
        "concept": "mask_card",
        "traps": ["lookahead instead of consuming", "preserve the tail"],
        "instruction": (
            "Mask a run of digits by replacing every digit that has at least four more "
            "digits after it with a '*', leaving the final four digits visible."
        ),
        "visible": [
            ["4111111111111234", "************1234"],
            ["card 1234567890 end", "card ******7890 end"],
        ],
        "hidden": [
            ["12345", "*2345"],
            ["1234", "1234"],
            ["999", "999"],
            ["pin 000011112222 ok", "pin ********2222 ok"],
        ],
        "gold": {"pattern": r"\d(?=\d{4})", "replacement": "*", "flags": ""},
    },
    {
        "concept": "swap_name",
        "traps": ["group reordering in the replacement"],
        "instruction": (
            "Rewrite every 'Firstname Lastname' pair (two capitalised words separated by "
            "a single space) as 'Lastname, Firstname'."
        ),
        "visible": [
            ["John Smith", "Smith, John"],
            ["Ada Lovelace and Alan Turing", "Lovelace, Ada and Turing, Alan"],
        ],
        "hidden": [
            ["Grace Hopper", "Hopper, Grace"],
            ["all lowercase words", "all lowercase words"],
            ["Marie Curie, Niels Bohr", "Curie, Marie, Bohr, Niels"],
        ],
        "gold": {"pattern": r"([A-Z][a-z]+) ([A-Z][a-z]+)", "replacement": r"\2, \1", "flags": ""},
    },
]

# --------------------------------------------------------------------------
# Multi-turn trajectories
#
# Each turn is independently executable, so we can report per-turn pass rates
# rather than only scoring the final turn. Turn N's answer must be *derivable*
# from turn N-1's visible answer, because the Thinking model's chat template
# strips <think> blocks from history -- the reasoning is gone by the next turn,
# only the pattern survives.
# --------------------------------------------------------------------------

TRAJECTORIES = [
    {
        "concept": "comma_decimal_chain",
        "turns": [
            {
                "family": "extract",
                "traps": ["locate before transform"],
                "instruction": "Capture the first number that uses a comma as its decimal point.",
                "visible": [["Prices: 1,50 and 2,75 EUR", "1,50"]],
                "hidden": [
                    ["weight 10,5 kg", "10,5"],
                    ["no decimals here 42", None],
                    ["a 0,99 b 12,00 c", "0,99"],
                ],
                "gold": {"pattern": r"(\d+,\d+)", "flags": ""},
            },
            {
                "family": "substitute",
                "traps": ["restructure into capture groups"],
                "instruction": "Ok great, rewrite that to replace the commas with dots.",
                "visible": [["Prices: 1,50 and 2,75 EUR", "Prices: 1.50 and 2.75 EUR"]],
                "hidden": [
                    ["weight 10,5 kg", "weight 10.5 kg"],
                    ["3,141592 approx", "3.141592 approx"],
                    ["a 0,99 b 12,00 c", "a 0.99 b 12.00 c"],
                ],
                "gold": {"pattern": r"(\d+),(\d+)", "replacement": r"\1.\2", "flags": ""},
            },
            {
                "family": "substitute",
                "traps": ["negative lookahead", "narrow a previous answer"],
                "instruction": (
                    "Careful -- some of those are thousands separators. Only convert when "
                    "there are exactly one or two digits after the comma."
                ),
                "visible": [["Total 1,234,567 items at 2,75 each", "Total 1,234,567 items at 2.75 each"]],
                "hidden": [
                    ["10,5 kg and 1,000,000 units", "10.5 kg and 1,000,000 units"],
                    ["3,141592 approx", "3,141592 approx"],
                    ["mixed 1,234 and 5,67", "mixed 1,234 and 5.67"],
                ],
                "gold": {"pattern": r"(\d+),(\d{1,2})(?!\d)", "replacement": r"\1.\2", "flags": ""},
            },
        ],
    },
    {
        "concept": "whitespace_chain",
        "turns": [
            {
                "family": "substitute",
                "traps": ["whitespace class"],
                "instruction": "Collapse every run of whitespace into a single space.",
                "visible": [["a   b", "a b"]],
                "hidden": [["x\t\ty", "x y"], ["one\ntwo", "one two"], ["p  q  r", "p q r"]],
                "gold": {"pattern": r"\s+", "replacement": " ", "flags": ""},
            },
            {
                "family": "substitute",
                "traps": ["exclude newline from the class"],
                "instruction": "That flattened my line breaks. Keep newlines intact and only collapse spaces and tabs.",
                "visible": [["a   b\nc   d", "a b\nc d"]],
                "hidden": [["x\t\ty\nz", "x y\nz"], ["one\n\ntwo", "one\n\ntwo"], ["p  q", "p q"]],
                "gold": {"pattern": r"[ \t]+", "replacement": " ", "flags": ""},
            },
            {
                "family": "substitute",
                "traps": ["MULTILINE flag", "end anchoring"],
                "instruction": "Now instead just remove spaces and tabs at the end of each line, leaving inner spacing alone.",
                "visible": [["a   b   \nc\n", "a   b\nc\n"]],
                "hidden": [["one  \ntwo\t\n", "one\ntwo\n"], ["  keep  \n", "  keep\n"], ["clean\n", "clean\n"]],
                "gold": {"pattern": r"[ \t]+$", "replacement": "", "flags": "m"},
            },
        ],
    },
    {
        "concept": "mask_chain",
        "turns": [
            {
                "family": "extract",
                "traps": ["fixed-width counting"],
                "instruction": "Capture the last four digits of a sixteen-digit card number.",
                "visible": [["4111111111111234", "1234"]],
                "hidden": [["0000111122223333", "3333"], ["9876543210987654", "7654"]],
                "gold": {"pattern": r"\d{12}(\d{4})", "flags": ""},
            },
            {
                "family": "substitute",
                "traps": ["lookahead instead of consuming"],
                "instruction": "Now mask everything except those last four digits.",
                "visible": [["4111111111111234", "************1234"]],
                "hidden": [["0000111122223333", "************3333"], ["12345", "*2345"], ["1234", "1234"]],
                "gold": {"pattern": r"\d(?=\d{4})", "replacement": "*", "flags": ""},
            },
            {
                "family": "substitute",
                "traps": ["lookbehind plus lookahead"],
                "instruction": "Actually keep the first four digits visible too, and mask only the middle.",
                "visible": [["4111111111111234", "4111********1234"]],
                "hidden": [["0000111122223333", "0000********3333"], ["123456789", "1234*6789"]],
                "gold": {"pattern": r"(?<=\d{4})\d(?=\d{4})", "replacement": "*", "flags": ""},
            },
        ],
    },
]


def tier(traps: list[str]) -> str:
    return {1: "easy", 2: "medium"}.get(len(traps), "hard")


def build_all() -> tuple[list[dict], list[dict], list[str]]:
    """Assemble specs and reject any whose own gold answer fails verification."""
    single, problems = [], []
    families = [("validate", VALIDATE), ("extract", EXTRACT), ("substitute", SUBSTITUTE)]

    for family, concepts in families:
        for i, c in enumerate(concepts):
            spec = {
                "id": f"rx_{family}_{c['concept']}_{i:03d}",
                "family": family,
                "concept": c["concept"],
                "traps": c["traps"],
                "difficulty": tier(c["traps"]),
                "instruction": c["instruction"],
                "visible": c["visible"],
                "hidden": c["hidden"],
                "gold": c["gold"],
            }
            for split in ("visible", "hidden"):
                r = verify(spec["gold"], spec, split)
                if not r.passed:
                    problems.append(f"{spec['id']} [{split}]: {r.error or r.failures}")
            single.append(spec)

    multi = []
    for t in TRAJECTORIES:
        turns = []
        for n, turn in enumerate(t["turns"], start=1):
            spec = {
                "id": f"rx_traj_{t['concept']}_t{n}",
                "family": turn["family"],
                "concept": t["concept"],
                "turn": n,
                "traps": turn["traps"],
                "difficulty": tier(turn["traps"]),
                "instruction": turn["instruction"],
                "visible": turn["visible"],
                "hidden": turn["hidden"],
                "gold": turn["gold"],
            }
            for split in ("visible", "hidden"):
                r = verify(spec["gold"], spec, split)
                if not r.passed:
                    problems.append(f"{spec['id']} [{split}]: {r.error or r.failures}")
            turns.append(spec)
        multi.append({"id": f"rx_traj_{t['concept']}", "concept": t["concept"], "turns": turns})

    return single, multi, problems


def main() -> int:
    single, multi, problems = build_all()

    print(f"single-turn specs : {len(single)}")
    print(f"trajectories      : {len(multi)} ({sum(len(t['turns']) for t in multi)} turns)")

    by_tier: dict[str, int] = {}
    for s in single:
        by_tier[s["difficulty"]] = by_tier.get(s["difficulty"], 0) + 1
    print(f"difficulty mix    : {by_tier}")

    if problems:
        print(f"\nGOLD VERIFICATION FAILURES ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "specs_single.jsonl", "w") as f:
        for s in single:
            f.write(json.dumps(s) + "\n")
    with open(OUT_DIR / "specs_multiturn.jsonl", "w") as f:
        for t in multi:
            f.write(json.dumps(t) + "\n")

    print("\nAll gold answers verified on both splits. Written to data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
