"""Held-out Gemma prompt variety study. Only the prompt is sent to the model."""

import ast
import json
import operator
import re


def item(category, prompt, max_tokens, rubric, form):
    return dict(category=category, prompt=prompt, max_tokens=max_tokens, rubric=rubric, format=form)


PROSE = "Plain prose, at most 50 words."
JSON_ONLY = "A single bare JSON value of the requested shape."
EXPRESSION = "One bare Python return expression. Omit the return keyword and Markdown."

# Interleave categories. Every block of four contains one prompt of each type.
WORKLOADS = {
    "mirror_delete": item("explanation",
        "A photographer saves files on two drives that mirror each other in real time. She accidentally deletes a photo. "
        "Why can the photo disappear from both drives, and what should she have kept to recover it? Use plain prose in at most 50 words.",
        96, "The deletion propagates through the mirror. An independent backup, older snapshot or retained version permits recovery.", PROSE),
    "generator_read": item("code_comprehension",
        "I'm reviewing a Python import script. What three numbers does this print, in order? Reply with just a JSON array.\n\n"
        "values = (int(s) for s in ['3', '8', '4'])\nfirst = next(values)\nprint([first, sum(values), sum(values)])",
        32, "The three results are 3, 12 and 0, since the iterator is exhausted.", JSON_ONLY),
    "gallery_label": item("document_extraction",
        "Exhibition catalog notes: the lidded spice box, accession 18, came from the Eastbank Collection. Its old card labels it pewter. "
        "A materials examination identified the box as tin and its lid as brass. The painted cup beside it belongs to the North Quay Trust. "
        'For the body of the spice box, return only a JSON object with the keys "material" and "lender".',
        48, "The box body is tin and its lender is the Eastbank Collection.", JSON_ONLY),
    "page_count": item("code_fix",
        "Our pagination helper drops a partly filled last page. Rows is a nonnegative integer and size is a positive integer. "
        "Replace the expression after return, using integer arithmetic; zero rows should need zero pages. Give only the Python expression, "
        "without the return keyword or Markdown.\n\ndef page_count(rows, size):\n    return rows // size",
        48, "Integer ceiling division, including zero and exact multiples. The returned value must be an integer.", EXPRESSION),
    "cold_glass": item("explanation",
        "On a humid day, droplets form on the outside of a sealed glass bottle filled with cold water. "
        "Where does the water on the outside come from, and why does it become liquid there? Use plain prose in at most 50 words.",
        96, "Water vapor in the surrounding air cools at the cold surface and condenses into liquid.", PROSE),
    "sql_count": item("code_comprehension",
        "A SQLite table named visits has five rows. In insertion order its zone column contains 'north', NULL, 'west', 'north', NULL. "
        "What three values does this query return? Give only a JSON array of three integers in column order.\n\n"
        "SELECT COUNT(*), COUNT(zone), COUNT(DISTINCT zone) FROM visits;",
        32, "COUNT(*) includes five rows, COUNT(zone) includes three non-NULL entries and DISTINCT includes two.", JSON_ONLY),
    "repair_invoice": item("document_extraction",
        "Service slip B-194 lists an estimate of CAD 62.00. The shop replaced a latch for CAD 38.00 and charged CAD 24.00 for labor. "
        "A warranty adjustment covered the entire latch. The customer paid CAD 10.00 at pickup; the receipt records CAD 14.00 still owing. "
        'Extract the remaining balance as a bare JSON object with keys "amount" (a number) and "currency" (a three-letter code).',
        40, "The remaining balance is 14 Canadian dollars.", JSON_ONLY),
    "csv_suffix": item("code_fix",
        "The upload filter below rejects REPORT.CSV and names with trailing spaces. Return a boolean for whether the name ends with "
        "the literal suffix .csv after ignoring letter case and trailing whitespace. Use built-in string operations. Reply with only "
        "the replacement Python expression, without return or Markdown.\n\ndef is_csv(name):\n    return name.endswith('.csv')",
        55, "Strip trailing whitespace and compare the case-insensitive literal suffix. Reject a missing dot or a later extension.", EXPRESSION),
    "clock_repeat": item("explanation",
        "A train log from one city contains two entries labeled 01:20 on the day the clocks go back. "
        "How could those be different instants, and what time information would let a reader tell them apart? Use plain prose in at most 50 words.",
        96, "The same local hour occurs twice when clocks roll back. A UTC timestamp, UTC offset or an explicit DST/standard-time marker identifies which.", PROSE),
    "array_copy": item("code_comprehension",
        "This JavaScript is preparing badge lists. What will the last line print? Return the JSON value only.\n\n"
        'const original = ["Elm", "Oak"];\nconst copy = original.slice();\noriginal.shift();\ncopy.push("Ash");\n'
        "console.log(JSON.stringify({ original, copy }));",
        64, "The original contains Oak; its shallow copy contains Elm, Oak, Ash.", JSON_ONLY),
    "trail_detour": item("document_extraction",
        "Park desk bulletin: cyclists approaching Orchard Lawn must follow Canal Walk. Walkers heading the other way, toward Mill Gate, "
        "use the north stairs. For pedestrians walking from Mill Gate to Orchard Lawn, start on Ridge Steps, then pass through Fern Arch "
        "before rejoining the main path. Birch Tunnel is closed in both directions. A family is walking from Mill Gate to Orchard Lawn. "
        "Return just a JSON array containing the two named detour landmarks in the order they reach them.",
        40, "This direction and mode of travel use Ridge Steps followed by Fern Arch.", JSON_ONLY),
    "overnight_hours": item("code_fix",
        "This hourly schedule helper fails on shifts that cross midnight. Inputs are integer hours 0 through 23; start and end differ. "
        "The shift includes start and excludes end. Fix the return expression for both daytime and overnight shifts with arithmetic, "
        "comparisons or boolean operators. Reply with only the Python expression, without return or Markdown.\n\n"
        "def on_shift(hour, start, end):\n    return start <= hour < end",
        72, "Half-open interval on a 24-hour clock. Test both boundaries and both sides of midnight. Return a boolean.", EXPRESSION),
}

EXPECTED = {
    "generator_read": [3, 12, 0], "sql_count": [5, 3, 2],
    "array_copy": {"original": ["Oak"], "copy": ["Elm", "Oak", "Ash"]},
    "gallery_label": {"material": "tin", "lender": "Eastbank Collection"},
    "repair_invoice": {"amount": 14, "currency": "CAD"},
    "trail_detour": ["Ridge Steps", "Fern Arch"],
}

# Each required group targets one part of the meaning. These are automatic proxy
# checks; a build-blind review can challenge their decisions without changing them.
CONCEPTS = {
    "mirror_delete": [
        ("deletion", r"delet|remov|eras"),
        ("propagates to mirror", r"propagat|replicat|mirror|synchron|sync\b|reflect|copied|copies|duplicat|both drives.{0,24}(same|match)"),
        ("recoverable previous copy", r"backup|back.up|snapshot|previous version|earlier version|older version|version histor|versioned|offline copy|separate copy|archiv"),
    ],
    "cold_glass": [
        ("surrounding air", r"\bair\b|atmospher|airborne|ambient humidity"),
        ("airborne water", r"vapor|vapour|humidity|moisture|humid air|gaseous water"),
        ("cooling", r"cool|cold|chill|dew point|loses? heat"),
        ("condensation", r"condens|liquefi|turns? (?:back )?into (?:a )?(?:liquid|droplets)|becomes? liquid"),
    ],
    "clock_repeat": [
        ("repeated local hour", r"repeat|twice|second time|duplicate|two (?:different )?(?:times|instants|occurrences)|same (?:local |clock )?(?:hour|time).{0,20}(again|appears)|occurs? again"),
        ("backward change or DST transition", r"back|fall|daylight|summer time|standard time|dst\b"),
        ("disambiguation", r"utc|gmt|offset|daylight.{0,24}standard|standard.{0,24}daylight|dst.{0,20}(status|flag|indicator)|timestamp in universal time|epoch|unix time"),
    ],
}

CONTRADICTIONS = {
    "mirror_delete": (r"\b(?:no backup is needed|backup is unnecessary|does not (?:replicate|mirror|propagate) deletions)\b",),
    "cold_glass": (r"(?:^|[.!?]\s+)(?:the )?water (?:leaks|seeps) (?:out )?through (?:the )?(?:glass|bottle)\b",),
    "clock_repeat": (r"\b(?:utc offsets? (?:cannot|can't) (?:distinguish|separate)|hour (?:does not|doesn't) repeat)\b",),
}

CASES = {
    "page_count": [({"rows": r, "size": s}, (r+s-1)//s) for r, s in
                   [(0, 5), (1, 5), (5, 5), (6, 5), (37, 8), (4, 1), (10**16+1, 4)]],
    "csv_suffix": [({"name": s}, ok) for s, ok in
                   [("report.csv", True), ("REPORT.CsV  ", True), ("a.csv\t", True),
                    ("csv", False), ("filecsv", False), ("file.csv.zip", False), ("", False), (".csv", True)]],
    "overnight_hours": [({"hour": h, "start": s, "end": e}, (h-s)%24 < (e-s)%24)
                        for s,e in [(9, 17), (22, 6), (23, 0), (0, 1)] for h in range(24)],
}

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
            ast.Eq: operator.eq, ast.NotEq: operator.ne}
_STRING_METHODS = {"strip": str.strip, "rstrip": str.rstrip, "lower": str.lower,
                   "casefold": str.casefold, "endswith": str.endswith}


def unformat(text):
    match = re.fullmatch(r"\s*```(?:python|py|json|javascript|js)?\s*\n(.*?)\n?\s*```\s*", text, re.I | re.S)
    return (match[1].strip(), True) if match else (text.strip(), False)


def calculate(node, env):
    """Interpret a small bounded expression subset on trusted scalar test inputs."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, bool, str):
        if isinstance(node.value, str) and len(node.value) > 256:
            raise ValueError("long literal")
        if type(node.value) is int and abs(node.value) > 10**18:
            raise ValueError("large literal")
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        return {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}[type(node.op)](calculate(node.operand, env))
    if isinstance(node, ast.BinOp):
        left, right = calculate(node.left, env), calculate(node.right, env)
        if type(left) is not int or type(right) is not int:
            raise ValueError("integer operands required")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.BoolOp):
        for child in node.values:
            result = calculate(child, env)
            if (isinstance(node.op, ast.And) and not result) or (isinstance(node.op, ast.Or) and result):
                break
        return result
    if isinstance(node, ast.Compare):
        left = calculate(node.left, env)
        for op, rhs in zip(node.ops, node.comparators):
            right = calculate(rhs, env)
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return calculate(node.body if calculate(node.test, env) else node.orelse, env)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not node.keywords:
        target = calculate(node.func.value, env)
        if type(target) is not str:
            raise ValueError("string methods only")
        return _STRING_METHODS[node.func.attr](target, *(calculate(arg, env) for arg in node.args))
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        target = calculate(node.value, env)
        if type(target) is not str:
            raise ValueError("only string slicing")
        sl = node.slice
        return target[slice(*(calculate(v, env) if v is not None else None for v in (sl.lower, sl.upper, sl.step)))]
    raise ValueError("unsupported expression")


def json_type(value, gold):
    if type(gold) in (int, float):
        return type(value) in (int, float)
    if type(value) is not type(gold):
        return False
    if isinstance(gold, dict):
        return value.keys() == gold.keys() and all(json_type(value[k], v) for k,v in gold.items())
    if isinstance(gold, list):
        return len(value) == len(gold) and all(json_type(x,y) for x,y in zip(value,gold))
    return True


def normalized_data(obj):
    if isinstance(obj, str):
        return " ".join(obj.casefold().split())
    if isinstance(obj, dict):
        return {k: normalized_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalized_data(v) for v in obj]
    return obj


def unique_fields(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON key")
    return obj


def grade(name, answer, token_count=None):
    spec = WORKLOADS[name]
    text, fenced = unformat(answer)
    category = spec["category"]
    if category == "explanation":
        lower = text.lower()
        missing = [label for label, pattern in CONCEPTS[name] if not re.search(pattern, lower)]
        contradicts = any(re.search(pattern, lower) for pattern in CONTRADICTIONS[name])
        semantic = not missing and not contradicts
        plain = bool(text) and not fenced and not re.search(r"(?m)^\s*(?:[-*#>]|\d+[.)]\s)|`{3}", answer)
        words = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", text))
        form = plain and words <= 50
        reason = "explicit contradiction" if contradicts else "missing " + ", ".join(missing) if missing else "concepts present"
        reason += f"; {words} words"
    elif name in EXPECTED:
        try:
            value = json.loads(text, object_pairs_hook=unique_fields)
            shape = json_type(value, EXPECTED[name])
            if name in ("gallery_label", "repair_invoice", "trail_detour") and shape:
                semantic = normalized_data(value) == normalized_data(EXPECTED[name])
            else:
                semantic = shape and value == EXPECTED[name]
            form = shape and not fenced
            if name == "sql_count" and shape:
                form = form and all(type(n) is int for n in value)
            reason = "expected data" if semantic else "wrong fields, values, or order"
        except (ValueError, TypeError, RecursionError):
            semantic = form = False
            reason = "JSON could not be parsed"
    else:
        # A harmless wrapper can be separated from the actual expression's behavior.
        returned = bool(re.match(r"^return\s+",text))
        source = re.sub(r"^return\s+","",text) if returned else text
        node = None
        try:
            node = ast.parse(source, mode="eval")
            if sum(1 for _ in ast.walk(node)) > 96:
                raise ValueError("expression too large")
            form = not fenced and not returned
            semantic = True
            for env, gold in CASES[name]:
                actual = calculate(node.body, env)
                if type(actual) is not type(gold) or actual != gold:
                    semantic = False
                    break
            reason = f"passed {len(CASES[name])} cases" if semantic else "functional case failed"
        except (SyntaxError, ValueError, KeyError, TypeError, ArithmeticError, RecursionError):
            semantic = False
            form = not fenced and not returned and node is not None
            reason = "invalid or unsupported expression"
    return {"semantic_correct": bool(semantic), "instruction_pass": bool(form),
            "strict_correct": bool(semantic and form), "reason": reason}


def self_check():
    good = {
        "mirror_delete": "The mirror replicates deletions to both drives. A backup or older snapshot would preserve the photo.",
        "cold_glass": "Water vapor in the surrounding air cools at the bottle's cold surface and condenses into droplets.",
        "clock_repeat": "The local hour repeats as clocks go back. Adding the UTC offset tells the two instants apart.",
        "generator_read": "[3, 12, 0]", "sql_count": "[5, 3, 2]", "array_copy": '{"original":["Oak"],"copy":["Elm","Oak","Ash"]}',
        "gallery_label": '{"material":"tin","lender":"Eastbank Collection"}', "repair_invoice": '{"amount":14.0,"currency":"CAD"}',
        "trail_detour": '["Ridge Steps","Fern Arch"]', "page_count": "(rows + size - 1) // size",
        "csv_suffix": 'name.rstrip().lower().endswith(".csv")',
        "overnight_hours": "start <= hour < end if start < end else hour >= start or hour < end",
    }
    wrong = {
        "mirror_delete": "The photo disappears because it was deleted.", "cold_glass":"Water leaks through the bottle.",
        "clock_repeat":"They are simultaneous. Store only local hours.", "generator_read":"[3,12,15]",
        "sql_count":"[5,5,3]", "array_copy":'{"original":["Oak"],"copy":["Oak","Ash"]}',
        "gallery_label":'{"material":"brass","lender":"North Quay Trust"}', "repair_invoice":'{"amount":24,"currency":"CAD"}',
        "trail_detour":'["Fern Arch","Ridge Steps"]', "page_count":"rows // size", "csv_suffix":'name.endswith("csv")',
        "overnight_hours":"start <= hour <= end",
    }
    assert len(WORKLOADS)==len(good)==len(wrong)==12
    for name in WORKLOADS:
        assert grade(name, good[name])["strict_correct"], name
        assert not grade(name, wrong[name])["semantic_correct"], name
        fenced = grade(name,"```\n"+good[name]+"\n```")
        assert fenced["semantic_correct"] and not fenced["instruction_pass"], name
    assert not grade("csv_suffix","__import__('os').system('echo forbidden')")["semantic_correct"]
    assert grade("overnight_hours","(hour - start) % 24 < (end - start) % 24")["strict_correct"]
    assert not grade("repair_invoice",'{"amount":true,"currency":"CAD"}')["semantic_correct"]
    assert not grade("gallery_label",'{"material":"brass","material":"tin","lender":"Eastbank Collection"}')["semantic_correct"]
    assert not grade("mirror_delete","The mirror copies deletions to the other drive. No backup is needed.")["semantic_correct"]
    assert not grade("sql_count","[5.0,3.0,2.0]")["strict_correct"]


if __name__ == "__main__":
    self_check()
    print("12 prompt graders passed reference, negative, and strict-format controls.")
