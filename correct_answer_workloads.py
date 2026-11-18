"""Frozen evaluation prompts and correctness checks for the paired Intel benchmark."""

import ast
import json
import operator
import re

from compare_avx2 import GRAMMAR, STREAM_EXPECTED, STREAM_PROMPT


RELEASE_HANDOFF = """Use the following handover notes to prepare a short release record for Atlas. The notes were written over several days, so preliminary plans may have been superseded.

Monday, planning call. Atlas and Beacon are separate deployments. Engineering proposed starting Atlas at 14:00 UTC. The chair said that time was provisional until the support team signed off. The draft listed Noah Reed as coordinator while the usual coordinator was away. The client announcement did not specify a time or an owner.

Tuesday, rehearsal. For Atlas the team restored a sandbox from a snapshot and tested the old fallback checklist, RB-201. The rehearsal used dummy accounts. It did not approve a production rollback plan or change the start window. Beacon completed its own rehearsal later that day. Its production fallback uses RB-690 and its coordinator is Eli Tan.

Wednesday, staffing. Noah handed production coordination for Atlas over to Mira Chen. She accepted ownership for this release. Noah will handle customer messages during the change. The test lead remains responsible only for the sandbox. The handover does not apply to Beacon or its release owner.

Thursday, change board. Support approved 17:40 UTC as the production start for Atlas, replacing the tentative Monday time. The board accepted RB-417 as its production rollback ticket. The old rehearsal ticket stays open for cleaning up test accounts. Beacon still starts at 18:30 UTC. Neither team should copy the other project's details into its release record.

Friday, operations inbox. The pending customer message is ready to send once deployment starts. A final checkpoint ten minutes before the start will confirm everyone is online. The checkpoint changes neither the approved time nor the rollback ticket.

Return a JSON object for the final Atlas production release. Use exactly the keys "owner", "utc" (24-hour HH:MM) and "rollback" (ticket ID)."""

WORKLOADS = {
    "cold": {
        "prompt": "An eight-slot worker pool has three slots occupied. How many slots are free? Reply with just the integer.",
        "max_tokens": 12, "description": "First request after a fresh server launch; exact integer answer.",
    },
    "explanation": {
        "prompt": ("In Python, cache = {}. A teammate thinks cache.get('jobs', 7) adds an entry, "
                   "so reading cache['jobs'] on the next line will work. In two short sentences, explain "
                   "what the get call returns, whether it changes the dictionary, and what happens "
                   "on the next line. Use your own words."),
        "max_tokens": 80, "description": "Explain the fallback value, missing mutation, and subsequent exception.",
    },
    "long_prompt": {
        "prompt": RELEASE_HANDOFF, "max_tokens": 64,
        "description": "Resolve three production fields from a multi-day release handover.",
    },
    "code_edit": {
        "prompt": ("Fix the Python function below. It subtracts the free allowance correctly, but "
                   "sometimes bills negative minutes. Return zero in that case. Keep the existing "
                   "function name, parameter names, and default, including support for keyword calls. "
                   "Use simple integer arithmetic, comparisons, if/else, or built-in min/max. "
                   "Return only the complete function.\n\ndef billable_minutes(minutes, allowance=5):\n"
                   "    return minutes - allowance"),
        "max_tokens": 80, "description": "Fix the lower bound and preserve the function's callable interface.",
    },
    "fixed_decode": {
        "prompt": STREAM_PROMPT, "grammar": GRAMMAR, "max_tokens": 144,
        "description": "Original grammar-limited 144-token repeated-sentence decode control.",
    },
}


def _strip_fence(text):
    match = re.fullmatch(r"\s*```(?:python|json)?\s*\n(.*?)\n?\s*```\s*", text, re.I | re.S)
    return match[1].strip() if match else text.strip()


def explanation_checks(text):
    """Three positive concepts plus an explicit contradiction check; publish prose for audit."""
    words = text.lower().replace("’", "'")
    words = re.sub(r"\s+", " ", words.replace("`", " "))
    tests = {
        "fallback of seven": bool(re.search(
            r"\b(?:return\w*|yield\w*|give\w*|value|default|fallback)\b[^.!?;]{0,40}\b(?:7|seven)\b"
            r"|\b(?:7|seven)\b[^.!?;]{0,25}\b(?:returned|fallback|default)\b", words)),
        "entry stays absent": bool(re.search(
            r"\b(?:without|never|no|not|doesn't|won't|isn't|wasn't|cannot|can't|doesnt|wont)"
            r"\b.{0,28}?\b(?:add\w*|insert\w*|stor\w*|set\w*|creat\w*|chang\w*|modif\w*|"
            r"mutat\w*|updat\w*|present|persist\w*|exist\w*|put)\b"
            r"|\b(?:remains?|stays?|still|leaves?|keeps?)\b.{0,34}?\b(?:empty|absent|missing|unchanged|unmodified)\b"
            r"|\bunchanged\b|\bnot in (?:the |your )?(?:cache|dict\w*)\b", words)),
        "KeyError": bool(re.search(r"\bkey\s*error\b", words)),
    }
    if re.search(r"(?:return\w*|yield\w*|give\w*)\s+(?:none|null|zero|0)\b|\bnot (?:7|seven)\b", words):
        tests["fallback of seven"] = False
    negated_error = re.search(r"(?:doesn't|does not|won't|will not|never) "
                              r"(?:raise|throw|cause|produce) (?:an? )?key\s*error", words)
    if negated_error and len(re.findall(r"\bkey\s*error\b", words)) == 1:
        tests["KeyError"] = False
    return tests


_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub}
_RELATION = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
             ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def _value(node, variables):
    """Interpret a small numerical subset of Python. Generated Python is never executed."""
    if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10000:
        return node.value
    if isinstance(node, ast.Name):
        return variables[node.id]
    if isinstance(node, ast.BinOp):
        return _ARITHMETIC[type(node.op)](_value(node.left, variables), _value(node.right, variables))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_value(node.operand, variables)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return _RELATION[type(node.ops[0])](_value(node.left, variables), _value(node.comparators[0], variables))
    if isinstance(node, ast.IfExp):
        return _value(node.body if _value(node.test, variables) else node.orelse, variables)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 2 and not node.keywords:
        return {"max": max, "min": min}[node.func.id](*(_value(arg, variables) for arg in node.args))
    raise ValueError("unsupported expression")


def _body(statements, variables):
    for stmt in statements:
        if isinstance(stmt, ast.Return):
            return _value(stmt.value, variables)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            variables[stmt.targets[0].id] = _value(stmt.value, variables)
        elif isinstance(stmt, ast.If):
            result = _body(stmt.body if _value(stmt.test, variables) else stmt.orelse, variables)
            if result is not None:
                return result
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        else:
            raise ValueError("unsupported statement")
    return None


def _code_grade(text):
    try:
        module = ast.parse(_strip_fence(text))
        if len(list(ast.walk(module))) > 120 or len(module.body) != 1:
            return False, "expected one compact function"
        fn = module.body[0]
        if not isinstance(fn, ast.FunctionDef) or fn.name != "billable_minutes" or fn.decorator_list:
            return False, "incorrect function name or unsupported decorator"
        args = fn.args
        if ([p.arg for p in args.args] != ["minutes", "allowance"] or args.posonlyargs or args.kwonlyargs
                or args.vararg or args.kwarg or len(args.defaults) != 1
                or _value(args.defaults[0], {}) != 5):
            return False, "signature or default changed"
        cases = [(0, 5, 0), (3, 5, 0), (5, 5, 0), (12, 5, 7),
                 (2, 0, 2), (11, 7, 4), (4, 9, 0)]
        for minutes, allowance, expected in cases:
            value = _body(fn.body, {"minutes": minutes, "allowance": allowance})
            if type(value) is not int or value != expected:
                return False, f"failed inputs minutes={minutes}, allowance={allowance}"
    except (SyntaxError, ValueError, KeyError, TypeError, RecursionError):
        return False, "invalid or unsupported Python in bounded interpreter"
    return True, "signature and all seven functional cases passed"


def grade(name, answer, token_count):
    if name == "cold":
        passed = _strip_fence(answer).strip().rstrip(".") == "5"
        return passed, "expected integer 5"
    if name == "explanation":
        checks = explanation_checks(answer)
        missing = [key for key, ok in checks.items() if not ok]
        return not missing, "missing " + ", ".join(missing) if missing else "all three concepts detected"
    if name == "long_prompt":
        try:
            record = json.loads(_strip_fence(answer))
        except (TypeError, ValueError):
            return False, "invalid JSON"
        if not isinstance(record, dict) or set(record) != {"owner", "utc", "rollback"}:
            return False, "expected owner, utc and rollback fields"
        passed = (str(record["owner"]).strip().casefold() == "mira chen"
                  and str(record["utc"]).strip() == "17:40"
                  and str(record["rollback"]).strip().upper() == "RB-417")
        return passed, "compared all three fields with the approved Atlas production record"
    if name == "code_edit":
        return _code_grade(answer)
    if name == "fixed_decode":
        passed = token_count == WORKLOADS[name]["max_tokens"] and bool(answer) and STREAM_EXPECTED.startswith(answer)
        return passed, "required full budget and exact repeated-sentence prefix"
    raise KeyError(name)


def self_check():
    assert grade("cold", "5", 1)[0] and not grade("cold", "8", 1)[0]
    sentences = [
        "It returns the default value 7 without inserting the key. The next line raises a KeyError.",
        "Seven is the fallback; the dictionary stays empty. Accessing jobs then gives a key error.",
    ]
    assert all(grade("explanation", text, 30)[0] for text in sentences)
    assert not grade("explanation", "7 is returned and stored, so access does not raise a KeyError.", 25)[0]
    assert not grade("explanation", "It returns None, not 7, without adding the key. The read raises KeyError.", 25)[0]
    assert not grade("explanation", "Returns 7; the next read raises KeyError.", 12)[0]
    record = {"utc": "17:40", "owner": "Mira Chen", "rollback": "RB-417"}
    assert grade("long_prompt", json.dumps(record), 28)[0]
    assert not grade("long_prompt", json.dumps(dict(record, rollback="RB-201")), 28)[0]
    header = "def billable_minutes(minutes, allowance=5):\n"
    assert grade("code_edit", header + "    return max(0, minutes - allowance)", 24)[0]
    assert grade("code_edit", header + "    if minutes < allowance:\n        return 0\n    return minutes - allowance", 32)[0]
    assert not grade("code_edit", header + "    return minutes - allowance", 20)[0]
    assert not grade("code_edit", header + "    return max(0, minutes - 5)", 20)[0]
    assert not grade("code_edit", header + "    return open('bad')", 15)[0]
    assert grade("fixed_decode", STREAM_EXPECTED[:517], 144)[0]
    assert not grade("fixed_decode", STREAM_EXPECTED[:517], 143)[0]


if __name__ == "__main__":
    self_check()
    print("All workload grader reference and negative controls passed.")
