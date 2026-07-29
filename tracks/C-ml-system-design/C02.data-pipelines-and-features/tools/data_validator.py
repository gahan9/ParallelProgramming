#!/usr/bin/env python3
"""Validate a CSV against a declared JSON schema, and check drift against a reference.

The point is to be a *gate*: exit status 1 means the data would be rejected, so this can sit
in a pipeline step. A pipeline that silently writes garbage is worse than one that fails loudly.

  python data_validator.py --demo
  python data_validator.py --schema schema.json --data batch.csv --reference baseline.csv
  python data_validator.py --write-demo _demo        # materialize editable sample files
  python data_validator.py --print-schema            # the schema format, annotated

Exit status: 0 clean, 1 violations found, 2 bad usage.
Standard library only.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_BAD_INPUT = 2

MAX_EXAMPLES = 4
# Widely used PSI convention: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant.
DEFAULT_MAX_PSI = 0.25
DEFAULT_MAX_CATEGORY_SHIFT = 0.15
PSI_BINS = 10
PSI_EPSILON = 1e-6
# Below this many reference rows a 10-bin PSI is noise, so report it as unmeasurable
# instead of pretending the number means something.
MIN_PSI_REFERENCE_ROWS = 100
# A numeric column with few distinct values (flags, small counts) is compared as
# categories; quantile bins over a binary column are degenerate.
MAX_DISTINCT_FOR_CATEGORICAL = 10


def fail(message):
    print("error: " + message, file=sys.stderr)
    raise SystemExit(EXIT_BAD_INPUT)


# ------------------------------------------------------------- demo fixtures ----

DEMO_SCHEMA = {
    "row_count": {"min": 8, "max": 100000},
    "columns": {
        "txn_id": {"type": "string", "nullable": False, "unique": True},
        "event_time": {"type": "timestamp", "nullable": False},
        "amount": {"type": "float", "nullable": False, "min": 0.0, "max": 25000.0},
        "country": {"type": "category", "nullable": False, "allowed": ["US", "GB", "DE", "IN"]},
        "account_age_days": {"type": "int", "nullable": True, "min": 0, "max": 20000},
        "label": {"type": "int", "nullable": False, "allowed": [0, 1]},
    },
    "drift": {"max_psi": DEFAULT_MAX_PSI, "max_category_shift": DEFAULT_MAX_CATEGORY_SHIFT},
}

DEMO_ROWS = 300
DEMO_SEED = 7


def build_demo_rows(kind):
    """Reference and batch fixtures for --demo.

    The batch differs from the reference in two ways that a real gate must separate:
    a handful of *cell-level* rule violations in the first rows, and *column-level*
    distribution shifts (amount inflated, country mix changed, label rate tripled).
    """
    rng = random.Random(DEMO_SEED if kind == "reference" else DEMO_SEED + 1)
    start = datetime(2026, 5, 1, 9, 0, 0) if kind == "reference" else datetime(2026, 6, 1, 9, 0, 0)
    prefix = "r" if kind == "reference" else "b"

    if kind == "reference":
        amount_scale, countries, weights, label_rate = 1.0, ["US", "GB", "DE", "IN"], [60, 20, 12, 8], 0.08
    else:
        amount_scale, countries, weights, label_rate = 3.0, ["US", "GB", "DE", "IN"], [30, 15, 10, 45], 0.30

    header = ["txn_id", "event_time", "amount", "country", "account_age_days", "label"]
    if kind == "batch":
        header.append("experiment_arm")

    rows = [header]
    for index in range(DEMO_ROWS):
        amount = round(math.exp(rng.gauss(3.9, 0.9)) * amount_scale, 2)
        row = [
            "%s%04d" % (prefix, index + 1),
            (start + timedelta(minutes=4 * index)).isoformat(),
            "%.2f" % min(amount, 24999.99),
            rng.choices(countries, weights=weights)[0],
            str(rng.randint(1, 3000)),
            "1" if rng.random() < label_rate else "0",
        ]
        if kind == "batch":
            row.append(rng.choice(["control", "treatment"]))
        rows.append(row)

    if kind == "batch":
        rows[1][2] = "-5.00"                 # range: negative amount
        rows[2][2] = ""                      # nullability: amount is nullable=false
        rows[3][1] = "not-a-timestamp"       # type: unparseable timestamp
        rows[4][3] = "FR"                    # allowed-values + unseen category
        rows[5][4] = "-12"                   # range: negative account age
        rows[6][5] = "2"                     # allowed-values: label must be 0 or 1
        rows[8][0] = rows[7][0]              # uniqueness: duplicated txn_id
    return rows


# ------------------------------------------------------------------ parsing ----

def load_schema(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            schema = json.load(handle)
    except OSError as error:
        fail("cannot read schema %r: %s" % (path, error.strerror or error))
    except json.JSONDecodeError as error:
        fail("schema %r is not valid JSON: %s (line %d, column %d)"
             % (path, error.msg, error.lineno, error.colno))
    return validate_schema_shape(schema, path)


def validate_schema_shape(schema, source):
    if not isinstance(schema, dict):
        fail("schema %s must be a JSON object with a 'columns' key" % source)
    columns = schema.get("columns")
    if not isinstance(columns, dict) or not columns:
        fail("schema %s must declare a non-empty 'columns' object" % source)
    known_types = {"string", "int", "float", "timestamp", "category"}
    for name, spec in columns.items():
        if not isinstance(spec, dict):
            fail("schema %s: column %r must map to an object" % (source, name))
        column_type = spec.get("type")
        if column_type not in known_types:
            fail("schema %s: column %r has type %r; expected one of %s"
                 % (source, name, column_type, ", ".join(sorted(known_types))))
    return schema


def read_csv(path):
    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError as error:
        fail("cannot read data file %r: %s" % (path, error.strerror or error))
    if not rows:
        fail("data file %r is empty (not even a header row)" % path)
    header = [field.strip() for field in rows[0]]
    if not any(header):
        fail("data file %r has a blank header row" % path)
    records = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row or all(field.strip() == "" for field in row):
            continue
        if len(row) != len(header):
            fail("data file %r line %d has %d fields but the header declares %d"
                 % (path, line_number, len(row), len(header)))
        records.append((line_number, dict(zip(header, [field.strip() for field in row]))))
    return header, records


def parse_typed(value, column_type):
    """Return (ok, parsed). Empty strings are handled by the nullability check, not here."""
    if column_type in ("string", "category"):
        return True, value
    if column_type == "int":
        try:
            return True, int(value)
        except ValueError:
            return False, None
    if column_type == "float":
        try:
            return True, float(value)
        except ValueError:
            return False, None
    if column_type == "timestamp":
        try:
            return True, datetime.fromisoformat(value)
        except ValueError:
            return False, None
    return False, None


# --------------------------------------------------------------- validation ----

class Report:
    def __init__(self):
        self.findings = []

    def add(self, severity, check, column, message, examples=None):
        self.findings.append({
            "severity": severity, "check": check, "column": column,
            "message": message, "examples": examples or [],
        })

    @property
    def errors(self):
        return [item for item in self.findings if item["severity"] == "ERROR"]

    @property
    def warnings(self):
        return [item for item in self.findings if item["severity"] == "WARN"]


def check_structure(header, schema, report):
    declared = set(schema["columns"])
    present = set(header)
    for name in sorted(declared - present):
        report.add("ERROR", "missing-column", name,
                   "declared in the schema but absent from the data")
    for name in sorted(present - declared):
        report.add("WARN", "unexpected-column", name,
                   "present in the data but not declared; a silent upstream schema change")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    for name in duplicates:
        report.add("ERROR", "duplicate-column", name, "appears more than once in the header")


def check_row_count(records, schema, report):
    rules = schema.get("row_count") or {}
    count = len(records)
    minimum = rules.get("min")
    maximum = rules.get("max")
    if minimum is not None and count < minimum:
        report.add("ERROR", "volume", "(table)",
                   "%d rows is below the declared minimum of %d - a truncated or failed upstream job"
                   % (count, minimum))
    if maximum is not None and count > maximum:
        report.add("ERROR", "volume", "(table)",
                   "%d rows exceeds the declared maximum of %d - a duplicated load or a fan-out bug"
                   % (count, maximum))


def check_cells(records, schema, report):
    columns = {}
    for name, spec in schema["columns"].items():
        columns[name] = {
            "spec": spec, "values": [], "nulls": 0,
            "type_errors": [], "null_errors": [], "range_errors": [], "allowed_errors": [],
            "seen": {},
        }

    for line_number, record in records:
        for name, state in columns.items():
            if name not in record:
                continue
            spec = state["spec"]
            raw = record[name]
            if raw == "":
                state["nulls"] += 1
                if not spec.get("nullable", True):
                    state["null_errors"].append("line %d" % line_number)
                continue
            ok, parsed = parse_typed(raw, spec["type"])
            if not ok:
                state["type_errors"].append("line %d: %r" % (line_number, raw))
                continue
            state["values"].append(parsed)
            state["seen"].setdefault(raw, []).append(line_number)

            allowed = spec.get("allowed")
            if allowed is not None and parsed not in allowed and raw not in [str(a) for a in allowed]:
                state["allowed_errors"].append("line %d: %r" % (line_number, raw))
            if spec["type"] in ("int", "float"):
                if spec.get("min") is not None and parsed < spec["min"]:
                    state["range_errors"].append("line %d: %s < %s" % (line_number, raw, spec["min"]))
                if spec.get("max") is not None and parsed > spec["max"]:
                    state["range_errors"].append("line %d: %s > %s" % (line_number, raw, spec["max"]))

    for name, state in columns.items():
        spec = state["spec"]
        for check, bucket, message in (
            ("type", "type_errors", "value does not parse as declared type '%s'" % spec["type"]),
            ("nullability", "null_errors", "empty value in a column declared nullable=false"),
            ("range", "range_errors", "value outside the declared [min, max]"),
            ("allowed-values", "allowed_errors",
             "value outside the declared allowed set %s" % (spec.get("allowed"),)),
        ):
            if state[bucket]:
                report.add("ERROR", check, name, "%d row(s): %s" % (len(state[bucket]), message),
                           state[bucket][:MAX_EXAMPLES])
        if spec.get("unique"):
            repeated = {value: lines for value, lines in state["seen"].items() if len(lines) > 1}
            if repeated:
                examples = ["%r on lines %s" % (value, lines)
                            for value, lines in list(repeated.items())[:MAX_EXAMPLES]]
                report.add("ERROR", "uniqueness", name,
                           "%d duplicated value(s) in a column declared unique" % len(repeated),
                           examples)
        expected_null_rate = spec.get("max_null_rate")
        if expected_null_rate is not None and records:
            rate = state["nulls"] / float(len(records))
            if rate > expected_null_rate:
                report.add("ERROR", "null-rate", name,
                           "null rate %.3f exceeds the declared maximum %.3f"
                           % (rate, expected_null_rate))
    return columns


# --------------------------------------------------------------------- drift ----

def population_stability_index(reference, current, bins=PSI_BINS):
    """PSI over quantile bins of the reference distribution."""
    if len(reference) < bins or not current:
        return None
    ordered = sorted(reference)
    edges = []
    for index in range(1, bins):
        position = int(round(index * len(ordered) / float(bins)))
        position = min(max(position, 0), len(ordered) - 1)
        edges.append(ordered[position])
    edges = sorted(set(edges))
    if not edges:
        return None

    def bucket(value):
        for index, edge in enumerate(edges):
            if value <= edge:
                return index
        return len(edges)

    total_bins = len(edges) + 1
    reference_counts = [0] * total_bins
    current_counts = [0] * total_bins
    for value in reference:
        reference_counts[bucket(value)] += 1
    for value in current:
        current_counts[bucket(value)] += 1

    psi = 0.0
    for index in range(total_bins):
        expected = max(reference_counts[index] / float(len(reference)), PSI_EPSILON)
        actual = max(current_counts[index] / float(len(current)), PSI_EPSILON)
        psi += (actual - expected) * math.log(actual / expected)
    return psi


def mean_and_sd(values):
    if not values:
        return None, None
    mean = sum(values) / float(len(values))
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    return mean, math.sqrt(variance)


def proportions(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(len(values)) or 1.0
    return {key: count / total for key, count in counts.items()}


def check_drift(current_columns, reference_records, schema, report):
    drift_rules = schema.get("drift") or {}
    max_psi = drift_rules.get("max_psi", DEFAULT_MAX_PSI)
    max_category_shift = drift_rules.get("max_category_shift", DEFAULT_MAX_CATEGORY_SHIFT)

    reference_values = {name: [] for name in schema["columns"]}
    for _, record in reference_records:
        for name, spec in schema["columns"].items():
            raw = record.get(name, "")
            if raw == "":
                continue
            ok, parsed = parse_typed(raw, spec["type"])
            if ok:
                reference_values[name].append(parsed)

    for name, spec in schema["columns"].items():
        current = current_columns.get(name, {}).get("values", [])
        reference = reference_values[name]
        if not current or not reference:
            continue
        if spec.get("unique") or spec["type"] == "timestamp":
            continue  # every value is new by construction; drift is not a meaningful question

        distinct = len(set(reference)) + len(set(current) - set(reference))
        numeric = spec["type"] in ("int", "float")
        if numeric and distinct > MAX_DISTINCT_FOR_CATEGORICAL:
            check_numeric_drift(name, reference, current, max_psi, report)
        else:
            check_categorical_drift(name, reference, current, max_category_shift, numeric, report)


def check_numeric_drift(name, reference, current, max_psi, report):
    reference = [float(value) for value in reference]
    current = [float(value) for value in current]
    reference_mean, reference_sd = mean_and_sd(reference)
    current_mean, _ = mean_and_sd(current)
    detail = "mean %.4f -> %.4f" % (reference_mean, current_mean)
    if reference_sd:
        detail += " (%+.2f reference sd)" % ((current_mean - reference_mean) / reference_sd)

    if len(reference) < MIN_PSI_REFERENCE_ROWS:
        report.add("WARN", "drift", name,
                   "only %d reference rows (need %d) for a stable PSI, so drift is unmeasured; %s"
                   % (len(reference), MIN_PSI_REFERENCE_ROWS, detail))
        return
    psi = population_stability_index(reference, current)
    if psi is None:
        report.add("WARN", "drift", name, "PSI could not be computed; %s" % detail)
    elif psi > max_psi:
        report.add("ERROR", "drift", name,
                   "PSI %.4f exceeds the declared maximum %.4f; %s" % (psi, max_psi, detail))
    else:
        report.add("INFO", "drift", name, "PSI %.4f within tolerance; %s" % (psi, detail))


def check_categorical_drift(name, reference, current, max_category_shift, numeric, report):
    reference_share = proportions([str(value) for value in reference])
    current_share = proportions([str(value) for value in current])
    unseen = sorted(set(current_share) - set(reference_share))
    if unseen:
        report.add("ERROR", "drift", name,
                   "%d %s value(s) never seen in the reference"
                   % (len(unseen), "numeric" if numeric else "category"), unseen[:MAX_EXAMPLES])
    shifts = []
    for key in sorted(set(reference_share) | set(current_share)):
        delta = current_share.get(key, 0.0) - reference_share.get(key, 0.0)
        if abs(delta) > max_category_shift:
            shifts.append("%s: %.3f -> %.3f (%+.3f)"
                          % (key, reference_share.get(key, 0.0), current_share.get(key, 0.0), delta))
    if shifts:
        report.add("ERROR", "drift", name,
                   "%d share(s) moved by more than %.3f" % (len(shifts), max_category_shift),
                   shifts[:MAX_EXAMPLES])
    elif not unseen:
        report.add("INFO", "drift", name, "all shares within %.3f" % max_category_shift)


# -------------------------------------------------------------------- output ----

def print_report(report, data_label, row_count, gate_on_warnings):
    print("data validation report")
    print("  source : %s" % data_label)
    print("  rows   : %d" % row_count)
    print()

    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    findings = sorted(report.findings, key=lambda item: (order[item["severity"]], item["check"],
                                                         item["column"]))
    if not findings:
        print("  no findings")
    for item in findings:
        print("  [%-5s] %-18s %-18s %s"
              % (item["severity"], item["check"], item["column"], item["message"]))
        for example in item["examples"]:
            print("            example: %s" % example)

    errors = len(report.errors)
    warnings = len(report.warnings)
    print()
    print("summary: %d error(s), %d warning(s)" % (errors, warnings))
    if errors or (gate_on_warnings and warnings):
        print("GATE: REJECT - do not publish this batch downstream")
        return EXIT_VIOLATIONS
    if warnings:
        print("GATE: PASS with warnings - review the unexpected-column findings")
        return EXIT_OK
    print("GATE: PASS")
    return EXIT_OK


def write_demo(directory):
    try:
        os.makedirs(directory, exist_ok=True)
        schema_path = os.path.join(directory, "schema.json")
        with open(schema_path, "w", encoding="utf-8") as handle:
            json.dump(DEMO_SCHEMA, handle, indent=2)
            handle.write("\n")
        for name, rows in (("reference.csv", build_demo_rows("reference")),
                           ("batch.csv", build_demo_rows("batch"))):
            with open(os.path.join(directory, name), "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
    except OSError as error:
        fail("cannot write demo files into %r: %s" % (directory, error.strerror or error))
    print("wrote %s/schema.json, %s/reference.csv, %s/batch.csv" % (directory, directory, directory))
    print("now run:")
    print("  python %s --schema %s/schema.json --data %s/batch.csv --reference %s/reference.csv"
          % (os.path.basename(__file__), directory, directory, directory))
    return EXIT_OK


def records_from_rows(rows):
    header = rows[0]
    return header, [(index, dict(zip(header, row))) for index, row in enumerate(rows[1:], start=2)]


def main(argv):
    parser = argparse.ArgumentParser(
        prog="data_validator.py",
        description="Validate a CSV against a declared JSON schema and check drift vs a reference.",
    )
    parser.add_argument("--schema", help="path to the JSON schema")
    parser.add_argument("--data", help="path to the CSV to validate")
    parser.add_argument("--reference", help="optional reference CSV for drift checks")
    parser.add_argument("--demo", action="store_true",
                        help="run against built-in sample data with deliberate violations")
    parser.add_argument("--write-demo", metavar="DIR",
                        help="write the sample schema and CSVs to DIR so you can edit them")
    parser.add_argument("--print-schema", action="store_true",
                        help="print the annotated schema format and exit")
    parser.add_argument("--gate-on-warnings", action="store_true",
                        help="treat warnings as blocking too")
    args = parser.parse_args(argv)

    if args.print_schema:
        print(json.dumps(DEMO_SCHEMA, indent=2))
        print()
        print("keys per column: type (string|int|float|timestamp|category), nullable,")
        print("min, max, allowed, unique, max_null_rate")
        print("table-level keys: row_count.min, row_count.max, drift.max_psi,")
        print("drift.max_category_shift")
        return EXIT_OK

    if args.write_demo:
        return write_demo(args.write_demo)

    if args.demo:
        if args.schema or args.data:
            fail("--demo uses built-in data; do not combine it with --schema/--data")
        schema = validate_schema_shape(DEMO_SCHEMA, "(built-in demo)")
        header, records = records_from_rows(build_demo_rows("batch"))
        _, reference_records = records_from_rows(build_demo_rows("reference"))
        data_label = ("built-in demo batch (%d rows; 7 injected cell violations and 3 column shifts)"
                      % DEMO_ROWS)
    else:
        if not args.schema or not args.data:
            fail("give --schema and --data, or use --demo (see --help)")
        schema = load_schema(args.schema)
        header, records = read_csv(args.data)
        reference_records = []
        if args.reference:
            _, reference_records = read_csv(args.reference)
        data_label = args.data

    report = Report()
    check_structure(header, schema, report)
    check_row_count(records, schema, report)
    columns = check_cells(records, schema, report)
    if reference_records:
        check_drift(columns, reference_records, schema, report)
    else:
        report.add("INFO", "drift", "(table)",
                   "no --reference given, so distribution drift was not checked")

    return print_report(report, data_label, len(records), args.gate_on_warnings)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        raise SystemExit(EXIT_OK)
    except KeyboardInterrupt:
        raise SystemExit(130)
