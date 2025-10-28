import re
import sys
from pathlib import Path

SHORT_GEN_DEFAULT = 12
LONG_GEN_FALLBACK = 128

SHORT_TASK_KEYS = {
    "BBBOOLEANEXPRESSIONS","BBCAUSALJUDGEMENT","BBDATEUNDERSTANDING","BBDISAMBIGUATIONQA",
    "BBDYCKLANGUAGES","BBFORMALFALLACIES","BBGEOMETRICSHAPES","BBHYPERBATON",
    "BBLOGICALDEDUCTION","BBMOVIERECOMMENDATION","BBNAVIGATE","BBOBJECTCOUNTING",
    "BBPENGUINSINATABLE","BBREASONINGABOUTCOLOREDOBJECTS","BBRUINNAMES",
    "BBSALIENTTRANSLATIONERRORDETECTION","BBSNARKS","BBSPORTSUNDERSTANDING",
    "BBTEMPORALSEQUENCES","BBTRACKINGSHUFFLEDOBJECTS","BBWEBOFLIES","BBWORDSORTING",
    "BBBBQLITEJSON",
    "BBCODELINEDESCRIPTION","BBCONCEPTUALCOMBINATIONS","BBEMOJIMOVIE","BBHINDUKNOWLEDGE",
    "BBKNOWNUNKNOWNS","BBLANGUAGEIDENTIFICATION","BBLOGICGRIDPUZZLE","BBMISCONCEPTIONSRUSSIAN",
    "BBNOVELCONCEPTS","BBPLAYDIALOGSAMEORDIFFERENT","BBSTRANGESTORIES","BBSTRATEGYQA",
    "BBSYMBOLINTERPRETATION","BBVITAMINCFACTVERIFICATION","BBWINOWHY",
}

MC_TO_GEN_METRIC = {
    'accuracy': 'exact_match',
    'accuracy_multiple_ans': 'exact_match_multiple_ans',
}
GEN_METRICS = {'exact_match', 'exact_match_multiple_ans', 'rouge'}

ROUGE_TASKS = {
    "BBCONLANGTRANSLATION",
}

def default_maxlen(task_key: str) -> int:
    return SHORT_GEN_DEFAULT if task_key in SHORT_TASK_KEYS else LONG_GEN_FALLBACK

BLOCK_HDR = re.compile(r'^\s*(D/([^/\s]+)/EVAL/InterfaceInfo(?::)?\s*:)\s*$')
TOP_BLOCK_HDR = re.compile(r'^\s*(D/InterfaceInfo\s*:)\s*$')

KV = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$')

INLINE_IFACE_LMMC = re.compile(
    r'^\s*(D/([^/\n]+)/EVAL/InterfaceInfo\.interface\s*=\s*")(?:lm|mc)("?)(\s*)$'
)
TOP_INLINE_IFACE_LMMC = re.compile(
    r'^\s*(D/InterfaceInfo\.[^=\n]*interface\s*=\s*")(?:lm|mc)("?)(\s*)$'
)

INLINE_MAX = re.compile(
    r'^\s*D/([^/\n]+)/EVAL/InterfaceInfo\.max_gen_length\s*=\s*\d+\s*$'
)

TOP_DATASET_HDR = re.compile(r'^\s*(D/BigBenchDataset\s*:)\s*$')
TOP_SAMPLE_HDR  = re.compile(r'^\s*(D/BigBenchSampleDataset\s*:)\s*$')

INLINE_METRICS_LINE = re.compile(
    r'^\s*(D/[^/\n]+/EVAL/(?:BigBenchDataset|BigBenchSampleDataset)\.metrics\s*=\s*)(\[[^\]]*\])\s*$'
)

def _normalize_metrics_value(text: str) -> str:
    val = text.strip()
    inner = val.strip()[1:-1].strip()
    items = [x.strip().strip('"').strip("'") for x in inner.split(',') if x.strip()]
    if not items:
        return '["exact_match"]'
    fixed = []
    for it in items:
        if it in MC_TO_GEN_METRIC:
            fixed.append(MC_TO_GEN_METRIC[it])
        else:
            fixed.append(it)
    if not any(m in GEN_METRICS for m in fixed):
        fixed = ["exact_match"]
    return "[" + ", ".join(f'"{m}"' for m in fixed) + "]"

def patch_text(text: str) -> str:
    lines = text.splitlines()

    patched = []
    for ln in lines:
        m_top = TOP_INLINE_IFACE_LMMC.match(ln)
        if m_top:
            ln = TOP_INLINE_IFACE_LMMC.sub(r'\1gen"\3', ln)
        else:
            m_inl = INLINE_IFACE_LMMC.match(ln)
            if m_inl:
                ln = INLINE_IFACE_LMMC.sub(r'\1gen"\4', ln)
        m_inline_metrics = INLINE_METRICS_LINE.match(ln)
        if m_inline_metrics:
            prefix, arr = m_inline_metrics.groups()
            ln = prefix + _normalize_metrics_value(arr)
        patched.append(ln)
    lines = patched

    existing_max_tasks = set()

    out_lines = []
    i = 0
    while i < len(lines):
        ln = lines[i]

        m_top_hdr = TOP_BLOCK_HDR.match(ln)
        if m_top_hdr:
            block = [ln]; i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("D/"):
                block.append(lines[i]); i += 1
            newb = []
            for b in block:
                mkv = KV.match(b)
                if mkv and mkv.group(1) == "interface":
                    raw = mkv.group(2).strip().strip('"')
                    if raw in {"lm","mc"}:
                        newb.append('    interface = "gen"')
                    else:
                        newb.append(b)
                else:
                    newb.append(b)
            out_lines.extend(newb)
            continue

        if m_top_ds:
            block = [ln]; i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("D/"):
                block.append(lines[i]); i += 1
            newb, saw_metrics = [], False
            for b in block:
                mkv = KV.match(b)
                if mkv and mkv.group(1) == "metrics":
                    saw_metrics = True
                    newb.append('    metrics = ["exact_match"]')
                else:
                    newb.append(b)
            if not saw_metrics:
                newb.append('    metrics = ["exact_match"]')
            out_lines.extend(newb)
            continue

        m_top_smpl = TOP_SAMPLE_HDR.match(ln)
        if m_top_smpl:
            block = [ln]; i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("D/"):
                block.append(lines[i]); i += 1
            out_lines.extend(block)
            continue

        m_hdr = BLOCK_HDR.match(ln)
        if m_hdr:
            task_key = m_hdr.group(2)
            block = [ln]; i += 1
            saw_max = False

            while i < len(lines) and not lines[i].lstrip().startswith("D/"):
                b = lines[i]
                block.append(b)
                mkv = KV.match(b)
                if mkv and mkv.group(1) == "max_gen_length":
                    saw_max = True
                i += 1

            newb = []
            for b in block:
                mkv = KV.match(b)
                if mkv and mkv.group(1) == "interface":
                    raw = mkv.group(2).strip().strip('"')
                    if raw in {"lm","mc"}:
                        newb.append('    interface = "gen"')
                    else:
                        newb.append(b)
                else:
                    newb.append(b)

            if not saw_max:
                ml = default_maxlen(task_key)
                newb.append(f'    max_gen_length = {ml}')
                existing_max_tasks.add(task_key)

            out_lines.extend(newb)
            continue

        out_lines.append(ln)
        i += 1

  
    final = []
    i = 0
    task_metrics_written = set()

    seen_iface_tasks = set()

    for ln in out_lines:
        m = re.match(r'^\s*D/([^/\n]+)/EVAL/InterfaceInfo\.interface\s*=\s*"gen"', ln)
        if m:
            seen_iface_tasks.add(m.group(1))

    while i < len(out_lines):
        ln = out_lines[i]

        m_inline_metrics = INLINE_METRICS_LINE.match(ln)
        if m_inline_metrics:
            prefix, arr = m_inline_metrics.groups()
            norm = _normalize_metrics_value(arr)
            ln = prefix + norm
            mt = re.match(r'^\s*D/([^/\n]+)/EVAL/', ln)
            if mt:
                task_metrics_written.add(mt.group(1))
            final.append(ln); i += 1
            continue

        m_ds_hdr = re.match(r'^\s*(D/([^/\s]+)/EVAL/(BigBenchDataset|BigBenchSampleDataset)\s*:)\s*$', ln)
        if m_ds_hdr:
            task_key = m_ds_hdr.group(2)
            block = [ln]; i += 1
            saw_metrics = False
            while i < len(out_lines) and not out_lines[i].lstrip().startswith("D/"):
                b = out_lines[i]
                mkv = KV.match(b)
                if mkv and mkv.group(1) == "metrics":
                    saw_metrics = True
                    raw = mkv.group(2).strip()
                    new_val = _normalize_metrics_value(raw)
                    block.append(f'    metrics = {new_val}')
                else:
                    block.append(b)
                i += 1

            if not saw_metrics:
                if task_key in ROUGE_TASKS:
                    block.append('    metrics = ["rouge"]')
                else:
                    block.append('    metrics = ["exact_match"]')
                task_metrics_written.add(task_key)

            final.extend(block)
            continue

        final.append(ln); i += 1

    return "\n".join(final) + "\n"

def main(in_path: str, out_path: str):
    src = Path(in_path).read_text(encoding="utf-8")
    patched = patch_text(src)
    Path(out_path).write_text(patched, encoding="utf-8")
    print(f"Wrote patched gin to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python gen_bigbench.py colm/datasets/bigbench.gin [colm/datasets/bigbench.gen.gin]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path + ".gen.gin"
    main(in_path, out_path)