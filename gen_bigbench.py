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

def patch_text(text: str) -> str:
    lines = text.splitlines()
    out_lines = []

    patched = []
    for ln in lines:
        m_top = TOP_INLINE_IFACE_LMMC.match(ln)
        if m_top:
            ln = TOP_INLINE_IFACE_LMMC.sub(r'\1gen"\3', ln)
        else:
            m_inl = INLINE_IFACE_LMMC.match(ln)
            if m_inl:
                ln = INLINE_IFACE_LMMC.sub(r'\1gen"\4', ln)
        patched.append(ln)
    lines = patched

    existing_max_tasks = set()
    for ln in lines:
        m_inline_max = INLINE_MAX.match(ln)
        if m_inline_max:
            existing_max_tasks.add(m_inline_max.group(1))

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

        m_hdr = BLOCK_HDR.match(ln)
        if m_hdr:
            task_key = m_hdr.group(2)
            block = [ln]; i += 1
            saw_max = False

            while i < len(lines) and not lines[i].lstrip().startswith("D/"):
                block.append(lines[i])
                mkv = KV.match(lines[i])
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

            if saw_max:
                existing_max_tasks.add(task_key)
            else:
                ml = default_maxlen(task_key)
                newb.append(f'    max_gen_length = {ml}')
                existing_max_tasks.add(task_key)

            out_lines.extend(newb)
            continue

        out_lines.append(ln); i += 1

    final = []
    i = 0
    while i < len(out_lines):
        ln = out_lines[i]
        m_inl = INLINE_IFACE_LMMC.match(ln)
        if m_inl:
            task = m_inl.group(2)
            ln = INLINE_IFACE_LMMC.sub(r'\1gen"\4', ln)
            final.append(ln)
            if task not in existing_max_tasks:
                ml = default_maxlen(task)
                final.append(f'D/{task}/EVAL/InterfaceInfo.max_gen_length = {ml}')
                existing_max_tasks.add(task)
            i += 1
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