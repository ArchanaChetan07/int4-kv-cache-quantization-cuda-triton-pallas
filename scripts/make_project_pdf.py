"""Generate docs/INT4_KV_Pallas_Project_Report.pdf -- the complete project explainer.

Every figure and number in the output is measured by this repository. Charts are
the ones scripts/make_pallas_charts.py produces.

Deliberately ASCII-only in body text: ReportLab's built-in fonts have no glyphs
for characters outside Latin-1, and a missing glyph renders as a solid black box
rather than failing loudly. Superscripts use the <super> tag, never Unicode.

    python scripts/make_project_pdf.py
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "docs", "assets")
OUT = os.path.join(HERE, "docs", "INT4_KV_Pallas_Project_Report.pdf")

INK = colors.HexColor("#14171F")
MUTED = colors.HexColor("#5B6472")
FAINT = colors.HexColor("#8A93A3")
RULE = colors.HexColor("#D8DDE5")
TEAL = colors.HexColor("#0B6E75")
TEAL_BG = colors.HexColor("#E8F2F3")
AMBER = colors.HexColor("#8A5B0B")
AMBER_BG = colors.HexColor("#FBF3E2")
RED = colors.HexColor("#B02A2A")
RED_BG = colors.HexColor("#FAEBEB")
GREEN = colors.HexColor("#2B7A3D")
SURFACE = colors.HexColor("#F5F6F8")

TITLE = "INT4 KV-Cache Quantization with Fused Flash-Decoding Attention"
SUBTITLE = "One decode kernel, three backends: CUDA, Triton, and Pallas/JAX"
AUTHOR = "Archana Suresh Patil"
DATE = "27 August 2026"
REPO = "github.com/ArchanaChetan07/int4-kv-cache-quantization-cuda-triton-pallas"

ss = getSampleStyleSheet()


def S(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.5, leading=14, textColor=INK,
                alignment=TA_LEFT, spaceAfter=8)
    base.update(kw)
    return ParagraphStyle(name, **base)


BODY = S("body")
SMALL = S("small", fontSize=8.5, leading=12, textColor=MUTED)
H1 = S("h1", fontName="Helvetica-Bold", fontSize=16, leading=20,
       textColor=INK, spaceBefore=18, spaceAfter=4)
H2 = S("h2", fontName="Helvetica-Bold", fontSize=11, leading=15,
       textColor=INK, spaceBefore=13, spaceAfter=4)
EYEBROW = S("eyebrow", fontName="Helvetica-Bold", fontSize=7.5, leading=11,
            textColor=TEAL, spaceAfter=2)
DECK = S("deck", fontSize=9.5, leading=14, textColor=MUTED, spaceAfter=10)
CODE = S("code", fontName="Courier", fontSize=8.5, leading=12.5,
         textColor=INK, backColor=SURFACE, borderPadding=7,
         spaceBefore=5, spaceAfter=9)
CAPTION = S("caption", fontSize=8, leading=11.5, textColor=FAINT, spaceAfter=14)
BULLET = S("bullet", leftIndent=13, bulletIndent=3, spaceAfter=5)

CELL = S("cell", fontSize=8.2, leading=11.5, spaceAfter=0)
CELL_B = S("cellb", fontSize=8.2, leading=11.5, spaceAfter=0,
           fontName="Helvetica-Bold")
CELL_H = S("cellh", fontSize=7.5, leading=10.5, spaceAfter=0,
           fontName="Helvetica-Bold", textColor=MUTED)
CELL_M = S("cellm", fontName="Courier", fontSize=7.8, leading=11, spaceAfter=0)


def p(text, style=BODY):
    return Paragraph(text, style)


def sect(heading, *paras):
    """Heading glued to its first paragraph so it never orphans at a page foot."""
    return [KeepTogether([p(heading, H2), p(paras[0], BODY)])] +            [p(t, BODY) for t in paras[1:]]


def rule(space_before=4, space_after=10):
    t = Table([[""]], colWidths=[6.9 * inch], rowHeights=[0.4])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.6, RULE)]))
    return [Spacer(1, space_before), t, Spacer(1, space_after)]


def table(rows, widths, header=True, zebra=True):
    data = []
    for i, row in enumerate(rows):
        style = CELL_H if (header and i == 0) else None
        data.append([c if isinstance(c, Paragraph)
                     else p(str(c), style or CELL) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), SURFACE),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#BFC6D2"))]
    if zebra:
        start = 1 if header else 0
        for i in range(start, len(rows)):
            if (i - start) % 2 == 1:
                cmds.append(("BACKGROUND", (0, i), (-1, i),
                             colors.HexColor("#FBFCFD")))
    t.setStyle(TableStyle(cmds))
    return t


def callout(tag, text, tone="teal"):
    fg, bg = {"teal": (TEAL, TEAL_BG), "amber": (AMBER, AMBER_BG),
              "red": (RED, RED_BG)}[tone]
    inner = [p(tag.upper(), S("ctag", fontName="Helvetica-Bold", fontSize=7.5,
                              leading=11, textColor=fg, spaceAfter=4)),
             p(text, S("ctext", fontSize=9, leading=13.5))]
    t = Table([[inner]], colWidths=[6.9 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, fg),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
    ]))
    return [Spacer(1, 5), t, Spacer(1, 11)]


def figure(name, width=6.5, caption=""):
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path):
        return [p(f"[missing figure: {name}]", CAPTION)]
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = width * inch
    img = Image(path, width=w, height=w * ih / iw)
    out = [Spacer(1, 4), img]
    if caption:
        out += [Spacer(1, 4), p(caption, CAPTION)]
    else:
        out += [Spacer(1, 12)]
    return out


# ---------------------------------------------------------------------------
# page furniture
# ---------------------------------------------------------------------------

def cover_page(canv, doc):
    canv.saveState()
    canv.setFillColor(TEAL)
    canv.rect(0, LETTER[1] - 0.32 * inch, LETTER[0], 0.32 * inch, stroke=0, fill=1)
    canv.setFillColor(FAINT)
    canv.setFont("Helvetica", 7.5)
    canv.drawString(0.85 * inch, 0.6 * inch, REPO)
    canv.restoreState()


def body_page(canv, doc):
    canv.saveState()
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.5)
    canv.line(0.85 * inch, LETTER[1] - 0.68 * inch,
              LETTER[0] - 0.85 * inch, LETTER[1] - 0.68 * inch)
    canv.setFillColor(FAINT)
    canv.setFont("Helvetica", 7.5)
    canv.drawString(0.85 * inch, LETTER[1] - 0.6 * inch,
                    "INT4 KV-Cache Quantization / CUDA, Triton, Pallas")
    canv.drawRightString(LETTER[0] - 0.85 * inch, 0.55 * inch, str(canv.getPageNumber()))
    canv.drawString(0.85 * inch, 0.55 * inch, AUTHOR)
    canv.restoreState()


def build():
    doc = BaseDocTemplate(OUT, pagesize=LETTER,
                          leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                          topMargin=0.85 * inch, bottomMargin=0.8 * inch,
                          title=TITLE, author=AUTHOR)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=cover_page),
        PageTemplate(id="body", frames=[frame], onPage=body_page),
    ])
    s = []

    # ---------------- cover ----------------
    s += [Spacer(1, 1.5 * inch)]
    s += [p("TECHNICAL PROJECT REPORT", S("cove", fontName="Helvetica-Bold",
                                          fontSize=8.5, leading=12,
                                          textColor=TEAL, spaceAfter=14))]
    s += [p(TITLE, S("covt", fontName="Helvetica-Bold", fontSize=25, leading=30,
                     textColor=INK, spaceAfter=12))]
    s += [p(SUBTITLE, S("covs", fontSize=13, leading=18, textColor=MUTED,
                        spaceAfter=26))]
    s += rule(0, 14)
    s += [table([
        ["Author", AUTHOR],
        ["Date", DATE],
        ["Repository", p(REPO, CELL_M)],
        ["License", "Apache-2.0"],
        ["Test suite", "76 passing with JAX, 53 without; CI green on 5 jobs"],
    ], [1.4 * inch, 5.5 * inch], header=False, zebra=False)]
    s += [Spacer(1, 26)]
    s += callout(
        "What this document is",
        "A complete account of a kernel-engineering project: what it computes, how "
        "it is implemented three times over, what was measured, what was learned "
        "from porting between two kernel-programming models, and -- stated as "
        "plainly as the results -- what has NOT been established.")
    s += [NextPageTemplate("body"), PageBreak()]

    # ---------------- 1 ----------------
    s += [p("01", EYEBROW), p("Executive summary", H1)]
    s += [p("The problem", H2)]
    s += [p(
        "Serving a large language model is dominated by memory, not arithmetic. During "
        "decoding, every generated token must read the entire key-value (KV) cache for "
        "every attention head. That cache grows linearly with sequence length and batch "
        "size, and reading it saturates memory bandwidth long before the GPU runs out of "
        "compute. The KV cache is therefore the binding constraint on how many concurrent "
        "requests a server can hold and how fast it can answer them.", BODY)]
    s += [p("The approach", H2)]
    s += [p(
        "Store the keys as 4-bit integers instead of 16-bit floats -- a 4x reduction -- and "
        "fuse the dequantization into the attention kernel so the full-precision key matrix "
        "is never written to memory at all. Combine this with flash decoding, which streams "
        "the online softmax over paged KV blocks and never materializes the attention "
        "matrix. The result moves roughly a quarter of the bytes.", BODY)]
    s += [p("What makes this project unusual", H2)]
    s += [p(
        "The same algorithm is implemented three times -- in CUDA C++, in Triton, and in "
        "JAX/Pallas -- and all three are validated against a single NumPy reference. Holding "
        "the algorithm and the oracle fixed turns a re-implementation into a controlled "
        "experiment: any difficulty encountered is attributable to the tool, not to the "
        "mathematics. The Pallas port was the experiment, and the findings in section 06 are "
        "its result.", BODY)]
    s += [Spacer(1, 4)]
    s += [table([
        ["Measured outcome", "Result"],
        ["Quantizer vs NumPy oracle (CUDA and Pallas)",
         p("<b>0.000% bin disagreement</b>", CELL)],
        ["KV memory compression vs FP16, incl. scale/zero-point overhead",
         p("<b>3.98x</b> measured", CELL)],
        ["Fused attention -- agreement with FP32 reference (head_dim 64)",
         p("MAE <b>3.1e-08</b>", CELL)],
        ["Kernel latency, batch 8 x 32 heads x dim 128 x seq 2048",
         p("<b>2.84 ms</b>, 3.9x over a serial baseline (T1000)", CELL)],
        ["Pallas attention error vs a float64 evaluation",
         p("within <b>1.4x</b> of the float32 accumulation floor", CELL)],
        ["Test suite", p("<b>76 passing</b> with JAX, 53 without; 5/5 CI jobs green", CELL)],
    ], [4.0 * inch, 2.9 * inch])]
    s += [Spacer(1, 8)]
    s += callout(
        "The honest headline",
        "Correctness is established on every backend. <b>Performance is not.</b> Pallas has "
        "no CPU code generator, and the available GPU is below the hardware floor of both "
        "live Pallas GPU backends, so every Pallas timing in this repository is "
        "interpret-mode and is correctness evidence rather than a performance number. "
        "Section 09 states the limits precisely.", "amber")

    # ---------------- 2 ----------------
    s += [PageBreak(), p("02", EYEBROW), p("What the kernel computes", H1)]
    s += [p("Per-channel asymmetric INT4 quantization", H2)]
    s += [p(
        "Each channel of the key cache gets its own scale and zero-point, derived from that "
        "channel's observed range. Per-channel granularity matters: activation channels in "
        "transformers have wildly different dynamic ranges, and a single scale across all of "
        "them wastes most of the 16 available levels on the widest channel.", BODY)]
    s += [p(
        "scale[c] = (max[c] - min[c]) / 15<br/>"
        "zp[c]    = -min[c] / scale[c]<br/>"
        "q        = clip(round((kv - min[c]) / scale[c]), 0, 15)", CODE)]
    s += [p(
        "Sixteen levels means fifteen intervals, so the maximum round-trip error per value is "
        "half a scale step. That bound is asserted directly in the test suite rather than "
        "assumed.", BODY)]
    s += [p("Flash decoding with online softmax", H2)]
    s += [p(
        "A naive attention implementation computes all logits, takes a softmax, then weights "
        "the values -- which requires holding the full attention matrix. Online softmax "
        "instead keeps a running maximum and a running normalizer, rescaling the accumulated "
        "output each time a larger logit appears. Memory stays constant in sequence length.", BODY)]
    s += [p(
        "m_new = max(m_old, max(logits))<br/>"
        "corr  = exp(m_old - m_new)<br/>"
        "l_new = l_old * corr + sum(exp(logits - m_new))<br/>"
        "o_new = o_old * corr + V @ exp(logits - m_new)", CODE)]
    s += [p(
        "Fusing the two is where the memory saving is actually realized. Keys are read as "
        "INT4, dequantized in registers, consumed immediately by the dot product, and "
        "discarded. The FP32 key matrix exists only transiently inside the kernel.", BODY)]
    s += callout(
        "Why an empty page is dangerous",
        "A paged KV cache can contain pages with zero valid rows. The online-softmax update "
        "then computes exp(-inf - -inf), which is NaN and silently poisons the whole "
        "sequence. Every implementation here guards it, and the guard differs by backend -- "
        "see section 06.")

    # ---------------- 3 ----------------
    s += [PageBreak(), p("03", EYEBROW), p("Three implementations, one oracle", H1)]
    s += [p(
        "The NumPy reference is the single source of truth. It is deliberately slow, "
        "deliberately simple, and every backend is measured against it and nothing else.", BODY)]
    s += [table([
        ["Implementation", "Covers", "Parallel decomposition"],
        [p("<b>NumPy reference</b><br/>quantize_int4_ref.py<br/>flash_decode_ref.py", CELL),
         "quantizer + attention", "none -- the definition of correct"],
        [p("<b>CUDA C++</b><br/>flash_decode_int4.cu", CELL),
         "quantizer + attention",
         "one thread block per (batch, head); warps stride over positions; "
         "warp-shuffle reduction; cross-warp log-sum-exp merge"],
        [p("<b>Triton</b><br/>quantize_int4_triton.py", CELL),
         "quantizer only",
         "one program per channel, looping over rows inside the program"],
        [p("<b>Pallas / JAX</b><br/>quantize_int4_pallas.py<br/>flash_decode_pallas.py", CELL),
         "quantizer + attention",
         "grid over (batch, heads, blocks); accumulators are resident output "
         "blocks carried across a sequential grid"],
    ], [1.85 * inch, 1.35 * inch, 3.7 * inch])]
    s += [Spacer(1, 6)]
    s += [p(
        "There is no Triton attention kernel. That is a real gap and it is enforced in code: "
        "requesting backend='triton' for attention raises an error naming the gap rather than "
        "silently returning reference results. An earlier version of the dispatch layer did "
        "fall through silently -- see section 08.", BODY)]
    s += [p("Validating without an accelerator", H2)]
    s += [p(
        "Both kernel DSLs offer an interpreter that runs kernels as ordinary host code. Triton "
        "has TRITON_INTERPRET=1; Pallas has pallas_call(interpret=True). Continuous "
        "integration uses both, so every numerical claim is re-checked on every push using "
        "free CPU runners. This is what keeps a kernel repository honest when the author does "
        "not own the target hardware.", BODY)]

    # ---------------- 4 ----------------
    s += [PageBreak(), p("04", EYEBROW), p("Why port a kernel you already wrote", H1)]
    s += [p(
        "The Pallas port exists to answer a question about tooling, not about attention. If "
        "an unfamiliar algorithm had been chosen, every obstacle would be ambiguous: is this "
        "hard because Pallas is hard, or because the mathematics is not yet understood? "
        "Holding the algorithm and the oracle fixed removes that ambiguity. Every hour of "
        "difficulty becomes attributable to the tool. That attribution is the deliverable.", BODY)]
    s += [p(
        "To keep the exercise honest, fourteen predictions were written down before any code "
        "was ported -- which Triton habits would transfer, which would need restructuring, and "
        "which would actively mislead. They were scored afterwards, including the ones that "
        "were wrong. A retrospective written after the fact can always make its author look "
        "prescient; a pre-registered one cannot.", BODY)]
    s += [table([
        ["Verdict", "Count", "Meaning"],
        [p("<b>CONFIRMED</b>", CELL), "10", "the prediction held"],
        [p("<b>REFUTED</b>", CELL), "1",
         "scalar prefetch was predicted to be required for variable-length pages; "
         "a plain input specification worked. It is a performance mechanism, not a "
         "correctness requirement."],
        [p("<b>UNRESOLVED</b>", CELL), "3",
         "all three require TPU silicon to test and are not asserted"],
    ], [1.2 * inch, 0.7 * inch, 5.0 * inch])]
    s += [Spacer(1, 6)]
    s += callout(
        "The most useful reframing",
        "The project was framed as 'where the mental model transferred and where it misled'. "
        "The more useful axis turned out to be <b>what disappears</b>. Warp machinery, "
        "shared-memory staging, the cross-warp merge, occupancy tuning -- these are not "
        "translated into Pallas equivalents. They stop being things. A transition guide "
        "organized as 'here is the Pallas way to do X' is the wrong shape for half the work, "
        "because for that half X is no longer something you do.")

    # ---------------- 5 ----------------
    s += [PageBreak(), p("05", EYEBROW), p("The hardware constraint", H1)]
    s += [p(
        "A scoping check before writing code invalidated the obvious version of this project, "
        "and it is worth stating plainly because it shapes everything after it.", DECK)]
    s += figure("pallas_backend_matrix.png", 6.6,
                "Figure 1. Which kernel backend runs on which silicon. Measured and "
                "documented, not estimated.")
    s += [p(
        "Pallas has two GPU backends. Mosaic GPU, the recommended one, targets Hopper and "
        "Blackwell. The Triton backend, which would otherwise have covered older cards, is "
        "deprecated with removal announced. The development machine is an NVIDIA T1000 -- "
        "Turing, compute capability 7.5 -- which sits below the floor of both.", BODY)]
    s += [p(
        "Pallas also has no CPU code generator at all. Calling pallas_call with "
        "interpret=False on a CPU host raises 'Only interpret mode is supported on CPU "
        "backend'. There is no slow-but-real fallback.", BODY)]
    s += callout(
        "Consequence",
        "Porting Triton to Pallas <i>on GPU</i> would have compiled the Pallas code back "
        "through Triton -- comparing a language against itself through a deprecated adapter. "
        "The port therefore targets Mosaic TPU, where there is no shared substrate and the "
        "comparison is real: no warps, no manual shared memory, no thread count, a fixed "
        "(8,128) register tiling, and a grid that runs sequentially rather than as a swarm of "
        "independent programs.", "red")

    # ---------------- 6 ----------------
    s += [PageBreak(), p("06", EYEBROW), p("Four findings that required building it", H1)]

    s += sect(
        "6.1  The oracle is less accurate than the kernel it validates",
        "Validating against the FP32 reference alone is insufficient. Measured against a "
        "float64 evaluation of the same mathematics, at head_dim 128 the reference's own "
        "error is 9.26e-07 while the Pallas kernel's is 2.59e-07. The reference is not broken "
        "-- it is simply sitting at the float32 accumulation floor, sqrt(D) * eps * |output|, "
        "which predicts 8.0e-07.",
        "The consequence is sharp: a gate of the form 'agreement with the reference below "
        "3e-08' is <b>unsatisfiable at that shape by any correct implementation</b>. Agreement "
        "with an FP32 oracle bounds nothing about accuracy. The attention tests therefore use "
        "a float64 arbiter and gate on the accumulation floor.")
    s += figure("pallas_accuracy_floor.png", 6.6,
                "Figure 2. Error against a float64 arbiter, normalized by the float32 "
                "accumulation floor. Every configuration is normal float32 behaviour.")

    s += sect(
        "6.2  Which implementation is more accurate is a property of the compiler",
        "The first version of that gate asserted the Pallas kernel was at least as accurate as "
        "the reference. It passed locally and failed in continuous integration. The reference "
        "is byte-identical on both platforms -- it is NumPy, and deterministic. Only the "
        "Pallas error moved, by 3.2x, because XLA is free to choose a different reduction "
        "order between versions.",
        "So the ordering is not a property of either implementation; it is a property of the "
        "compiler build. The gate now asserts what is stable: both sit within 3x of the "
        "float32 floor. That still fails loudly for a real defect, which misses by orders of "
        "magnitude rather than by 1.4x. A tolerance never checked on a second toolchain is an "
        "assumption, not a measurement.")

    s += sect(
        "6.3  A correctness bug that only exists on GPU",
        "The Pallas accumulators live in output blocks whose index map ignores the sequence "
        "axis, so Pallas keeps them resident and consecutive writes accumulate. This is safe "
        "only because the TPU grid executes sequentially in lexicographic order.",
        "On GPU that guarantee does not hold: Mosaic GPU partitions dimensions marked "
        "'parallel' across CUDA thread blocks, which would make the accumulator a race. "
        "Interpret mode cannot reveal this, because it executes the grid sequentially by "
        "construction. The fix declares the semantics explicitly -- (parallel, parallel, "
        "sequential) on GPU, (parallel, parallel, arbitrary) on TPU -- and was applied before "
        "any GPU time was purchased.")

    s += sect(
        "6.4  The same defect class, a different shape of fix",
        "The CUDA kernel guards the empty-page NaN with a branch on negative infinity. Pallas "
        "has no cheap per-lane branch, and the fix is arithmetic instead: substitute a finite "
        "surrogate for the running maximum whenever it is negative infinity, which drives "
        "every exponential to exactly zero -- the correct contribution of an empty page. Same "
        "hazard, same reasoning, entirely different remedy. The CUDA instinct points at a "
        "construct that should not be reached for.")

    # ---------------- 7 ----------------
    s += [PageBreak(), p("07", EYEBROW), p("Benchmark methodology", H1)]
    s += [p(
        "The three backends cannot all run on one device, so the benchmark is built to make "
        "dishonest comparison structurally impossible rather than merely discouraged.", DECK)]
    s += [table([
        ["Rule", "Rationale"],
        ["Identical seeded inputs, byte for byte, across backends",
         "removes input variation as an explanation for any gap"],
        ["Milliseconds compared only within one device",
         "two chips with different memory systems are not comparable in absolute time"],
        ["Memory-bandwidth utilization is the cross-device metric",
         "the kernel is bandwidth-bound, so MBU is what transfers between machines; "
         "where a device's peak bandwidth is not known, MBU is reported as null "
         "rather than guessed"],
        ["Milliseconds compared only within one execution mode",
         "interpret mode is a correctness vehicle, not a shipped code path; the "
         "harness records the mode and refuses to print a cross-mode ratio"],
        ["Unavailable backends recorded with a reason",
         "an absent row and a failed row must not look the same"],
        ["Shape sweep fixed before any backend is timed",
         "prevents choosing shapes after seeing results"],
    ], [2.5 * inch, 4.4 * inch])]
    s += [Spacer(1, 8)]
    s += [p("Anti-cheat assertions", H2)]
    s += [p(
        "Any benchmark graded on wall-clock is gameable, including unintentionally by its "
        "author. Three assertions run inside the harness: that the output is actually consumed "
        "(a result nothing reads can be eliminated as dead code); that the kernel is not "
        "specialized to round numbers (every configuration is re-run at sequence length plus "
        "one); and that the kernel is present in the compiled output rather than having been "
        "replaced by a library call -- a kernel that gets optimized away benchmarks "
        "beautifully.", BODY)]

    # ---------------- 8 ----------------
    s += [PageBreak(), p("08", EYEBROW), p("A defect worth reporting", H1)]
    s += [p(
        "A review of the dispatch layer found that requesting a backend which does not "
        "implement an operation returned results from a different backend, silently.", DECK)]
    s += [p(
        "ops.flash_decode(backend='triton') returned NumPy reference results, byte-identical "
        "to backend='reference'. This repository has a Triton quantizer but no Triton "
        "attention kernel, and the dispatch chain fell through instead of refusing. Any "
        "misspelled backend name behaved the same way, for either operation.", BODY)]
    s += callout(
        "Why this one matters more than it looks",
        "Benchmarking that configuration would have measured NumPy and reported it as Triton. "
        "It is precisely the failure the harness anti-cheat assertions exist to catch -- "
        "sitting one layer beneath them, where they could not see it. The lesson is that "
        "integrity checks at the measurement layer do not protect against dishonesty in the "
        "layer that selects what gets measured.", "red")
    s += [p(
        "Backends are now declared per operation and validated: a named backend is delivered "
        "or raises. Auto-detection is unchanged, so no existing caller is affected. Four "
        "further input-validation defects were fixed alongside it -- degenerate shapes raising "
        "ZeroDivisionError, a zero-iteration timing loop raising UnboundLocalError, and a "
        "misleading error that reported a missing CUDA extension when the extension was "
        "present but the input shape was unsupported. All have regression tests.", BODY)]

    # ---------------- 9 ----------------
    s += [PageBreak(), p("09", EYEBROW), p("What has NOT been established", H1)]
    s += [p(
        "Stated as plainly as the results, because a report that only lists what worked is "
        "not a technical report.", DECK)]
    s += [table([
        ["Claim", "Status"],
        ["Pallas kernels are correct",
         p("<b>Established</b> in interpret mode, on two platforms, in CI", CELL)],
        ["Pallas kernels compile on a real backend",
         p("<b>Not established.</b> They have never been lowered by Mosaic. "
           "Interpret mode enforces none of the constraints a real backend imposes.", CELL)],
        ["Pallas kernels are fast",
         p("<b>Not established.</b> No compiled Pallas timing exists on any device.", CELL)],
        ["Three-backend performance comparison",
         p("<b>Not established.</b> Requires one device that runs all three; "
           "no such device has been used.", CELL)],
        ["INT4 KV preserves model quality",
         p("<b>Simulated only.</b> The perplexity gate has not run against real "
           "model weights.", CELL)],
        ["Speedup versus a production kernel",
         p("<b>Not measured.</b> The 3.9x figure is against a serial baseline in "
           "this repository, not against FlashAttention or a serving engine.", CELL)],
        ["TPU-specific behaviour (int4 dtype, packing axis, reduction direction)",
         p("<b>Unresolved.</b> Three pre-registered hypotheses require TPU silicon.", CELL)],
    ], [2.6 * inch, 4.3 * inch])]
    s += [Spacer(1, 10)]
    s += [p(
        "The distinction between the first row and the rest is the distinction between a "
        "kernel that is right and a kernel that is good. Only the first has been shown.", BODY)]

    # ---------------- 10 ----------------
    s += [PageBreak(), p("10", EYEBROW), p("Next steps", H1)]
    s += [table([
        ["Step", "Cost", "What it resolves"],
        [p("<b>Rented H100 run</b> -- tooling is built and tested: preflight, "
           "bootstrap, ordered runbook, isolated lowering probe", CELL),
         "~$5, 1.5 h",
         "all three backends on identical silicon; real datacenter numbers; "
         "the first compilation of these kernels; the real perplexity gate"],
        [p("<b>TPU leg</b> on free Colab or Kaggle", CELL), "free",
         "the three unresolved hypotheses, which an H100 cannot touch because "
         "they are Mosaic TPU properties"],
        [p("<b>Baseline against PyTorch SDPA</b>", CELL), "free",
         "replaces a speedup measured against this repository's own first draft "
         "with a defensible external comparison"],
        [p("<b>Upstream compiler contribution</b> -- an open Triton pull request "
           "awaiting review", CELL), "free",
         "external validation, which outweighs every self-reported number here"],
    ], [2.5 * inch, 0.85 * inch, 3.55 * inch])]
    s += [Spacer(1, 10)]
    s += callout(
        "The fork to plan for",
        "The lowering probe has three outcomes and they are not equally good news. LOWERED "
        "means the port is proven. REJECTED means Mosaic refused the kernel -- keep the error "
        "text, it is the finding. MISMATCH means it compiled but disagreed with interpret "
        "mode, which is the signature of a grid-order race rather than a numerics problem, and "
        "must not be chased with tolerances.")
    s += rule(10, 8)
    s += [p("Reproducing everything in this report", H2)]
    s += [p(
        "git clone https://" + REPO + "<br/>"
        "pip install -e \".[pallas]\"<br/>"
        "pytest tests/ -q<br/>"
        "python benchmarks/bench_three_backend.py --quick<br/>"
        "python scripts/make_pallas_charts.py<br/>"
        "python scripts/make_project_pdf.py", CODE)]
    s += [p(
        "Every figure, table and number in this document is produced by the repository it "
        "describes. Nothing is illustrative.", SMALL)]

    doc.build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
