import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspaceDir = process.cwd();
const runtimeNodeModules = process.env.RUNTIME_NODE_MODULES;
if (!path.isAbsolute(runtimeNodeModules ?? "")) {
  throw new Error("Set RUNTIME_NODE_MODULES to the bundled dependency path");
}

const { Presentation, PresentationFile, FileBlob } = await import(
  pathToFileURL(path.join(runtimeNodeModules, "@oai/artifact-tool/dist/artifact_tool.mjs")).href,
);

const WIDTH = 1280;
const HEIGHT = 720;
const COLORS = {
  navy: "#10283d",
  blue: "#173b60",
  blueAccent: "#1c7293",
  orange: "#b04c27",
  amber: "#e7a126",
  green: "#2f7d5d",
  ink: "#233240",
  muted: "#687b8b",
  light: "#f4f6f8",
  paleGreen: "#e9f2ee",
  paleAmber: "#fff5e7",
  border: "#d8e0e6",
  white: "#ffffff",
};
const FONTS = { heading: "Cambria", body: "Calibri" };
const metricsPath = path.join(workspaceDir, "results/bnp_circuit_regime_20260917_v4/metrics.json");
const figurePath = path.join(workspaceDir, "figures/bnp_ckt5_gaussian_comparison.png");
const outputPath = path.join(workspaceDir, ".deck-build/bnp_viability_explainer_candidate.pptx");
const previewDir = path.join(workspaceDir, ".deck-build/previews");

function textbox(slide, text, x, y, w, h, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: style.typeface ?? FONTS.body,
    fontSize: style.fontSize ?? 18,
    color: style.color ?? COLORS.ink,
    bold: style.bold ?? false,
    italic: style.italic ?? false,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
    autoFit: style.autoFit ?? "shrinkText",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function panel(slide, x, y, w, h, fill = COLORS.white, line = COLORS.border) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { fill: line, width: 1 },
  });
}

function kicker(slide, text, color = COLORS.blueAccent) {
  textbox(slide, text.toUpperCase(), 68, 46, 500, 24, {
    typeface: FONTS.body,
    fontSize: 13,
    color,
    bold: true,
  });
}

function title(slide, text, y = 86, color = COLORS.blue) {
  textbox(slide, text, 68, y, 1145, 62, {
    typeface: FONTS.heading,
    fontSize: 38,
    color,
    bold: true,
  });
}

function body(slide, text, x, y, w, h, style = {}) {
  return textbox(slide, text, x, y, w, h, {
    typeface: FONTS.body,
    fontSize: style.fontSize ?? 17,
    color: style.color ?? COLORS.ink,
    bold: style.bold ?? false,
    italic: style.italic ?? false,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
  });
}

function notes(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}

function addSlideNumber(slide, number) {
  body(slide, String(number).padStart(2, "0"), 1180, 680, 32, 18, { fontSize: 11, color: COLORS.muted, alignment: "right" });
}

function makeCover(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.navy;
  textbox(slide, "BNP VIABILITY", 86, 200, 510, 26, { fontSize: 14, color: COLORS.amber, bold: true });
  textbox(slide, "Bounded noise becomes usable\nwhen the released statistic fits the bound", 86, 258, 950, 142, {
    typeface: FONTS.heading,
    fontSize: 40,
    color: COLORS.white,
    bold: true,
  });
  body(slide, "EPRI Ckt5 evidence, with a direct comparison to analytic Gaussian noise", 88, 448, 900, 34, { fontSize: 19, color: "#c4d1dc", italic: true });
  slide.shapes.add({ geometry: "rect", position: { left: 86, top: 506, width: 212, height: 3 }, fill: COLORS.amber, line: { fill: COLORS.amber, width: 0 } });
  body(slide, "Saved v4 metrics  ·  δ = 0.02  ·  two saved seeds", 88, 546, 700, 25, { fontSize: 14, color: "#9fb1c0" });
  notes(slide, `Visual reference: C:/Users/15397/Downloads/bnp_correctness_and_comparison (1).pptx\nEvidence source: ${metricsPath}`);
}

function makeMechanismSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.light;
  kicker(slide, "The mechanism");
  title(slide, "BNP offers a hard error bound, then charges for sensitivity");
  panel(slide, 68, 164, 1144, 118, COLORS.blue, COLORS.blue);
  textbox(slide, "noise ~ Uniform[-B, B]      δ = S / 2B", 110, 190, 1060, 48, { typeface: FONTS.heading, fontSize: 34, color: COLORS.white, bold: true, alignment: "center" });
  body(slide, "S is the sensitivity of the released statistic. B is the maximum perturbation.", 260, 246, 760, 24, { fontSize: 15, color: "#d4e0e8", italic: true, alignment: "center" });
  panel(slide, 68, 324, 552, 286, COLORS.paleGreen, "#c7dfd4");
  textbox(slide, "What BNP guarantees", 100, 350, 430, 28, { typeface: FONTS.heading, fontSize: 23, color: COLORS.green, bold: true });
  body(slide, "• Every released value stays within B of its true value.\n\n• The guarantee comes from the support of the distribution.\n\n• The bound holds at every draw, rather than with high probability.", 100, 402, 450, 170, { fontSize: 16, color: COLORS.ink });
  panel(slide, 660, 324, 552, 286, COLORS.white, COLORS.border);
  textbox(slide, "Why the 15-minute covariance failed", 692, 350, 470, 28, { typeface: FONTS.heading, fontSize: 23, color: COLORS.orange, bold: true });
  body(slide, "• A 96-step covariance release has high sensitivity.\n\n• At δ = 0.02, uniform noise was large relative to the small covariance entries that carry temporal structure.\n\n• The mechanism stayed correct, but the release target was too rich for the bound.", 692, 402, 460, 176, { fontSize: 16, color: COLORS.ink });
  notes(slide, `Mechanism definition and guarantee follow the existing privacy implementation and the reference deck.\nEvidence source: ${metricsPath}`);
  addSlideNumber(slide, 2);
}

function makeChangesSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.light;
  kicker(slide, "The design change");
  title(slide, "The mechanism stayed fixed. The release design changed");
  body(slide, "Viability came from matching the statistic's sensitivity to the hard bound.", 68, 148, 950, 28, { fontSize: 18, color: COLORS.muted });

  const cards = [
    [68, "1", "Feeder and records", "Use the EPRI Ckt5 load model with load-object-day records so the release represents a bounded load statistic.", COLORS.blueAccent],
    [356, "2", "Temporal dimension", "Move from 15-minute vectors (T=96) to hourly vectors (T=24). Fewer coordinates reduce sensitivity.", COLORS.green],
    [644, "3", "Clip norm", "Use C=3 for the hourly regime instead of C=6. The smaller clip norm lowers the calibrated bound.", COLORS.orange],
    [932, "4", "Predeclared screen", "Require noise/signal ≤ 1 and covariance lag-1 change ≤ 0.05. This makes usefulness measurable.", COLORS.amber],
  ];
  for (const [x, number, heading, copy, accent] of cards) {
    panel(slide, x, 220, 248, 300, COLORS.white, COLORS.border);
    textbox(slide, number, x + 24, 244, 42, 40, { typeface: FONTS.heading, fontSize: 31, color: accent, bold: true });
    textbox(slide, heading, x + 24, 300, 198, 56, { typeface: FONTS.heading, fontSize: 21, color: COLORS.blue, bold: true });
    body(slide, copy, x + 24, 372, 198, 124, { fontSize: 15, color: COLORS.ink });
  }
  panel(slide, 68, 560, 1112, 72, COLORS.paleAmber, "#f0d49e");
  body(slide, "Measured effect: BNP parameter noise / signal falls from 0.840 at T=96,C=6 to 0.074 at T=24,C=3.", 96, 582, 1040, 30, { fontSize: 18, color: COLORS.orange, bold: true, alignment: "center" });
  notes(slide, `The four changes are the actual Ckt5 v4 protocol choices. The BNP distribution, analytic Gaussian calibration, and eigenvalue floor were unchanged.\nEvidence source: ${metricsPath}`);
  addSlideNumber(slide, 3);
}

async function makeFigureSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.light;
  kicker(slide, "Measured comparison");
  title(slide, "Hourly BNP keeps the temporal signal within the acceptance screen");
  const imageBytes = await fs.readFile(figurePath);
  slide.images.add({ blob: imageBytes, contentType: "image/png", alt: "BNP and Gaussian comparison across Ckt5 regimes", fit: "contain", position: { left: 68, top: 152, width: 1144, height: 490 } });
  notes(slide, `Figure embedded from ${figurePath}. The figure reads only ${metricsPath}; no experiment was rerun.`);
  addSlideNumber(slide, 4);
}

function makeEvidenceSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.light;
  kicker(slide, "Reading the result");
  title(slide, "Hourly Ckt5 is the viable BNP regime under the stated screen");
  body(slide, "Means across two saved seeds. Covariance lag-1 is measured after the existing eigenvalue floor.", 68, 146, 1100, 28, { fontSize: 16, color: COLORS.muted });

  const table = slide.tables.add({
    rows: 5,
    columns: 5,
    left: 68,
    top: 204,
    width: 770,
    height: 300,
    columnWidths: [150, 165, 150, 165, 140],
    values: [
      ["Regime", "BNP noise / signal", "BNP covariance lag-1", "Truth covariance lag-1", "Screen"],
      ["15-min\nT=96,C=6", "0.840", "0.617", "0.901", "FAIL"],
      ["Hourly\nT=24,C=3", "0.074", "0.867", "0.901", "PASS"],
      ["Gaussian 15-min", "0.068", "0.872", "0.901", "PASS"],
      ["Gaussian hourly", "0.006", "0.898", "0.901", "PASS"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: COLORS.border, width: 1 });
  for (let c = 0; c < 5; c++) {
    table.getCell(0, c).fill = COLORS.blue;
    table.getCell(0, c).text.style = { typeface: FONTS.body, fontSize: 14, bold: true, color: COLORS.white, alignment: "center", verticalAlignment: "middle" };
  }
  for (let r = 1; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      table.getCell(r, c).text.style = { typeface: FONTS.body, fontSize: 14, color: c === 4 ? (table.getCell(r, c).value === "PASS" ? COLORS.green : COLORS.orange) : COLORS.ink, bold: c === 4, alignment: "center", verticalAlignment: "middle" };
    }
  }
  panel(slide, 884, 204, 328, 300, COLORS.paleGreen, "#c7dfd4");
  textbox(slide, "What changed in practice", 916, 234, 265, 28, { typeface: FONTS.heading, fontSize: 22, color: COLORS.green, bold: true });
  body(slide, "Hourly BNP:\n\n• 0.074 noise / signal\n• 0.867 covariance lag-1\n• -0.034 lag-1 change\n• voltage lag-1 stays near 0.60\n\nThis meets the predefined screen for the bounded hourly release.", 916, 284, 255, 180, { fontSize: 16, color: COLORS.ink });
  panel(slide, 68, 540, 1144, 76, COLORS.white, COLORS.border);
  body(slide, "Gaussian remains the lower-noise option in this comparison. BNP's distinct benefit is the hard error bound, when the release is scoped tightly enough.", 100, 562, 1080, 32, { fontSize: 17, color: COLORS.blue, bold: true, alignment: "center" });
  notes(slide, `Table values are means across the two saved seeds in ${metricsPath}. Gaussian rows use ε=1, δ=0.02; BNP rows use ε=0, δ=0.02. This is not a matched-ε comparison.`);
  addSlideNumber(slide, 5);
}

function makeLimitsSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.light;
  kicker(slide, "Scope and limits");
  title(slide, "The claim is viability for one bounded release");
  panel(slide, 68, 178, 552, 332, COLORS.paleGreen, "#c7dfd4");
  textbox(slide, "Supported by the measurements", 100, 208, 450, 30, { typeface: FONTS.heading, fontSize: 23, color: COLORS.green, bold: true });
  body(slide, "• Ckt5 hourly load-shape covariance, T=24 and C=3, passes the screen for both saved seeds.\n\n• BNP still gives an exact bounded-noise guarantee at δ=0.02.\n\n• Voltage W1 and voltage lag-1 remain close to the nonprivate release.", 100, 262, 454, 192, { fontSize: 17 });
  panel(slide, 660, 178, 552, 332, COLORS.white, COLORS.border);
  textbox(slide, "Still open", 692, 208, 450, 30, { typeface: FONTS.heading, fontSize: 23, color: COLORS.orange, bold: true });
  body(slide, "• BNP is not superior to Gaussian on parameter noise.\n\n• The 15-minute Ckt5 covariance row fails the temporal screen.\n\n• Full IEEE 123-bus covariance and voltage-trajectory releases remain harder targets.", 692, 262, 454, 192, { fontSize: 17 });
  panel(slide, 68, 548, 1144, 70, COLORS.blue, COLORS.blue);
  textbox(slide, "BNP is a good choice when the application values a hard error tolerance and can release a lower-sensitivity statistic.", 102, 568, 1076, 30, { typeface: FONTS.heading, fontSize: 21, color: COLORS.white, bold: true, alignment: "center" });
  notes(slide, `Scope statement based on the saved Ckt5 v4 measurements in ${metricsPath}. The original IEEE 123-bus measurements remain historical and were not modified.`);
  addSlideNumber(slide, 6);
}

const presentation = Presentation.create({ slideSize: { width: WIDTH, height: HEIGHT } });
makeCover(presentation);
makeMechanismSlide(presentation);
makeChangesSlide(presentation);
await makeFigureSlide(presentation);
makeEvidenceSlide(presentation);
makeLimitsSlide(presentation);

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
await (await PresentationFile.exportPptx(presentation)).save(outputPath);
for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const preview = await presentation.slides.items[index].export({ format: "png", scale: 1 });
  await fs.writeFile(path.join(previewDir, `slide-${index + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(outputPath);
