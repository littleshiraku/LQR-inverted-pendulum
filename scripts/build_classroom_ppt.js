const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const JSZip = require("jszip");
const imageSizeModule = require("image-size");

const sizeOf = imageSizeModule.imageSize || imageSizeModule;

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "LQR与LLM_课堂汇报.pptx");

const W = 13.333;
const H = 7.5;
const FONT = "Microsoft YaHei";
const BLACK = "000000";
const WHITE = "FFFFFF";
const GRAY = "666666";
const LINE = "D9D9D9";

const assets = {
  work: "resource/work.jpg",
  stage1Figure: "figures/stage1_openloop_response.png",
  stage1Gif: "animations/stage1_openloop.gif",
  stage2Figure: "figures/stage2_lqr_vs_openloop.png",
  stage2Gif: "animations/stage2_lqr.gif",
  stage3ResourceFigure: "resource/figures/stage3_llm_vs_lqr.png",
  stage3ResourceGif: "resource/animations/stage3_llm_control.gif",
  stage3Figure: "figures/stage3_llm_vs_lqr.png",
  stage3Gif: "animations/stage3_llm_control.gif",
  models: Array.from({ length: 7 }, (_, i) => `resource/models${i + 1}.png`),
};

function abs(rel) {
  return path.join(ROOT, rel);
}

function assertAssets() {
  const files = [
    assets.work,
    assets.stage1Figure,
    assets.stage1Gif,
    assets.stage2Figure,
    assets.stage2Gif,
    assets.stage3ResourceFigure,
    assets.stage3ResourceGif,
    assets.stage3Figure,
    assets.stage3Gif,
    ...assets.models,
  ];
  const missing = files.filter((file) => !fs.existsSync(abs(file)));
  if (missing.length) {
    throw new Error(`Missing assets:\n${missing.join("\n")}`);
  }
}

function fitImage(rel, box) {
  const img = sizeOf(abs(rel));
  const ratio = img.width / img.height;
  const boxRatio = box.w / box.h;
  let w = box.w;
  let h = box.h;
  if (ratio > boxRatio) {
    h = w / ratio;
  } else {
    w = h * ratio;
  }
  return {
    path: abs(rel),
    altText: path.basename(rel),
    x: box.x + (box.w - w) / 2,
    y: box.y + (box.h - h) / 2,
    w,
    h,
  };
}

function addTitle(slide, title, subtitle) {
  slide.background = { color: WHITE };
  slide.addText(title, {
    x: 0.55,
    y: 0.3,
    w: 12.25,
    h: 0.45,
    fontFace: FONT,
    fontSize: 24,
    bold: true,
    color: BLACK,
    margin: 0,
    breakLine: false,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.55,
      y: 0.82,
      w: 12.25,
      h: 0.3,
      fontFace: FONT,
      fontSize: 12,
      color: GRAY,
      margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55,
    y: 1.2,
    w: 12.25,
    h: 0,
    line: { color: LINE, width: 0.8 },
  });
}

function addBullets(slide, items, x, y, w, fontSize = 22, lineGap = 0.52) {
  items.forEach((item, idx) => {
    slide.addText(`• ${item}`, {
      x,
      y: y + idx * lineGap,
      w,
      h: 0.34,
      fontFace: FONT,
      fontSize,
      color: BLACK,
      margin: 0,
      fit: "shrink",
    });
  });
}

function addCaption(slide, text, x, y, w) {
  slide.addText(text, {
    x,
    y,
    w,
    h: 0.24,
    fontFace: FONT,
    fontSize: 12,
    color: GRAY,
    align: "center",
    margin: 0,
  });
}

function addPicture(slide, rel, box) {
  slide.addImage(fitImage(rel, box));
}

function addVisualPairSlide(pptx, title, figureRel, gifRel) {
  const slide = pptx.addSlide();
  addTitle(slide, title, "结果截图 / 动画");
  addCaption(slide, "结果截图", 0.65, 1.33, 5.85);
  addCaption(slide, "GIF 动画", 6.83, 1.33, 5.85);
  addPicture(slide, figureRel, { x: 0.65, y: 1.65, w: 5.85, h: 5.25 });
  addPicture(slide, gifRel, { x: 6.83, y: 1.65, w: 5.85, h: 5.25 });
}

function addStackedVisualSlide(pptx, title, figureRel, gifRel, subtitle) {
  const slide = pptx.addSlide();
  addTitle(slide, title, subtitle);
  addPicture(slide, figureRel, { x: 1.1, y: 1.42, w: 11.1, h: 2.75 });
  addPicture(slide, gifRel, { x: 1.1, y: 4.38, w: 11.1, h: 2.6 });
}

function addTextSlide(pptx, title, bullets, subtitle) {
  const slide = pptx.addSlide();
  addTitle(slide, title, subtitle);
  addBullets(slide, bullets, 1.08, 1.78, 11.2, 23, 0.72);
}

function addConclusionSlide(pptx, title, text) {
  const slide = pptx.addSlide();
  addTitle(slide, title);
  slide.addText(text, {
    x: 1.1,
    y: 2.2,
    w: 11.1,
    h: 1.2,
    fontFace: FONT,
    fontSize: 23,
    bold: true,
    color: BLACK,
    margin: 0,
    fit: "shrink",
    breakLine: false,
  });
}

function makeDeck() {
  assertAssets();

  const deck = new PptxGenJS();
  global.pptx = deck;
  deck.layout = "LAYOUT_WIDE";
  deck.author = "23级自动化3班";
  deck.subject = "LQR与LLM";
  deck.title = "LQR与LLM";
  deck.company = "23级自动化3班";
  deck.lang = "zh-CN";
  deck.theme = {
    headFontFace: FONT,
    bodyFontFace: FONT,
    lang: "zh-CN",
  };

  let slide = deck.addSlide();
  slide.background = { color: WHITE };
  slide.addText("LQR与LLM", {
    x: 0.8,
    y: 2.55,
    w: 11.75,
    h: 0.85,
    fontFace: FONT,
    fontSize: 44,
    bold: true,
    color: BLACK,
    align: "center",
    margin: 0,
  });
  slide.addText("23级自动化3班", {
    x: 0.8,
    y: 3.58,
    w: 11.75,
    h: 0.42,
    fontFace: FONT,
    fontSize: 22,
    color: BLACK,
    align: "center",
    margin: 0,
  });

  slide = deck.addSlide();
  addTitle(slide, "工作分配");
  addBullets(
    slide,
    [
      "系统建模",
      "控制算法设计",
      "代码编写",
      "仿真与调试",
      "数据分析与结果对比",
      "报告制作",
      "汇报人",
    ],
    1.0,
    1.55,
    11.4,
    21,
    0.45
  );
  addPicture(slide, assets.work, { x: 2.25, y: 5.05, w: 8.85, h: 1.75 });

  addTextSlide(deck, "阶段 1：开环验证", ["先验证在无任何控制情况下，一阶倒立摆的运动状态"], "实验目的");
  addVisualPairSlide(deck, "阶段 1：开环验证", assets.stage1Figure, assets.stage1Gif);
  addConclusionSlide(deck, "阶段 1：结论与不足", "没有控制，倒立摆直接倒下。");

  addTextSlide(deck, "阶段 2：LQR 闭环", ["设计 LQR 控制器并验证非线性倒立摆可在小扰动下稳定到直立平衡点。"], "实验目的");
  assets.models.forEach((model, idx) => {
    const modelSlide = deck.addSlide();
    addTitle(modelSlide, `小车倒立摆欧拉-拉格朗日动态建模过程 ${idx + 1}/7`);
    addPicture(modelSlide, model, { x: 0.5, y: 1.42, w: 12.35, h: 5.75 });
  });
  addVisualPairSlide(deck, "阶段 2：LQR 闭环", assets.stage2Figure, assets.stage2Gif);
  addConclusionSlide(deck, "阶段 2：结论与不足", "LQR 可显著抑制开环发散，角度和位置均回到指定误差带。");

  addTextSlide(
    deck,
    "阶段 3：LLM 直接控制",
    [
      "将 Llama 3.2 1B 4-bit 指令模型经过 LoRA 专家优化。",
      "直接决策策略网络，验证状态输入到连续控制力输出的闭环接口。",
      "与阶段 2 的 LQR 基准进行对比。",
    ],
    "实验目的"
  );
  addStackedVisualSlide(deck, "阶段 3：LLM 直接控制", assets.stage3ResourceFigure, assets.stage3ResourceGif, "resource 版本");
  addStackedVisualSlide(deck, "阶段 3：LLM 直接控制", assets.stage3Figure, assets.stage3Gif, "项目根目录版本");
  addConclusionSlide(deck, "阶段 3：结论与不足", "虽然系统还不能保持稳定状态，但也表现出一定的控制能力与鲁棒性。");

  addTextSlide(
    deck,
    "扩展与总结：现在的问题",
    [
      "现在训出来的模型，只认它训练时见过的那套摆。",
      "换个质量、换个杆长，它可能就不会控了。",
      "核心不足是泛化能力还不够。",
    ],
    "后续方向"
  );
  addTextSlide(
    deck,
    "扩展与总结：LLM + LQR",
    [
      "不让 LLM 直接输出“用多大的力”。",
      "让 LLM 根据当前系统状态输出 LQR 要用到的 Q 矩阵和 R 矩阵。",
      "真正计算控制量的事情交给 LQR 完成。",
      "LLM 像一个调参者，LQR 是执行控制器。",
    ],
    "后续方向"
  );
  addTextSlide(
    deck,
    "扩展与总结：目前做到这一步",
    [
      "这个层级控制想法很有意思。",
      "但目前算力和训练样本还不够。",
      "现阶段先做到固定系统下的 LLM 模仿控制。",
      "后续再推进 LLM + LQR 的层级控制。",
    ],
    "后续方向"
  );

  return deck;
}

async function validatePptx(file) {
  const buf = fs.readFileSync(file);
  const zip = await JSZip.loadAsync(buf);
  const names = Object.keys(zip.files);
  const media = names.filter((name) => name.startsWith("ppt/media/") && !zip.files[name].dir);
  const rels = names.filter((name) => name.endsWith(".rels"));
  const relXml = await Promise.all(rels.map((name) => zip.files[name].async("string")));
  const externalRefs = relXml.filter((xml) => xml.includes('TargetMode="External"'));
  const gifMedia = media.filter((name) => name.toLowerCase().endsWith(".gif"));
  const pngMedia = media.filter((name) => name.toLowerCase().endsWith(".png"));
  const jpgMedia = media.filter((name) => /\.(jpg|jpeg)$/i.test(name));

  if (externalRefs.length) {
    throw new Error("PPTX contains external relationships.");
  }
  if (!gifMedia.length || !pngMedia.length || !jpgMedia.length) {
    throw new Error(`Unexpected media package. media=${media.join(", ")}`);
  }

  return {
    mediaCount: media.length,
    gifCount: gifMedia.length,
    pngCount: pngMedia.length,
    jpgCount: jpgMedia.length,
    externalRefCount: externalRefs.length,
  };
}

async function main() {
  const deck = makeDeck();
  await deck.writeFile({ fileName: OUT });
  const result = await validatePptx(OUT);
  console.log(JSON.stringify({ output: OUT, ...result }, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
