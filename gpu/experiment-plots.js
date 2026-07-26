const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const HISTORY_PARAMETERS = new Set([
  "historyTokenBudget",
  "relevanceHistoryBudget",
  "distractionHistoryBudget",
]);

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NAMESPACE, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

function createPlot(title, points, valueKey) {
  const figure = document.createElement("figure");
  figure.className = "plot-card";
  const caption = document.createElement("figcaption");
  caption.textContent = title;
  const svg = createSvgElement("svg", {
    viewBox: "0 0 360 220",
    role: "img",
    "aria-label": title,
  });
  const bounds = { left: 48, right: 344, top: 16, bottom: 184 };
  const values = points.map((point) => point[valueKey]);
  const minX = Math.min(...points.map((point) => point.historyTokens));
  const maxX = Math.max(...points.map((point) => point.historyTokens));
  let minY = Math.min(...values);
  let maxY = Math.max(...values);

  if (valueKey === "qualityPassRate") {
    minY = 0;
    maxY = 100;
  } else {
    const padding = Math.max((maxY - minY) * 0.12, maxY * 0.04, 1);
    minY = Math.max(0, minY - padding);
    maxY += padding;
  }
  if (maxY === minY) maxY = minY + 1;

  const x = (value) => bounds.left +
    (value - minX) / (maxX - minX || 1) * (bounds.right - bounds.left);
  const y = (value) => bounds.bottom -
    (value - minY) / (maxY - minY) * (bounds.bottom - bounds.top);

  svg.append(
    createSvgElement("line", {
      x1: bounds.left,
      y1: bounds.top,
      x2: bounds.left,
      y2: bounds.bottom,
      class: "plot-axis",
    }),
    createSvgElement("line", {
      x1: bounds.left,
      y1: bounds.bottom,
      x2: bounds.right,
      y2: bounds.bottom,
      class: "plot-axis",
    }),
    createSvgElement("polyline", {
      points: points.map((point) => `${x(point.historyTokens)},${y(point[valueKey])}`).join(" "),
      class: "plot-line",
    })
  );

  for (const point of points) {
    const pointX = x(point.historyTokens);
    const pointY = y(point[valueKey]);
    const xLabel = createSvgElement("text", {
      x: pointX,
      y: bounds.bottom + 17,
      "text-anchor": "middle",
      class: "plot-label",
    });
    const valueLabel = createSvgElement("text", {
      x: pointX,
      y: Math.max(bounds.top + 9, pointY - 8),
      "text-anchor": "middle",
      class: "plot-label",
    });
    xLabel.textContent = String(point.historyTokens);
    valueLabel.textContent = point[valueKey].toFixed(0);
    svg.append(
      createSvgElement("circle", {
        cx: pointX,
        cy: pointY,
        r: 4,
        class: "plot-point",
      }),
      xLabel,
      valueLabel
    );
  }

  const axisLabel = createSvgElement("text", {
    x: (bounds.left + bounds.right) / 2,
    y: 214,
    "text-anchor": "middle",
    class: "plot-label",
  });
  axisLabel.textContent = "configured history tokens";
  svg.appendChild(axisLabel);
  figure.append(caption, svg);
  return figure;
}

export function renderDoseResponsePlots(section, grid, analysis, parameterId) {
  grid.replaceChildren();
  if (!HISTORY_PARAMETERS.has(parameterId)) {
    section.hidden = true;
    return;
  }

  const points = analysis.summaries
    .filter((summary) => Number.isFinite(Number(summary.experimentValue)))
    .map((summary) => ({
      historyTokens: Number(summary.experimentValue),
      medianPromptTokens: summary.medianPromptTokens,
      medianTtftMs: summary.medianTtftMs,
      medianWallMs: summary.medianWallMs,
      qualityPassRate: summary.qualityPassRate * 100,
    }))
    .sort((a, b) => a.historyTokens - b.historyTokens);

  if (points.length < 2) {
    section.hidden = true;
    return;
  }

  grid.append(
    createPlot("Prompt tokens", points, "medianPromptTokens"),
    createPlot("TTFT (ms)", points, "medianTtftMs"),
    createPlot("Wall time (ms)", points, "medianWallMs"),
    createPlot("Strict quality pass rate (%)", points, "qualityPassRate")
  );
  section.hidden = false;
}
