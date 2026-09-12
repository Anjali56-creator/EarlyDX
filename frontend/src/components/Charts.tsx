import { useState } from "react";

/* Chart marks use the deeper, validated steps of the EarlyDX palette
 * (pink #C4559F · deep lavender #4B3FA6 · soft blue #6C9BDD — checked for
 * CVD and normal-vision separation). Values are always direct-labelled and a
 * table view sits next to every chart, so colour is never the only channel. */
export const SERIES_COLORS: Record<string, string> = {
  LogisticRegression: "#4B3FA6",
  RandomForestClassifier: "#C4559F",
  GradientBoostingClassifier: "#6C9BDD",
  HistGradientBoostingClassifier: "#8FA8E8",
};
const SINGLE = "#8B7BDD"; // lavender mid-step for single-series bars
const SINGLE_BEST = "#4B3FA6";

const fmt = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));

type Tip = { x: number; y: number; text: string } | null;

function useTip() {
  const [tip, setTip] = useState<Tip>(null);
  const show = (e: React.MouseEvent, text: string) => setTip({ x: e.clientX + 12, y: e.clientY + 12, text });
  const hide = () => setTip(null);
  const node = tip ? <div className="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div> : null;
  return { show, hide, node };
}

export interface BarDatum { label: string; value: number | null; sub?: string; best?: boolean; key: string }

/** Horizontal single-series bar chart on a fixed 0–1 scale (all metrics here
 * are proportions). Bars are ranked by the caller. */
export function HBarChart({ data, title, valueLabel }: { data: BarDatum[]; title: string; valueLabel: string }) {
  const { show, hide, node } = useTip();
  const rowH = 30, labelW = 170, valueW = 52, padT = 8, padB = 22;
  const width = 720, plotW = width - labelW - valueW - 12;
  const height = padT + data.length * rowH + padB;
  const x = (v: number) => (Math.max(0, Math.min(1, v)) * plotW);
  return (
    <>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}: ${data.map((d) => `${d.label} ${fmt(d.value)}`).join(", ")}`}>
        <g className="grid" transform={`translate(${labelW},0)`}>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={padT} y2={height - padB} />
              <text className="axis-label" x={x(t)} y={height - 6} textAnchor="middle">{t.toFixed(2)}</text>
            </g>
          ))}
        </g>
        {data.map((d, i) => {
          const y = padT + i * rowH;
          const w = d.value == null ? 0 : x(d.value);
          return (
            <g key={d.key} onMouseMove={(e) => show(e, `${d.label}: ${valueLabel} ${fmt(d.value)}${d.sub ? ` · ${d.sub}` : ""}`)} onMouseLeave={hide}>
              <rect x={0} y={y} width={width} height={rowH} fill="transparent" />
              <text x={labelW - 10} y={y + rowH / 2 + 4} textAnchor="end" style={{ fontWeight: d.best ? 700 : 500 }}>{d.label}</text>
              <rect
                className={`bar ${d.best ? "bar-best" : ""}`}
                x={labelW} y={y + 7} width={Math.max(w, 2)} height={rowH - 14} rx={4}
                fill={d.best ? SINGLE_BEST : SINGLE}
              />
              <text className="value-label" x={labelW + w + 6} y={y + rowH / 2 + 4}>{fmt(d.value)}{d.best ? " ★" : ""}</text>
            </g>
          );
        })}
      </svg>
      {node}
    </>
  );
}

export interface GroupDatum { metric: string; label: string; values: Record<string, number | null> }

/** Grouped vertical bars: one group per metric, one bar per candidate model. */
export function GroupedBarChart({ groups, series, highlight }: { groups: GroupDatum[]; series: string[]; highlight?: string }) {
  const { show, hide, node } = useTip();
  const width = 720, height = 240, padL = 36, padR = 8, padT = 12, padB = 40;
  const plotW = width - padL - padR, plotH = height - padT - padB;
  const groupW = plotW / groups.length;
  const barW = Math.min(26, (groupW - 16) / series.length);
  const y = (v: number) => padT + plotH - Math.max(0, Math.min(1, v)) * plotH;
  return (
    <>
      <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Candidate comparison: ${groups.map((g) => `${g.label} ${series.map((s) => `${s} ${fmt(g.values[s])}`).join(", ")}`).join("; ")}`}>
        <g className="grid">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={padL} x2={width - padR} y1={y(t)} y2={y(t)} />
              <text className="axis-label" x={padL - 6} y={y(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
            </g>
          ))}
        </g>
        {groups.map((g, gi) => {
          const gx = padL + gi * groupW + (groupW - series.length * barW - (series.length - 1) * 2) / 2;
          return (
            <g key={g.metric}>
              {series.map((s, si) => {
                const v = g.values[s];
                const bx = gx + si * (barW + 2);
                const by = v == null ? y(0) : y(v);
                const isHl = s === highlight;
                return (
                  <g key={s} onMouseMove={(e) => show(e, `${s} · ${g.label}: ${fmt(v)}`)} onMouseLeave={hide}>
                    <rect x={bx} y={padT} width={barW} height={plotH} fill="transparent" />
                    <rect className={`bar ${isHl ? "bar-best" : ""}`} x={bx} y={by} width={barW} height={Math.max(2, y(0) - by)} rx={4} fill={SERIES_COLORS[s] ?? SINGLE} />
                    {isHl && v != null && <text className="value-label" x={bx + barW / 2} y={by - 4} textAnchor="middle">{v.toFixed(2)}</text>}
                  </g>
                );
              })}
              <text className="axis-label" x={padL + gi * groupW + groupW / 2} y={height - 14} textAnchor="middle">{g.label}</text>
            </g>
          );
        })}
      </svg>
      <div className="legend" aria-label="Series">
        {series.map((s) => (
          <span key={s} style={{ ["--sw" as string]: SERIES_COLORS[s] ?? SINGLE }}>{s}{s === highlight ? " (deployed — labelled)" : ""}</span>
        ))}
      </div>
      {node}
    </>
  );
}
