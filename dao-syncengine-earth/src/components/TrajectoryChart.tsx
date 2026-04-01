import { useEffect, useRef } from 'react';
import type { StitchedResult } from '../lib/event-engine';

interface TrajectoryChartProps {
  result: StitchedResult;
}

const PANELS = [
  { idx: 0, name: '\u03C6_jing', title: '\u03C6_jing \u2014 Generative Essence', color: '#B8860B' },
  { idx: 1, name: '\u03C6_qi', title: '\u03C6_qi \u2014 Vitality', color: '#2E8B57' },
  { idx: 2, name: '\u03C6_shen', title: '\u03C6_shen \u2014 Awareness', color: '#4169E1' },
  { idx: 3, name: 'C', title: 'C \u2014 Circuit Coherence', color: '#9370DB' },
  { idx: 4, name: 'I', title: 'I \u2014 Imaginative Capacity', color: '#CD853F' },
  { idx: -1, name: '\u03A6_total', title: '\u03A6_total \u2014 Total Field', color: '#e8ddd0' },
];

const EVENT_LINE_STYLES: Record<string, { dash: string; color: string }> = {
  set: { dash: 'dash', color: '#4488ff' },
  pulse: { dash: 'dot', color: '#ff4444' },
  ramp: { dash: 'dashdot', color: '#44cc88' },
};

export default function TrajectoryChart({ result }: TrajectoryChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotlyRef = useRef<any>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const Plotly = (await import('plotly.js-basic-dist')).default;
      if (cancelled || !containerRef.current) return;
      plotlyRef.current = Plotly;

      const { t, y, eventsLog } = result;

      // Compute total field
      const phiTotal = t.map((_, i) => y[0][i] + y[1][i] + y[2][i]);

      const traces: any[] = [];
      const annotations: any[] = [];
      const shapes: any[] = [];

      PANELS.forEach((panel, pi) => {
        const row = Math.floor(pi / 2) + 1;
        const col = (pi % 2) + 1;
        const axisIdx = pi === 0 ? '' : String(pi + 1);

        const yData = panel.idx >= 0 ? y[panel.idx] : phiTotal;

        traces.push({
          x: t,
          y: yData,
          type: 'scatter',
          mode: 'lines',
          name: panel.name,
          line: { color: panel.color, width: 1.5 },
          xaxis: `x${axisIdx}`,
          yaxis: `y${axisIdx}`,
          showlegend: false,
        });

        // Event markers
        for (const ev of eventsLog) {
          const style = EVENT_LINE_STYLES[ev.kind] ?? EVENT_LINE_STYLES.set;
          shapes.push({
            type: 'line',
            x0: ev.time, x1: ev.time,
            y0: 0, y1: 1,
            xref: `x${axisIdx}`,
            yref: `y${axisIdx} domain`,
            line: { color: style.color, width: 1, dash: style.dash },
            opacity: 0.5,
          });
        }

        // Panel title annotation
        annotations.push({
          text: panel.title,
          xref: `x${axisIdx} domain`,
          yref: `y${axisIdx} domain`,
          x: 0, y: 1.12,
          xanchor: 'left', yanchor: 'bottom',
          showarrow: false,
          font: { color: panel.color, size: 12 },
        });
      });

      // Build subplot layout axes
      const layout: any = {
        grid: { rows: 3, columns: 2, pattern: 'independent', xgap: 0.06, ygap: 0.08 },
        paper_bgcolor: '#0d0d14',
        plot_bgcolor: '#0d0d14',
        font: { color: '#c8c0b8', size: 11 },
        margin: { l: 50, r: 20, t: 40, b: 40 },
        showlegend: false,
        annotations,
        shapes,
        height: 600,
      };

      // Configure all axes
      for (let i = 0; i < 6; i++) {
        const suffix = i === 0 ? '' : String(i + 1);
        layout[`xaxis${suffix}`] = {
          title: i >= 4 ? { text: 'Days', font: { size: 11 } } : undefined,
          gridcolor: '#2a2a35',
          zerolinecolor: '#2a2a35',
          tickfont: { size: 10 },
        };
        layout[`yaxis${suffix}`] = {
          gridcolor: '#2a2a35',
          zerolinecolor: '#2a2a35',
          tickfont: { size: 10 },
        };
      }

      const config = { responsive: true, displayModeBar: false };

      Plotly.react(containerRef.current, traces, layout, config);
    })();

    return () => { cancelled = true; };
  }, [result]);

  return (
    <div>
      <h3 style={{ fontSize: 16, fontWeight: 600, color: '#e8ddd0', marginBottom: 8 }}>
        Trajectory Plots
      </h3>
      <div ref={containerRef} style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }} />
    </div>
  );
}
