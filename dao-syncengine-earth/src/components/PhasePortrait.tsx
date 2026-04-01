import { useEffect, useRef } from 'react';
import type { StitchedResult } from '../lib/event-engine';
import type { Params } from '../lib/holon-model';

interface PhasePortraitProps {
  result: StitchedResult;
  params: Params;
}

export default function PhasePortrait({ result, params }: PhasePortraitProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const Plotly = (await import('plotly.js-basic-dist')).default;
      if (cancelled || !containerRef.current) return;

      const { y, eventsLog, t } = result;
      const cData = y[3];
      const iData = y[4];

      // --- Threshold curves ---
      const phi_qi_ref = 2.5;
      const f_ref = params.f_min + (1 - params.f_min) * phi_qi_ref / (phi_qi_ref + params.Kf);
      const g_ref = params.g_min + (1 - params.g_min) * phi_qi_ref / (phi_qi_ref + params.Kg);
      const kappa1_thresh = 0.08;
      const kappa2_thresh = 0.04;

      const nPts = 200;
      const cVals: number[] = [];
      const iThresh1: number[] = [];
      const iThresh2: number[] = [];

      for (let j = 0; j < nPts; j++) {
        const cVal = 0.01 + (0.99 * j) / (nPts - 1);
        cVals.push(cVal);

        // kappa1_eff = kappa10 * C^a1 * I^b1 * f_ref = kappa1_thresh
        // => I = (kappa1_thresh / (kappa10 * C^a1 * f_ref))^(1/b1)
        const base1 = kappa1_thresh / (params.kappa10 * Math.pow(cVal, params.a1) * f_ref);
        const i1 = base1 > 0 ? Math.min(1, Math.max(0, Math.pow(base1, 1 / params.b1))) : 1;
        iThresh1.push(i1);

        const base2 = kappa2_thresh / (params.kappa20 * Math.pow(cVal, params.a2) * g_ref);
        const i2 = base2 > 0 ? Math.min(1, Math.max(0, Math.pow(base2, 1 / params.b2))) : 1;
        iThresh2.push(i2);
      }

      // Fill regions: we create filled areas
      const zeros = cVals.map(() => 0);
      const ones = cVals.map(() => 1);

      const traces: any[] = [];

      // Region 1: Dissipative (below thresh1)
      traces.push({
        x: cVals, y: iThresh1,
        type: 'scatter', mode: 'lines',
        line: { color: '#8B0000', width: 1, dash: 'dash' },
        fill: 'tozeroy',
        fillcolor: 'rgba(139, 0, 0, 0.15)',
        name: 'Dissipative',
        showlegend: true,
      });

      // Region 2: Vitality-Only (between thresh1 and thresh2)
      traces.push({
        x: cVals, y: iThresh2,
        type: 'scatter', mode: 'lines',
        line: { color: '#2E8B57', width: 1, dash: 'dash' },
        fill: 'tonexty',
        fillcolor: 'rgba(46, 139, 87, 0.12)',
        name: 'Vitality-Only',
        showlegend: true,
      });

      // Region 3: Full Circulation (above thresh2)
      traces.push({
        x: cVals, y: ones,
        type: 'scatter', mode: 'none',
        fill: 'tonexty',
        fillcolor: 'rgba(65, 105, 225, 0.15)',
        name: 'Full Circulation',
        showlegend: true,
      });

      // Trajectory line
      traces.push({
        x: Array.from(cData),
        y: Array.from(iData),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#e8ddd0', width: 2 },
        name: 'Trajectory',
        showlegend: true,
      });

      // Start marker
      traces.push({
        x: [cData[0]], y: [iData[0]],
        type: 'scatter', mode: 'markers',
        marker: { color: '#44cc88', size: 10, symbol: 'circle' },
        name: 'Start',
        showlegend: true,
      });

      // End marker
      const last = cData.length - 1;
      traces.push({
        x: [cData[last]], y: [iData[last]],
        type: 'scatter', mode: 'markers',
        marker: { color: '#ff6666', size: 10, symbol: 'square' },
        name: 'End',
        showlegend: true,
      });

      // Event markers along trajectory
      for (const ev of eventsLog) {
        // Find closest time index
        let closest = 0;
        let minDist = Infinity;
        for (let k = 0; k < t.length; k++) {
          const dist = Math.abs(t[k] - ev.time);
          if (dist < minDist) { minDist = dist; closest = k; }
        }
        const kindColor = ev.kind === 'pulse' ? '#ff4444' : ev.kind === 'ramp' ? '#44cc88' : '#4488ff';
        traces.push({
          x: [cData[closest]], y: [iData[closest]],
          type: 'scatter', mode: 'markers+text',
          marker: { color: kindColor, size: 8, symbol: 'diamond' },
          text: [ev.label],
          textposition: 'top center',
          textfont: { size: 9, color: '#999' },
          showlegend: false,
        });
      }

      const layout: any = {
        paper_bgcolor: '#0d0d14',
        plot_bgcolor: '#0d0d14',
        font: { color: '#c8c0b8', size: 11 },
        margin: { l: 50, r: 20, t: 20, b: 50 },
        xaxis: {
          title: { text: 'C (Circuit Coherence)', font: { size: 12 } },
          range: [0, 1],
          gridcolor: '#2a2a35',
          zerolinecolor: '#2a2a35',
        },
        yaxis: {
          title: { text: 'I (Imaginative Capacity)', font: { size: 12 } },
          range: [0, 1],
          gridcolor: '#2a2a35',
          zerolinecolor: '#2a2a35',
        },
        legend: {
          x: 1, y: 1, xanchor: 'right',
          bgcolor: 'rgba(13, 13, 20, 0.8)',
          bordercolor: '#2a2a35', borderwidth: 1,
          font: { size: 10 },
        },
        height: 450,
      };

      const config = { responsive: true, displayModeBar: false };
      Plotly.react(containerRef.current, traces, layout, config);
    })();

    return () => { cancelled = true; };
  }, [result, params]);

  return (
    <div style={{ marginTop: 24 }}>
      <h3 style={{ fontSize: 16, fontWeight: 600, color: '#e8ddd0', marginBottom: 8 }}>
        Phase Portrait (C\u2013I Plane)
      </h3>
      <div ref={containerRef} style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }} />
    </div>
  );
}
