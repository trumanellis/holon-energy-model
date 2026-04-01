/// <reference types="astro/client" />

declare module 'plotly.js-basic-dist' {
  const Plotly: {
    react(
      root: HTMLElement,
      data: any[],
      layout?: any,
      config?: any,
    ): Promise<void>;
    newPlot(
      root: HTMLElement,
      data: any[],
      layout?: any,
      config?: any,
    ): Promise<void>;
    purge(root: HTMLElement): void;
  };
  export default Plotly;
}
