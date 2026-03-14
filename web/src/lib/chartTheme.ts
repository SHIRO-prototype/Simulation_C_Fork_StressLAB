export const plotConfig: Partial<Plotly.Config> = {
  responsive: true,
  displayModeBar: false,
  displaylogo: false,
};

export const plotColors = {
  baseline: "#6ee7ff",
  stress: "#ff8f6b",
  amber: "#f5b85c",
  mint: "#78f0c7",
  coral: "#ff7c7c",
  plum: "#c3a6ff",
  grid: "rgba(162, 186, 205, 0.14)",
  axis: "rgba(223, 232, 239, 0.22)",
  ink: "#d9e7f2",
  paper: "rgba(0, 0, 0, 0)",
  plot: "#081521",
};

type AxisOptions = Partial<Plotly.LayoutAxis> & {
  title?: string;
};

interface LayoutOptions {
  title: string;
  height: number;
  xAxis: AxisOptions;
  yAxis: AxisOptions;
  yAxis2?: AxisOptions;
  margin?: Partial<Plotly.Margin>;
  legend?: Partial<Plotly.Legend>;
  annotations?: Plotly.Layout["annotations"];
  shapes?: Plotly.Layout["shapes"];
}

function createAxis({ title, ...rest }: AxisOptions): Partial<Plotly.LayoutAxis> {
  return {
    automargin: true,
    zeroline: false,
    linecolor: plotColors.axis,
    tickcolor: plotColors.axis,
    showline: false,
    showgrid: true,
    gridcolor: plotColors.grid,
    griddash: "dot",
    tickfont: {
      family: '"IBM Plex Sans", sans-serif',
      size: 11,
      color: "rgba(217, 231, 242, 0.72)",
    },
    title: title
      ? {
          text: title,
          font: {
            family: '"IBM Plex Sans", sans-serif',
            size: 12,
            color: "rgba(217, 231, 242, 0.8)",
          },
        }
      : undefined,
    ...rest,
  };
}

export function createPlotLayout({
  title,
  height,
  xAxis,
  yAxis,
  yAxis2,
  margin,
  legend,
  annotations,
  shapes,
}: LayoutOptions): Partial<Plotly.Layout> {
  return {
    autosize: true,
    paper_bgcolor: plotColors.paper,
    plot_bgcolor: plotColors.plot,
    font: {
      family: '"IBM Plex Sans", sans-serif',
      color: plotColors.ink,
    },
    title: {
      text: title,
      x: 0.02,
      xanchor: "left",
      font: {
        family: '"Fraunces", serif',
        size: 18,
        color: "#f4efe4",
      },
    },
    hoverlabel: {
      bgcolor: "#102131",
      bordercolor: "rgba(223, 232, 239, 0.18)",
      font: {
        family: '"IBM Plex Sans", sans-serif',
        size: 12,
        color: "#f4efe4",
      },
    },
    height,
    margin: {
      t: 64,
      b: 56,
      l: 64,
      r: 24,
      ...margin,
    },
    legend: {
      orientation: "h",
      x: 0,
      y: -0.2,
      font: {
        family: '"IBM Plex Sans", sans-serif',
        size: 11,
        color: "rgba(217, 231, 242, 0.78)",
      },
      ...legend,
    },
    xaxis: createAxis(xAxis),
    yaxis: createAxis(yAxis),
    yaxis2: yAxis2 ? createAxis(yAxis2) : undefined,
    annotations,
    shapes,
  };
}
