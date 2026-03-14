interface StoryPanelProps {
  story: { bullets: string[] } | null | undefined;
}

export default function StoryPanel({ story }: StoryPanelProps) {
  if (!story || !story.bullets || story.bullets.length === 0) return null;

  return (
    <div className="section-card story-panel section-pad">
      <span className="section-kicker">Narrative Trace</span>
      <h2 className="section-heading" style={{ fontSize: "1.6rem" }}>
        What the run is actually saying
      </h2>
      <p className="section-copy">
        A concise operator narrative stitched from the backend metrics so the plots read like a coherent scenario, not isolated traces.
      </p>
      <ul className="mt-6">
        {story.bullets.map((bullet, i) => (
          <li key={i} className="story-bullet">
            <span className="story-index">{i + 1}</span>
            <span className="text-sm leading-7 text-slate-700">{bullet}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
