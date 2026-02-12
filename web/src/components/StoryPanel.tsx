/**
 * StoryPanel — displays the auto-generated run narrative.
 *
 * Renders the 6 bullet points produced by the backend story generator
 * in a styled card between the sanity badges and the metric cards.
 */

interface StoryPanelProps {
  story: { bullets: string[] } | null | undefined;
}

export default function StoryPanel({ story }: StoryPanelProps) {
  if (!story || !story.bullets || story.bullets.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">
        Run Narrative
      </h2>
      <ul className="space-y-2 text-sm text-gray-800 leading-relaxed">
        {story.bullets.map((bullet, i) => (
          <li key={i} className="flex gap-2">
            <span className="mt-0.5 flex-shrink-0 text-blue-500 font-bold">
              {i + 1}.
            </span>
            <span>{bullet}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
