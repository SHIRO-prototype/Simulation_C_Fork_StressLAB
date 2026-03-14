import { useEffect, useMemo, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { fetchHealth } from "../lib/api";

interface LayoutProps {
  children: ReactNode;
}

const NAV_ITEMS = [
  {
    to: "/stress-test/new",
    label: "Stress Tester",
    glyph: "ST",
    copy: "Build a case, stress the sensing chain, compare outcomes.",
  },
] as const;

function pageMeta(pathname: string) {
  if (pathname.startsWith("/cases/")) {
    return {
      eyebrow: "Decision Evidence",
      title: "Baseline versus stress overlays",
      copy: "Inspect where degraded tracking bends confidence, timing, and posture.",
    };
  }

  if (pathname.startsWith("/stress-test/")) {
    return {
      eyebrow: "Case Builder",
      title: "Operator-grade conjunction stress testing",
      copy: "Turn a snapshot into a polished evidence trail, from validation through differential analysis.",
    };
  }

  return {
    eyebrow: "Simulation Workspace",
    title: "StressLAB dashboard",
    copy: "A local control room for orbital safety stress testing.",
  };
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const [version, setVersion] = useState<string>("");

  useEffect(() => {
    fetchHealth()
      .then((h) => setVersion(h.code_version))
      .catch(() => setVersion("unknown"));
  }, []);

  const meta = useMemo(() => pageMeta(location.pathname), [location.pathname]);

  return (
    <div className="app-shell">
      <aside className="chrome-sidebar">
        <div className="sidebar-inner">
          <div className="brand-row">
            <div className="brand-mark">SL</div>
            <div>
              <span className="eyebrow" style={{ color: "rgba(244, 238, 229, 0.68)" }}>
                Orbital Decision Lab
              </span>
              <h1 className="brand-title">StressLAB</h1>
              <p className="brand-copy">
                Visual stress testing for conjunction response, uncertainty drift, and operational timing pressure.
              </p>
            </div>
          </div>

          <div className="signal-grid" style={{ marginTop: 28 }}>
            <dl className="signal-tile">
              <dt>Runtime</dt>
              <dd>Local workspace</dd>
            </dl>
            <dl className="signal-tile">
              <dt>Version</dt>
              <dd>{version ? `v${version}` : "Detecting"}</dd>
            </dl>
          </div>

          <nav className="sidebar-nav">
            {NAV_ITEMS.map(({ to, label, glyph, copy }) => (
              <NavLink
                key={to}
                to={to}
                end
                className={({ isActive }) =>
                  ["nav-card", isActive ? "nav-card-active" : "nav-card-idle"].join(" ")
                }
              >
                <span className="nav-glyph">{glyph}</span>
                <span>
                  <span className="nav-label">{label}</span>
                  <span className="nav-copy">{copy}</span>
                </span>
              </NavLink>
            ))}
          </nav>

          <div className="sidebar-footer">
            <div className="section-card section-card-dark section-pad" style={{ padding: 20 }}>
              <span className="section-kicker" style={{ color: "rgba(244, 238, 229, 0.68)" }}>
                Mission Pulse
              </span>
              <h2
                className="section-heading"
                style={{ marginTop: 10, color: "#fbf5ec", fontSize: "1.5rem" }}
              >
                Case-first workflow
              </h2>
              <p className="section-copy" style={{ color: "rgba(244, 238, 229, 0.68)", maxWidth: "none" }}>
                Validate a snapshot, run the control, inject operational stress, and export the evidence pack without leaving the same workspace.
              </p>
            </div>
          </div>
        </div>
      </aside>

      <div className="chrome-main">
        <header className="topbar">
          <div>
            <span className="eyebrow">{meta.eyebrow}</span>
            <h2 className="topbar-title">{meta.title}</h2>
            <p className="topbar-copy">{meta.copy}</p>
          </div>
          <div className="topbar-meta">
            <span className="topbar-chip">
              <strong>Workspace</strong> outputs/
            </span>
            <span className="topbar-chip">
              <strong>Mode</strong> Local analysis
            </span>
          </div>
        </header>

        <main className="content-stage">{children}</main>
      </div>
    </div>
  );
}
