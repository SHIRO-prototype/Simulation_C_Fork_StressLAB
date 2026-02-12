import { useEffect, useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { fetchHealth } from "../lib/api";

interface LayoutProps {
  children: ReactNode;
}

const NAV_ITEMS = [
  { to: "/", label: "Runs" },
  { to: "/sweeps", label: "Sweeps" },
  { to: "/mc", label: "Monte Carlo" },
] as const;

export default function Layout({ children }: LayoutProps) {
  const [version, setVersion] = useState<string>("");

  useEffect(() => {
    fetchHealth()
      .then((h) => setVersion(h.code_version))
      .catch(() => setVersion("unknown"));
  }, []);

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="flex flex-col w-60 shrink-0 bg-gray-900 text-white">
        {/* Branding */}
        <div className="flex items-center gap-2 px-5 py-5">
          <span className="text-xl font-bold tracking-tight">StressLAB</span>
          {version && (
            <span className="text-xs text-gray-400">v{version}</span>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex flex-col gap-1 px-3 mt-2">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                [
                  "block rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-gray-700 text-white"
                    : "text-gray-300 hover:bg-gray-800 hover:text-white",
                ].join(" ")
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Spacer pushes footer down */}
        <div className="flex-1" />

        <div className="px-5 py-4 text-xs text-gray-500">
          &copy; StressLAB
        </div>
      </aside>

      {/* Main area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center h-14 shrink-0 border-b border-gray-200 bg-white px-6">
          <h1 className="text-lg font-semibold text-gray-800">Dashboard</h1>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto bg-gray-50 p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
