import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import RunsBrowser from "./pages/RunsBrowser";
import RunDetail from "./pages/RunDetail";
import SweepViewer from "./pages/SweepViewer";
import MCViewer from "./pages/MCViewer";
import DemoIndex from "./pages/DemoIndex";
import RunOnePager from "./pages/RunOnePager";
import StressTestNew from "./pages/StressTestNew";
import CaseCompare from "./pages/CaseCompare";

export default function App() {
  const enableDevRoutes = String(import.meta.env.VITE_ENABLE_DEV_ROUTES || "").toLowerCase() === "true";

  return (
    <Routes>
      {/* One-pager renders outside Layout for clean printing */}
      <Route path="/runs/:runId/onepager" element={<RunOnePager />} />

      {/* All other routes inside the Layout shell */}
      <Route
        path="*"
        element={
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/stress-test/new" replace />} />
              <Route path="/stress-test/new" element={<StressTestNew />} />
              <Route path="/stress-test/:caseId" element={<StressTestNew />} />
              <Route path="/cases/:caseId/compare" element={<CaseCompare />} />

              {enableDevRoutes && (
                <>
                  <Route path="/runs" element={<RunsBrowser />} />
                  <Route path="/runs/:runId" element={<RunDetail />} />
                  <Route path="/demo" element={<DemoIndex />} />
                  <Route path="/sweeps" element={<SweepViewer />} />
                  <Route path="/sweeps/:sweepId" element={<SweepViewer />} />
                  <Route path="/monte-carlo" element={<MCViewer />} />
                  <Route path="/monte-carlo/:batchId" element={<MCViewer />} />
                  <Route path="/mc" element={<Navigate to="/monte-carlo" replace />} />
                  <Route path="/mc/:batchId" element={<Navigate to="/monte-carlo" replace />} />
                </>
              )}

              <Route path="*" element={<Navigate to="/stress-test/new" replace />} />
            </Routes>
          </Layout>
        }
      />
    </Routes>
  );
}
