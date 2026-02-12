import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import RunsBrowser from "./pages/RunsBrowser";
import RunDetail from "./pages/RunDetail";
import SweepViewer from "./pages/SweepViewer";
import MCViewer from "./pages/MCViewer";
import DemoIndex from "./pages/DemoIndex";
import RunOnePager from "./pages/RunOnePager";

export default function App() {
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
              <Route path="/" element={<RunsBrowser />} />
              <Route path="/runs/:runId" element={<RunDetail />} />
              <Route path="/demo" element={<DemoIndex />} />
              <Route path="/sweeps" element={<SweepViewer />} />
              <Route path="/sweeps/:sweepId" element={<SweepViewer />} />
              <Route path="/mc" element={<MCViewer />} />
              <Route path="/mc/:batchId" element={<MCViewer />} />
            </Routes>
          </Layout>
        }
      />
    </Routes>
  );
}
