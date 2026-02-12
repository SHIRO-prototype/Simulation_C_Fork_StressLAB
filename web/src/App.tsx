import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import RunsBrowser from "./pages/RunsBrowser";
import RunDetail from "./pages/RunDetail";
import SweepViewer from "./pages/SweepViewer";
import MCViewer from "./pages/MCViewer";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<RunsBrowser />} />
        <Route path="/runs/:runId" element={<RunDetail />} />
        <Route path="/sweeps" element={<SweepViewer />} />
        <Route path="/sweeps/:sweepId" element={<SweepViewer />} />
        <Route path="/mc" element={<MCViewer />} />
        <Route path="/mc/:batchId" element={<MCViewer />} />
      </Routes>
    </Layout>
  );
}
