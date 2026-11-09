const { useState, useEffect } = React;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [viewMode, setViewMode] = useState("overlay"); // "scan", "overlay", "mask"
  const [doctorName, setDoctorName] = useState("Dr. Sarah Chen, MD");
  const [doctorNotes, setDoctorNotes] = useState("");
  const [auditLogs, setAuditLogs] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [notification, setNotification] = useState(null);

  const API_BASE = "http://localhost:8000";

  useEffect(() => {
    fetchStatus();
    fetchAuditLogs();
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
    } catch (e) {
      console.warn("Backend not running or offline:", e);
    }
  };

  const fetchAuditLogs = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/audit-logs`);
      if (res.ok) {
        const data = await res.json();
        setAuditLogs(data);
      }
    } catch (e) {
      console.warn("Failed to fetch logs:", e);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setNotification(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Inference server responded with ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setNotification({ type: "success", message: "Scan successfully analyzed by MONAI & Grad-CAM pipeline." });
    } catch (e) {
      setNotification({ type: "error", message: `Analysis failed: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };

  const handleDecision = async (decision) => {
    if (!result) return;
    setLoading(true);

    try {
      const payload = {
        study_id: result.study_id || `STD_${Date.now()}`,
        decision: decision,
        radiologist_name: doctorName,
        notes: doctorNotes,
        override_prediction: decision === "Approved" ? result.prediction : (result.prediction === "malignant" ? "benign" : "malignant"),
      };

      const res = await fetch(`${API_BASE}/api/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: json_payload => JSON.stringify(payload),
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        setNotification({ type: "success", message: `Study ${decision} successfully logged to clinical audit trail!` });
        setDoctorNotes("");
        fetchAuditLogs();
      } else {
        throw new Error("Failed to record review decision.");
      }
    } catch (e) {
      setNotification({ type: "error", message: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Top Clinical Safety Banner */}
      <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 flex items-center justify-between text-amber-300 text-xs font-semibold">
        <div className="flex items-center gap-2">
          <i className="fa-solid fa-triangle-exclamation text-amber-400"></i>
          <span>FOR INVESTIGATIONAL & EDUCATIONAL USE ONLY — NOT FOR PRIMARY CLINICAL DIAGNOSIS</span>
        </div>
        <div className="flex items-center gap-4">
          <span>Orthanc PACS: {systemStatus?.orthanc_connected ? <span className="text-emerald-400">Connected</span> : <span className="text-slate-400">Standby (Port 8042)</span>}</span>
          <span>Engine: <span className="text-blue-400">MONAI UNet + Grad-CAM</span></span>
        </div>
      </div>

      {/* Main Navbar */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-blue-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <i className="fa-solid fa-heart-pulse text-white text-lg"></i>
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight">AI-Assisted Medical Imaging System</h1>
            <p className="text-xs text-slate-400">Deep Learning Diagnostic & Explainability Platform</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700 text-xs">
            <i className="fa-solid fa-user-doctor text-blue-400"></i>
            <span className="text-slate-300 font-medium">{doctorName}</span>
          </div>
        </div>
      </header>

      {/* Main Content Workspace */}
      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-7xl mx-auto w-full">
        {/* Left Column: Upload, Metadata, and Image Viewer */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          {/* Scan Controls / Ingestion Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-sm">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold text-slate-200">Study Ingestion & Analysis</h2>
                <p className="text-xs text-slate-400">Upload dermoscopy scan, X-Ray, or DICOM study file</p>
              </div>

              <div className="flex items-center gap-3 w-full sm:w-auto">
                <input
                  type="file"
                  id="fileUpload"
                  onChange={handleFileChange}
                  accept=".jpg,.jpeg,.png,.dcm"
                  className="hidden"
                />
                <label
                  htmlFor="fileUpload"
                  className="cursor-pointer px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg border border-slate-700 transition flex items-center gap-2"
                >
                  <i className="fa-solid fa-file-medical"></i> Select Scan
                </label>

                <button
                  onClick={runAnalysis}
                  disabled={!selectedFile || loading}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white text-xs font-semibold rounded-lg shadow-md shadow-blue-600/20 transition flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <i className="fa-solid fa-circle-notch fa-spin"></i> Processing...
                    </>
                  ) : (
                    <>
                      <i className="fa-solid fa-brain"></i> Analyze Scan
                    </>
                  )}
                </button>
              </div>
            </div>

            {selectedFile && (
              <div className="mt-3 text-xs text-slate-400 flex items-center gap-2">
                <i className="fa-regular fa-file-lines text-slate-500"></i>
                <span className="font-mono text-slate-300">{selectedFile.name}</span>
                <span>({(selectedFile.size / 1024).toFixed(1)} KB)</span>
              </div>
            )}
          </div>

          {/* Interactive Viewer Canvas */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 flex flex-col flex-1">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-4">
              <div className="flex items-center gap-2">
                <i className="fa-solid fa-microscope text-blue-400"></i>
                <h3 className="text-sm font-semibold text-slate-200">Radiological Viewer Canvas</h3>
              </div>

              {result && (
                <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                  <button
                    onClick={() => setViewMode("scan")}
                    className={`px-3 py-1 rounded-md transition ${viewMode === "scan" ? "bg-slate-800 text-white font-semibold" : "text-slate-400 hover:text-slate-200"}`}
                  >
                    Raw Scan
                  </button>
                  <button
                    onClick={() => setViewMode("overlay")}
                    className={`px-3 py-1 rounded-md transition ${viewMode === "overlay" ? "bg-blue-600 text-white font-semibold" : "text-slate-400 hover:text-slate-200"}`}
                  >
                    Grad-CAM Overlay
                  </button>
                  <button
                    onClick={() => setViewMode("mask")}
                    className={`px-3 py-1 rounded-md transition ${viewMode === "mask" ? "bg-slate-800 text-white font-semibold" : "text-slate-400 hover:text-slate-200"}`}
                  >
                    MONAI Mask
                  </button>
                </div>
              )}
            </div>

            <div className="flex-1 min-h-[360px] bg-slate-950 rounded-lg border border-slate-800/60 flex items-center justify-center overflow-hidden relative">
              {result ? (
                <div className="relative max-h-[460px] flex items-center justify-center">
                  <img
                    src={
                      viewMode === "scan"
                        ? result.visualizations.scan_image
                        : viewMode === "mask"
                        ? result.visualizations.mask_image
                        : result.visualizations.gradcam_overlay
                    }
                    alt="Scan View"
                    className="max-h-[460px] object-contain rounded shadow-lg"
                  />
                  {viewMode === "overlay" && (
                    <div className="absolute bottom-2 left-2 bg-slate-950/80 backdrop-blur px-2.5 py-1 rounded text-[10px] text-slate-300 border border-slate-700/50">
                      Heatmap: Grad-CAM Saliency | Contour: MONAI UNet
                    </div>
                  )}
                </div>
              ) : previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Pre-analysis Preview"
                  className="max-h-[380px] object-contain rounded"
                />
              ) : (
                <div className="text-center p-8 text-slate-500">
                  <i className="fa-solid fa-cloud-arrow-up text-4xl mb-3 text-slate-600"></i>
                  <p className="text-sm">No medical image loaded.</p>
                  <p className="text-xs text-slate-600 mt-1">Select an ISIC dermoscopy scan or DICOM file above to begin.</p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Right Column: AI Analysis, Diagnostic Metrics & Clinician Action */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          {/* AI Diagnostic Summary Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <i className="fa-solid fa-chart-pie text-blue-400"></i> AI Inference Analysis
            </h3>

            {result ? (
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-xs text-slate-400 mb-1">Algorithmic Assessment</div>
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-xl font-bold uppercase tracking-tight ${
                        result.prediction === "malignant" ? "text-rose-400" : "text-emerald-400"
                      }`}
                    >
                      {result.prediction}
                    </span>
                    <span className="text-sm font-semibold text-slate-200">
                      {(result.confidence * 100).toFixed(1)}% Confidence
                    </span>
                  </div>

                  {/* Confidence bar */}
                  <div className="w-full bg-slate-800 h-2 rounded-full mt-3 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        result.prediction === "malignant" ? "bg-rose-500" : "bg-emerald-500"
                      }`}
                      style={{ width: `${result.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Quantitative Class Distribution */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-slate-950/70 p-2.5 rounded border border-slate-800/80">
                    <span className="text-slate-400 block">P(Benign)</span>
                    <span className="text-sm font-bold text-slate-200">
                      {(result.probabilities.benign * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-slate-950/70 p-2.5 rounded border border-slate-800/80">
                    <span className="text-slate-400 block">P(Malignant)</span>
                    <span className="text-sm font-bold text-slate-200">
                      {(result.probabilities.malignant * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Lesion Geometry Metrics */}
                <div className="p-3 bg-slate-950/70 rounded border border-slate-800/80 text-xs space-y-1.5">
                  <div className="font-semibold text-slate-300 flex items-center justify-between">
                    <span>Segmentation Extent</span>
                    <span className="text-blue-400">{result.lesion_metrics.area_percentage}% of Frame</span>
                  </div>
                  <div className="text-slate-400 text-[11px]">
                    Lesion Area: <span className="font-mono text-slate-300">{result.lesion_metrics.area_pixels} px</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-slate-500 text-xs">
                Run analysis on a scan to see real-time classification, MONAI segmentation, and Grad-CAM confidence.
              </div>
            )}
          </div>

          {/* Clinician Review & Decision Audit Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-sm flex flex-col">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
              <i className="fa-solid fa-signature text-purple-400"></i> Clinician Sign-off
            </h3>

            <div className="space-y-3">
              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Radiologist Reviewer</label>
                <input
                  type="text"
                  value={doctorName}
                  onChange={(e) => setDoctorName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Diagnostic Observations / Notes</label>
                <textarea
                  value={doctorNotes}
                  onChange={(e) => setDoctorNotes(e.target.value)}
                  placeholder="Enter diagnostic confirmation or clinical remarks..."
                  rows="3"
                  className="w-full bg-slate-950 border border-slate-800 rounded p-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
                ></textarea>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <button
                  onClick={() => handleDecision("Approved")}
                  disabled={!result || loading}
                  className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white text-xs font-bold rounded-lg shadow-sm transition flex items-center justify-center gap-2"
                >
                  <i className="fa-solid fa-check"></i> Approve
                </button>
                <button
                  onClick={() => handleDecision("Rejected")}
                  disabled={!result || loading}
                  className="px-4 py-2.5 bg-rose-600 hover:bg-rose-500 disabled:opacity-40 text-white text-xs font-bold rounded-lg shadow-sm transition flex items-center justify-center gap-2"
                >
                  <i className="fa-solid fa-xmark"></i> Reject
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Audit Trail & Compliance Log (Full Width) */}
        <section className="lg:col-span-12 bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
              <i className="fa-solid fa-list-check text-blue-400"></i> Regulatory Audit Trail & Review Log
            </h3>
            <span className="text-xs text-slate-500 font-mono">{auditLogs.length} Records</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Timestamp (UTC)</th>
                  <th className="py-2.5 px-3">Study ID</th>
                  <th className="py-2.5 px-3">Decision</th>
                  <th className="py-2.5 px-3">Reviewer</th>
                  <th className="py-2.5 px-3">Override / Target</th>
                  <th className="py-2.5 px-3">Notes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {auditLogs.length > 0 ? (
                  auditLogs.slice(-5).reverse().map((log, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/30 transition">
                      <td className="py-2.5 px-3 font-mono text-[11px] text-slate-400">{log.timestamp}</td>
                      <td className="py-2.5 px-3 font-mono text-blue-400">{log.study_id}</td>
                      <td className="py-2.5 px-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            log.decision === "Approved"
                              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                              : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                          }`}
                        >
                          {log.decision}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-medium">{log.radiologist_name}</td>
                      <td className="py-2.5 px-3 capitalize text-slate-400">{log.override_prediction || "-"}</td>
                      <td className="py-2.5 px-3 text-slate-400 max-w-xs truncate">{log.notes || "No notes"}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="6" className="text-center py-6 text-slate-500">
                      No review actions recorded in this audit cycle.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      {/* Global Notification Toast */}
      {notification && (
        <div
          className={`fixed bottom-5 right-5 max-w-md px-4 py-3 rounded-lg shadow-xl text-xs flex items-center justify-between gap-3 border ${
            notification.type === "error"
              ? "bg-rose-950/90 border-rose-800 text-rose-200"
              : "bg-emerald-950/90 border-emerald-800 text-emerald-200"
          }`}
        >
          <span>{notification.message}</span>
          <button onClick={() => setNotification(null)} className="text-slate-400 hover:text-white">
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>
      )}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
