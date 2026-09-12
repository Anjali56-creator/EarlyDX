import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { warmUp } from "../api/client";

/* Minimal inline icons (stroke-based, inherit currentColor) — no icon library. */
const Icon = ({ d }: { d: string }) => (
  <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={d} />
  </svg>
);
const ICONS = {
  dashboard: "M3 12l9-8 9 8M5 10v10h14V10",
  assessment: "M9 5h6M9 3h6v4H9zM5 7h14v14H5zM9 12h6M9 16h4",
  results: "M4 19h16M7 16V9M12 16V5M17 16v-6",
  diseases: "M12 21s-7-4.5-7-10a7 7 0 0114 0c0 5.5-7 10-7 10zM12 8v6M9 11h6",
  models: "M4 7h16v4H4zM4 13h16v4H4zM8 9h.01M8 15h.01",
  datasets: "M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  validation: "M9 12l2 2 4-4M12 3l8 4v5c0 5-3.5 8-8 9-4.5-1-8-4-8-9V7l8-4z",
  about: "M12 22a10 10 0 100-20 10 10 0 000 20zM12 16v-4M12 8h.01",
};

const navGroups: { heading: string; links: { to: string; label: string; icon: keyof typeof ICONS; primary?: boolean }[] }[] = [
  {
    heading: "Application",
    links: [
      { to: "/dashboard", label: "Dashboard", icon: "dashboard" },
      { to: "/assessment", label: "New Assessment", icon: "assessment", primary: true },
      { to: "/results", label: "Results", icon: "results" },
    ],
  },
  {
    heading: "Research",
    links: [
      { to: "/validation", label: "Validation", icon: "validation" },
      { to: "/models", label: "Models", icon: "models" },
      { to: "/datasets", label: "Datasets", icon: "datasets" },
      { to: "/diseases", label: "Conditions", icon: "diseases" },
    ],
  },
  { heading: "Project", links: [{ to: "/about", label: "About", icon: "about" }] },
];

const PAGE_CONTEXT: Record<string, string> = {
  "/": "Overview",
  "/dashboard": "Overview",
  "/assessment": "Guided risk assessment",
  "/results": "Assessment result",
  "/validation": "Model validation · comparative analysis",
  "/models": "Model registry",
  "/datasets": "Dataset registry",
  "/diseases": "Conditions in scope",
  "/about": "About the project",
};

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

export function Layout() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();

  // wake the (free-tier, sleeping) API as soon as the shell renders
  useEffect(() => { warmUp(); }, []);
  // close the mobile drawer on navigation
  useEffect(() => { setOpen(false); }, [pathname]);

  return (
    <div className="app">
      <header className="header">
        <button type="button" className="menu-btn" aria-label={open ? "Close navigation" : "Open navigation"} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
          {open ? "✕" : "☰"}
        </button>
        <Link to="/" className="brand" aria-label="EarlyDX home">
          <span className="brand-mark" aria-hidden="true">Dx</span>
          <span>
            <span className="brand-name">EarlyDX</span>
            <span className="brand-tag">Multi-disease early risk assessment</span>
          </span>
        </Link>
        <span className="header-context">{PAGE_CONTEXT[pathname] ?? ""}</span>
        <span className="header-spacer" />
        <nav className="header-links" aria-label="Primary">
          <NavLink to="/dashboard">Dashboard</NavLink>
          <NavLink to="/validation">Validation</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
        <Link to="/assessment" className="btn-link btn-sm">New Assessment</Link>
      </header>

      <div className="shell">
        <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Application navigation">
          <div className="sidebar-inner">
          <nav className="nav">
            {navGroups.map((group) => (
              <div className="nav-group" key={group.heading}>
                <div className="nav-heading">{group.heading}</div>
                {group.links.map((l) => (
                  <NavLink
                    key={l.to}
                    to={l.to}
                    className={({ isActive }) => [isActive || (l.to === "/dashboard" && pathname === "/") ? "active" : "", l.primary ? "primary" : ""].filter(Boolean).join(" ")}
                  >
                    <Icon d={ICONS[l.icon]} />
                    {l.label}
                  </NavLink>
                ))}
              </div>
            ))}
          </nav>
          <div className="sidebar-foot">
            Research prototype — risk estimates are not medical diagnoses.
          </div>
          </div>
        </aside>

        <main className="main" id="main">
          <Outlet />
        </main>
      </div>

      <footer className="footer">
        <div className="footer-inner">
          <div>
            <Link to="/" className="brand" style={{ marginBottom: 8 }}>
              <span className="brand-mark" aria-hidden="true">Dx</span>
              <span><span className="brand-name">EarlyDX</span><span className="brand-tag">Early risk assessment — research prototype</span></span>
            </Link>
            <p>
              A final-year ML + full-stack research project: one independently trained classifier per condition,
              compared against alternatives and validated on held-out data, served with full provenance.
            </p>
          </div>
          <div>
            <h4>Navigate</h4>
            <div className="footer-nav">
              <Link to="/dashboard">Dashboard</Link>
              <Link to="/assessment">New Assessment</Link>
              <Link to="/results">Results</Link>
              <Link to="/validation">Validation</Link>
              <Link to="/models">Models</Link>
              <Link to="/datasets">Datasets</Link>
              <Link to="/diseases">Conditions</Link>
              <Link to="/about">About</Link>
            </div>
          </div>
          <div>
            <h4>Technology &amp; deployment</h4>
            <p>React + TypeScript + Vite on Vercel · FastAPI + scikit-learn on Render · SQLAlchemy (SQLite / PostgreSQL).</p>
            <p>
              {API_BASE ? <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">API documentation</a> : "API documentation at /docs"}
              {" · "}
              <a href="https://github.com/Anjali56-creator/EarlyDX" target="_blank" rel="noreferrer">Source on GitHub</a>
            </p>
          </div>
        </div>
        <div className="footer-bottom">
          <span className="footer-disclaimer">Research prototype. Risk estimates are not medical diagnoses.</span>
          <span>Not a medical device · not clinically validated · public de-identified datasets only</span>
        </div>
      </footer>
    </div>
  );
}
