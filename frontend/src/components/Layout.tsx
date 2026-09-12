import { useEffect } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { warmUp } from "../api/client";

const navGroups: { heading: string; links: [string, string, boolean?][] }[] = [
  {
    heading: "Main",
    links: [
      ["/dashboard", "Dashboard"],
      ["/assessment", "New Assessment", true],
      ["/results", "Results"],
    ],
  },
  {
    heading: "Research",
    links: [
      ["/diseases", "Diseases"],
      ["/models", "Models"],
      ["/datasets", "Datasets"],
      ["/validation", "Validation"],
    ],
  },
  {
    heading: "Info",
    links: [["/about", "About"]],
  },
];

export function Layout() {
  // wake the (free-tier, sleeping) API as soon as the shell renders
  useEffect(() => { warmUp(); }, []);
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          EarlyDX
          <small>early risk assessment — research prototype</small>
        </div>
        <nav className="nav">
          {navGroups.map((group) => (
            <div className="nav-group" key={group.heading}>
              <div className="nav-heading">{group.heading}</div>
              {group.links.map(([to, label, primary]) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) => [isActive ? "active" : "", primary ? "primary" : ""].filter(Boolean).join(" ")}
                >
                  {label}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
