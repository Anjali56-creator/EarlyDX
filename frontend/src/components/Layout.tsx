import { NavLink, Outlet } from "react-router-dom";

const links = [
  ["/dashboard", "Dashboard"],
  ["/assessment", "Assessment"],
  ["/results", "Results"],
  ["/diseases", "Diseases"],
  ["/models", "Models"],
  ["/datasets", "Datasets"],
  ["/about", "About"],
];

export function Layout() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          EarlyDX
          <small>early risk assessment — research prototype</small>
        </div>
        <nav className="nav">
          {links.map(([to, label]) => (
            <NavLink key={to} to={to} className={({ isActive }) => (isActive ? "active" : "")}>
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
