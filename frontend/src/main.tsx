import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./styles.css";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { Assessment } from "./pages/Assessment";
import { Results } from "./pages/Results";
import { Diseases } from "./pages/Diseases";
import { Models } from "./pages/Models";
import { Datasets } from "./pages/Datasets";
import { About } from "./pages/About";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: "dashboard", element: <Dashboard /> },
      { path: "assessment", element: <Assessment /> },
      { path: "results", element: <Results /> },
      { path: "diseases", element: <Diseases /> },
      { path: "models", element: <Models /> },
      { path: "datasets", element: <Datasets /> },
      { path: "about", element: <About /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
