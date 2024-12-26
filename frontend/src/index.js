import React from "react";
import ReactDOM from "react-dom/client"; // Burayı güncelledik
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root")); // createRoot ile root oluşturduk
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
