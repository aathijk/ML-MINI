const form = document.getElementById("predict-form");
const fillSample = document.getElementById("fill-sample");
const placeholder = document.getElementById("result-placeholder");
const output = document.getElementById("result-output");
const usdValue = document.getElementById("usd-value");
const rawValue = document.getElementById("raw-value");
const errorMsg = document.getElementById("error-msg");

const SAMPLE = {
  MedInc: 3.8476,
  HouseAge: 52.0,
  AveRooms: 6.281853,
  AveBedrms: 0.981645,
  Population: 1425.0,
  AveOccup: 2.181467,
  Latitude: 37.85,
  Longitude: -122.24,
};

document.querySelectorAll(".desc[data-feature]").forEach((el) => {
  const key = el.getAttribute("data-feature");
  const labels = window.__FEATURE_LABELS__ || {};
  el.textContent = labels[key] || "";
});

function showError(text) {
  errorMsg.textContent = text;
  errorMsg.classList.remove("hidden");
  output.classList.add("hidden");
  placeholder.classList.remove("hidden");
}

function clearError() {
  errorMsg.classList.add("hidden");
  errorMsg.textContent = "";
}

fillSample.addEventListener("click", () => {
  clearError();
  for (const [name, val] of Object.entries(SAMPLE)) {
    const input = document.getElementById(`f-${name}`);
    if (input) input.value = val;
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  const fd = new FormData(form);
  const features = {};
  for (const [k, v] of fd.entries()) {
    features[k] = v;
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Request failed");
      return;
    }
    placeholder.classList.add("hidden");
    output.classList.remove("hidden");
    usdValue.textContent = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(data.prediction_usd_approx);
    rawValue.textContent = data.prediction_hundred_k.toFixed(4);
  } catch {
    showError("Network error. Is the server running?");
  }
});
