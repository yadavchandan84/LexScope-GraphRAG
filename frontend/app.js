const API = "http://localhost:8000";

const queryEl = document.getElementById("q");
const askButton = document.getElementById("ask");
const resultEl = document.getElementById("result");
const statusEl = document.getElementById("status");
const bannerEl = document.getElementById("insufficientBanner");
const answerEl = document.getElementById("answer");
const citationsEl = document.getElementById("citations");

function setLoading(isLoading) {
  askButton.disabled = isLoading;
  askButton.querySelector(".button-label").textContent = isLoading ? "Searching" : "Ask";
  resultEl.setAttribute("aria-busy", String(isLoading));
}

function renderCitations(citations) {
  citationsEl.replaceChildren();
  citations.forEach((citation, index) => {
    const item = document.createElement("li");
    const number = document.createElement("span");
    const text = document.createElement("span");
    number.className = "citation-number";
    number.textContent = String(index + 1).padStart(2, "0");
    text.textContent = `${citation.doc_name} — p.${citation.page}`;
    item.append(number, text);
    citationsEl.append(item);
  });
}

async function ask() {
  const query = queryEl.value.trim();
  if (!query) {
    queryEl.focus();
    statusEl.textContent = "Enter a question to begin your search.";
    resultEl.classList.remove("hidden");
    return;
  }

  setLoading(true);
  resultEl.classList.remove("hidden");
  statusEl.textContent = "Searching sources and preparing a grounded answer…";
  bannerEl.classList.add("hidden");
  answerEl.textContent = "";
  citationsEl.replaceChildren();

  try {
    const response = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, doc_type_filter: null, top_k: 8 }),
    });

    if (!response.ok) throw new Error(`The API returned ${response.status}`);

    const data = await response.json();
    statusEl.textContent = data.insufficient_context
      ? "Answer generated with limited supporting context."
      : `Confidence: ${data.confidence}`;
    bannerEl.classList.toggle("hidden", !data.insufficient_context);
    answerEl.textContent = data.answer || "No answer was returned.";
    renderCitations(data.citations || []);
  } catch (error) {
    statusEl.textContent = `Could not reach the API. Start it on port 8000 and try again. (${error.message})`;
  } finally {
    setLoading(false);
  }
}

askButton.addEventListener("click", ask);
queryEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
});
