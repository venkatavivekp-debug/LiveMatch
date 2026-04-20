import { useEffect, useLayoutEffect, useMemo, useState } from "react";

import {
  fetchHeadToHead,
  fetchMatches,
  fetchMatchEvaluation,
  fetchModelStatus,
  fetchPrediction,
  fetchSports,
  fetchTournaments,
  refreshLiveData
} from "../api";
import PredictionDashboard from "./PredictionDashboard";

const DEFAULT_TOURNAMENT = {
  cricket: "IPL",
  football: "EPL"
};

const MODE_LABELS = {
  live: "Live",
  upcoming: "Upcoming",
  historical: "Historical Backtest"
};

function DashboardPage() {
  const [sports, setSports] = useState([]);
  const [tournaments, setTournaments] = useState([]);
  const [matches, setMatches] = useState([]);
  const [selectedMatchId, setSelectedMatchId] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [matchEvaluation, setMatchEvaluation] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [headToHeadHistory, setHeadToHeadHistory] = useState([]);
  const [expandedHistory, setExpandedHistory] = useState(false);

  const [loadingMatches, setLoadingMatches] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [refreshingLive, setRefreshingLive] = useState(false);
  const [error, setError] = useState("");

  const [selectedSport, setSelectedSport] = useState("cricket");
  const [filters, setFilters] = useState({
    tournament: "IPL",
    team: "",
    venue: "",
    state: "upcoming"
  });
  const [numScenarios, setNumScenarios] = useState(4);

  const selectedMatch = useMemo(
    () => matches.find((match) => match.match_id === selectedMatchId),
    [matches, selectedMatchId]
  );
  const activeMode = MODE_LABELS[filters.state] || "Upcoming";
  const modeStateClass = filters.state === "live" ? "live"
    : filters.state === "historical" ? "historical"
    : "upcoming";

  const formatApiError = (detail, fallback) => {
    const text = String(detail || "").toLowerCase();
    if (!text) return fallback;
    if (text.includes("not found")) return "Selected match is no longer available. Please refresh matches.";
    if (text.includes("timeout")) return "Live data is not available for this match right now.";
    return fallback;
  };

  // ── Data loading (unchanged logic) ──────────────────────────

  const loadTournaments = async (sport) => {
    const data = await fetchTournaments(sport);
    setTournaments(data);
    const fallbackTournament = DEFAULT_TOURNAMENT[sport] || data?.[0]?.code || "";
    const nextTournament = data.find((item) => item.code === fallbackTournament)?.code || data?.[0]?.code || "";
    setFilters((previous) => ({ ...previous, tournament: nextTournament }));
  };

  const loadMatches = async (overrides = {}) => {
    setLoadingMatches(true);
    setError("");
    try {
      const query = {
        sport: selectedSport,
        tournament: overrides.tournament ?? filters.tournament ?? undefined,
        team: filters.team || undefined,
        venue: filters.venue || undefined,
        state: filters.state || undefined
      };
      const data = await fetchMatches(query);
      setMatches(data);
      const nextSelected = data.some((item) => item.match_id === selectedMatchId)
        ? selectedMatchId
        : data[0]?.match_id || "";
      if (data.length === 0) {
        setPrediction(null);
        setMatchEvaluation(null);
        setHeadToHeadHistory([]);
      } else if (selectedMatchId && selectedMatchId !== nextSelected) {
        setPrediction(null);
        setMatchEvaluation(null);
        setHeadToHeadHistory([]);
        setError("Selected match is no longer available. Please refresh matches.");
      }
      setSelectedMatchId(nextSelected);
    } catch {
      setError("Failed to fetch matches.");
    } finally {
      setLoadingMatches(false);
    }
  };

  const loadHeadToHead = async (match) => {
    if (!match?.team_a || !match?.team_b) {
      setHeadToHeadHistory([]);
      return;
    }
    setLoadingHistory(true);
    setExpandedHistory(false);
    try {
      const data = await fetchHeadToHead({
        sport: match.sport,
        tournament: match.tournament,
        team_a: match.team_a,
        team_b: match.team_b,
        limit: 5,
        include_evaluation: true,
        k: numScenarios
      });
      setHeadToHeadHistory(Array.isArray(data) ? data : []);
    } catch {
      setHeadToHeadHistory([]);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    const initialize = async () => {
      try {
        const sportsData = await fetchSports();
        setSports(sportsData);
        try {
          const modelData = await fetchModelStatus();
          setModelStatus(modelData);
        } catch {
          setModelStatus(null);
        }
        const firstSport = sportsData?.[0]?.code || "cricket";
        setSelectedSport(firstSport);
        await loadTournaments(firstSport);
      } catch {
        setError("Failed to load sports catalog.");
      }
    };
    initialize();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedSport) return;
    loadTournaments(selectedSport).catch(() => setError("Failed to load tournaments."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSport]);

  useEffect(() => {
    if (!selectedSport || !filters.tournament) return;
    loadMatches();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSport, filters.tournament, filters.state]);

  useLayoutEffect(() => {
    setPrediction(null);
    setMatchEvaluation(null);
    setError("");
    setSelectedMatchId("");
  }, [filters.state]);

  useEffect(() => {
    if (!selectedMatch) {
      setHeadToHeadHistory([]);
      return;
    }
    loadHeadToHead(selectedMatch);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedMatch, numScenarios]);

  const handlePredict = async () => {
    if (!selectedMatch) { setError("Select a match first."); return; }
    setPredicting(true);
    setError("");
    try {
      const response = await fetchPrediction({
        k: numScenarios,
        match: {
          match_id: selectedMatch.match_id,
          sport: selectedMatch.sport,
          tournament: selectedMatch.tournament,
          team_a: selectedMatch.team_a,
          team_b: selectedMatch.team_b,
          venue: selectedMatch.venue,
          match_date: selectedMatch.match_date,
          state: selectedMatch.state
        }
      });
      setPrediction(response);
      setExpandedHistory(false);
      if (selectedMatch.match_id && String(selectedMatch.state || "").toLowerCase() === "historical") {
        try {
          const evalPayload = await fetchMatchEvaluation(selectedMatch.match_id, { k: numScenarios });
          setMatchEvaluation(evalPayload.evaluation || null);
        } catch {
          setMatchEvaluation(null);
        }
      } else {
        setMatchEvaluation(null);
      }
    } catch (predictionError) {
      const detail = predictionError?.response?.data?.detail;
      setError(
        formatApiError(
          typeof detail === "string" ? detail : "",
          "Prediction request failed."
        )
      );
      setMatchEvaluation(null);
    } finally {
      setPredicting(false);
    }
  };

  const handleRefreshLiveData = async () => {
    setRefreshingLive(true);
    setError("");
    try {
      await refreshLiveData({ sport: selectedSport, tournament: filters.tournament || undefined });
      const modelData = await fetchModelStatus();
      setModelStatus(modelData);
      await loadMatches();
    } catch {
      setError("Live refresh failed. Please try again.");
    } finally {
      setRefreshingLive(false);
    }
  };

  // ── RENDER ──────────────────────────────────────────────────
  return (
    <main className="app-shell">

      {/* ── TOPBAR ─────────────────────────────────────────── */}
      <header className="topbar anim-up">
        <div>
          <p className="eyebrow">LiveMatch Analytics</p>
          <h1>Probabilistic Match Forecasting</h1>
          <p className="meta-text">
            Multi-scenario ML forecasts · Uncertainty modeling · Player signals
          </p>
        </div>
        <div className="top-actions">
          <span className={`mode-pill ${modeStateClass}`}>
            {filters.state === "live" && <span className="live-dot" />}
            {activeMode}
          </span>
          <button
            className="ghost-btn"
            onClick={handleRefreshLiveData}
            disabled={refreshingLive}
          >
            {refreshingLive ? "Refreshing…" : "↻ Refresh"}
          </button>
        </div>
      </header>

      {/* ── FILTER BAR ─────────────────────────────────────── */}
      <div className="filter-panel anim-up delay-1">
        {/* Sport chips */}
        <span className="filter-label">Sport</span>
        {sports.map((sport) => (
          <button
            key={sport.code}
            className={`filter-chip ${selectedSport === sport.code ? "active" : ""}`}
            onClick={() => setSelectedSport(sport.code)}
          >
            {sport.name}
          </button>
        ))}

        {/* Tournament chips */}
        <span className="filter-label" style={{ marginLeft: "0.5rem" }}>Tournament</span>
        {tournaments.map((t) => (
          <button
            key={t.code}
            className={`filter-chip ${filters.tournament === t.code ? "active" : ""}`}
            onClick={() => setFilters((prev) => ({ ...prev, tournament: t.code }))}
          >
            {t.name}
          </button>
        ))}

        {/* Mode tab strip */}
        <span className="filter-label" style={{ marginLeft: "0.5rem" }}>Mode</span>
        <div className="mode-tabs">
          {[
            { value: "live", label: "Live" },
            { value: "upcoming", label: "Upcoming" },
            { value: "historical", label: "Historical" }
          ].map((mode) => (
            <button
              key={mode.value}
              type="button"
              className={`mode-tab ${filters.state === mode.value ? "active" : ""}`}
              onClick={() => setFilters((prev) => ({ ...prev, state: mode.value }))}
            >
              {mode.label}
            </button>
          ))}
        </div>

        {/* Extra filters */}
        <input
          value={filters.team}
          onChange={(e) => setFilters((prev) => ({ ...prev, team: e.target.value }))}
          placeholder="Team filter…"
          style={{ width: "130px", fontSize: "12px" }}
        />
        <input
          value={filters.venue}
          onChange={(e) => setFilters((prev) => ({ ...prev, venue: e.target.value }))}
          placeholder="Venue filter…"
          style={{ width: "130px", fontSize: "12px" }}
        />

        {/* Forecast heads */}
        <select
          value={numScenarios}
          onChange={(e) => setNumScenarios(Number(e.target.value))}
          style={{ width: "90px", fontSize: "12px" }}
        >
          {[3, 4, 5, 6].map((n) => (
            <option key={n} value={n}>K = {n}</option>
          ))}
        </select>

        <div className="filter-actions">
          <button className="ghost-btn" onClick={loadMatches} disabled={loadingMatches}>
            {loadingMatches ? "Loading…" : "Refresh Matches"}
          </button>
          <button
            className="primary-btn"
            onClick={handlePredict}
            disabled={predicting || !selectedMatch}
          >
            {predicting ? "Running…" : "Run Forecast →"}
          </button>
        </div>
      </div>

      {/* ── CONTENT LAYOUT ─────────────────────────────────── */}
      <div className="content-layout anim-up delay-2">

        {/* Match list side column */}
        <aside>
          <div className="match-list-panel">
            <div className="panel-head">
              Matches · {matches.length}
            </div>
            <div className="match-list">
              {matches.length === 0 ? (
                <p className="meta-text" style={{ padding: "0.7rem 0.75rem" }}>
                  {filters.state === "live"
                    ? "No live matches from the provider."
                    : filters.state === "upcoming"
                    ? "No upcoming matches available."
                    : "Historical matches for evaluation."}
                </p>
              ) : (
                matches.map((match, i) => (
                  <button
                    key={match.match_id}
                    className={`match-item anim-up delay-${Math.min(i + 1, 6)} ${selectedMatchId === match.match_id ? "active" : ""}`}
                    onClick={() => setSelectedMatchId(match.match_id)}
                  >
                    <p className="match-teams">
                      {match.team_a} vs {match.team_b}
                    </p>
                    <p className="match-meta">
                      {match.venue}
                      {match.match_date ? ` · ${match.match_date}` : ""}
                      <span className={`match-state-badge ${match.state === "live" ? "live" : ""}`}>
                        {match.state}
                      </span>
                    </p>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Head-to-head below match list */}
          {selectedMatch && (
            <div className="context-card anim-up delay-3" style={{ marginTop: "0.75rem" }}>
              <div className="card-title">Head-to-Head</div>
              {loadingHistory ? (
                <p className="meta-text">Loading…</p>
              ) : headToHeadHistory.length === 0 ? (
                <p className="meta-text">No recent meetings found.</p>
              ) : (
                <div>
                  {[headToHeadHistory[0], ...(expandedHistory ? headToHeadHistory.slice(1, 5) : [])].map((item) => (
                    <div key={item.match_id} className="h2h-item">
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <strong style={{ fontSize: "12px" }}>
                          {item.team_a} vs {item.team_b}
                        </strong>
                        {item.winner && (
                          <span className="h2h-winner-badge" style={{
                            background: "var(--primary-dim)",
                            color: "var(--primary)"
                          }}>
                            {item.winner}
                          </span>
                        )}
                      </div>
                      <p className="meta-text">
                        {item.match_date ? new Date(item.match_date).toLocaleDateString() : ""}
                        {item.venue ? ` · ${item.venue}` : ""}
                      </p>
                      {item.evaluation && (
                        <p className="meta-text" style={{ marginTop: "3px" }}>
                          Pred correct: {item.evaluation.winner_correct ? "✓" : "✗"}
                          {" · "}In range: {item.evaluation.in_range ? "✓" : "✗"}
                        </p>
                      )}
                    </div>
                  ))}
                  {headToHeadHistory.length > 1 && (
                    <button
                      type="button"
                      className="text-btn"
                      style={{ marginTop: "0.5rem" }}
                      onClick={() => setExpandedHistory((prev) => !prev)}
                    >
                      {expandedHistory ? "Show latest only" : "View more"}
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </aside>

        {/* Right panel — prediction results */}
        <div className="right-panel">
          {error && <div className="error-banner anim-up">{error}</div>}

          {!prediction && !predicting && (
            <div className="empty-state anim-fade">
              <p>
                {selectedMatch
                  ? `Select "${selectedMatch.team_a} vs ${selectedMatch.team_b}" and click Run Forecast to generate predictions.`
                  : "Select a match from the list to get started."}
              </p>
            </div>
          )}

          {prediction && (
            <PredictionDashboard
              prediction={prediction}
              modelStatus={modelStatus}
              matchEvaluation={matchEvaluation}
            />
          )}
        </div>
      </div>
    </main>
  );
}

export default DashboardPage;
