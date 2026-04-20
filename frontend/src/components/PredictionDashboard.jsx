// ── Scenario accent colors (Low/Baseline/High/Aggressive) ──
const SCENARIO_COLORS = {
  Low: "#f0a040",
  Baseline: "#3d8ef8",
  High: "#2ec97a",
  Aggressive: "#b44cf0"
};
const SCENARIO_COLOR_DEFAULT = "#8b9fc4";

function scenarioColor(name) {
  for (const [key, color] of Object.entries(SCENARIO_COLORS)) {
    if (String(name || "").toLowerCase().includes(key.toLowerCase())) return color;
  }
  return SCENARIO_COLOR_DEFAULT;
}

const HIDDEN_REASON_FEATURES = new Set(["name_resolution", "model_availability"]);

function ScenarioReasonList({ reasons, limit = 3 }) {
  const seen = new Set();
  const visible = (reasons || [])
    .filter(
      (reason) =>
        typeof reason === "object" &&
        reason !== null &&
        !HIDDEN_REASON_FEATURES.has(reason.feature)
    )
    .map((reason) => {
      const explanation = String(reason?.explanation || "").trim();
      if (!explanation) return null;
      const key = explanation.toLowerCase();
      if (seen.has(key)) return null;
      seen.add(key);
      return explanation;
    })
    .filter(Boolean)
    .slice(0, limit);

  if (visible.length === 0) {
    return <p className="meta-text">Balanced matchup with no strong differentiator.</p>;
  }
  return (
    <ul className="reason-list">
      {visible.map((explanation, idx) => (
        <li key={`${explanation}-${idx}`}>{explanation}</li>
      ))}
    </ul>
  );
}

function hasStrongPlayerSignal(player) {
  if (!player || !player.name || player.name === "Unavailable") return false;
  const confidence = Number(player.confidence || 0);
  const reasons = Array.isArray(player.reason) ? player.reason : [];
  return confidence >= 0.45 && reasons.length > 0;
}

function displayRole(role) {
  const normalized = String(role || "").trim().toLowerCase();
  if (!normalized) return "player";
  if (normalized === "all-round impact") return "impact";
  return normalized.replaceAll("_", " ");
}

function PlayerGroup({ title, players, accent }) {
  const visible = (players || []).filter(hasStrongPlayerSignal).slice(0, 3);
  if (visible.length === 0) return null;

  const confClass = (conf) => {
    if (conf >= 0.8) return "conf-strong";
    if (conf >= 0.6) return "conf-mid";
    return "conf-soft";
  };

  return (
    <article className="player-card anim-up">
      <div className="player-group-label" style={{ color: accent }}>{title}</div>
      <div>
        {visible.map((player, idx) => (
          <div className="player-entry" key={`${title}-${player.name}-${idx}`}>
            <div className="player-name">{player.name}</div>
            <div className="player-role-meta">
              {player.team ? `${player.team} · ` : ""}
              {displayRole(player.role)}
            </div>
            <div className={`player-conf-value ${confClass(Number(player.confidence || 0))}`}>
              {(Number(player.confidence || 0) * 100).toFixed(0)}% confidence
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

// ── Unchanged logic helpers ──────────────────────────────────

function scenarioWeight(scenario, fallbackWeight = 0.25) {
  const prob = Number(scenario?.scenario_probability);
  if (Number.isFinite(prob) && prob > 0) return prob;
  const confidence = Number(scenario?.confidence);
  if (Number.isFinite(confidence) && confidence > 0) return confidence;
  return fallbackWeight;
}

function formatWhole(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(0) : "n/a";
}

function deriveForecastSummary(prediction) {
  const supplied = prediction?.forecast_summary || {};
  if (supplied?.favored_team) {
    const low = Number.isFinite(Number(supplied.predicted_band_low)) ? Number(supplied.predicted_band_low) : null;
    const high = Number.isFinite(Number(supplied.predicted_band_high)) ? Number(supplied.predicted_band_high) : null;
    const suppliedRange =
      typeof supplied.expected_score_range === "string" && supplied.expected_score_range.trim()
        ? supplied.expected_score_range.trim()
        : null;
    return {
      favoredTeam: supplied.favored_team,
      confidence: Number((supplied.win_probability ?? supplied.favored_team_confidence) || 0),
      bandLow: low,
      bandHigh: high,
      rangeText: suppliedRange || (low !== null && high !== null ? `${Math.round(low)}–${Math.round(high)}` : "n/a"),
      keyRisk: supplied.key_risk || "Outcome can shift with small match events.",
      riskLevel: supplied.risk_level || "Medium",
      riskExplanation: supplied.risk_explanation || "Moderate volatility with a few plausible swings.",
      finalSummary: supplied.final_summary || ""
    };
  }

  const scenarios = Array.isArray(prediction?.predictions) ? prediction.predictions : [];
  const teamA = prediction?.match?.team_a;
  const teamB = prediction?.match?.team_b;
  const norm = (s) => String(s || "").trim().toLowerCase();
  const teamWeights = {};
  if (teamA) teamWeights[teamA] = 0;
  if (teamB) teamWeights[teamB] = 0;
  scenarios.forEach((scenario) => {
    const winnerRaw = String(scenario?.winner || "");
    const weight = scenarioWeight(scenario);
    const matched = [teamA, teamB].find((t) => t && norm(winnerRaw) === norm(t));
    if (matched && Number.isFinite(weight)) {
      teamWeights[matched] = (teamWeights[matched] || 0) + weight;
    }
  });
  const favoredTeam =
    (teamWeights[teamA] || 0) >= (teamWeights[teamB] || 0) ? teamA : teamB;
  const total = (teamWeights[teamA] || 0) + (teamWeights[teamB] || 0);
  const confidence = total > 0 ? (teamWeights[favoredTeam] || 0) / total : 0.5;
  const bandLow = prediction?.uncertainty?.interval_low;
  const bandHigh = prediction?.uncertainty?.interval_high;

  return {
    favoredTeam,
    confidence,
    bandLow,
    bandHigh,
    rangeText:
      bandLow !== null && bandHigh !== null
        ? `${Math.round(Number(bandLow))}–${Math.round(Number(bandHigh))}`
        : "n/a",
    keyRisk: "Outcome can shift with batting order and early momentum.",
    riskLevel: "Medium",
    riskExplanation: "Moderate volatility with a few plausible swings.",
    finalSummary: `${favoredTeam || "Neither side"} hold a narrow edge from current scenario weights, but batting-order swing keeps uncertainty in play.`
  };
}

function evaluationInterpretation(evaluation) {
  if (!evaluation?.available) return evaluation?.message || "Actual result not available.";
  if (evaluation?.evaluation_summary) return evaluation.evaluation_summary;
  if (evaluation.winner_correct && evaluation.interval_covered) return "Model got the winner right and stayed inside the predicted range.";
  if (evaluation.winner_correct && !evaluation.interval_covered) return "Model got the winner right but missed the predicted range.";
  if (!evaluation.winner_correct && evaluation.interval_covered) return "Model missed the winner despite a close score fit.";
  return "Model missed winner and range on this match.";
}

// ── Win Probability Ring ─────────────────────────────────────

function WinProbRing({ confidence }) {
  const pct = Math.round(confidence * 100);
  const circumference = 2 * Math.PI * 36;
  const offset = circumference * (1 - confidence);
  const ringClass = confidence >= 0.7 ? "confidence-high"
    : confidence >= 0.5 ? "confidence-medium"
    : "confidence-low";
  const confidenceBand = confidence >= 0.7 ? "high"
    : confidence >= 0.5 ? "medium"
    : "low";

  return (
    <div
      className="win-prob-ring"
      data-confidence={confidenceBand}
      role="img"
      aria-label={`Win probability ${pct} percent`}
    >
      <svg width="84" height="84" viewBox="0 0 84 84" aria-hidden>
        <circle className="ring-track" cx="42" cy="42" r="36" />
        <circle
          className={`ring-fill ${ringClass}`}
          cx="42"
          cy="42"
          r="36"
          style={{ "--offset": offset }}
        />
      </svg>
      <div className="ring-label">
        <span className="ring-pct">{pct}%</span>
        <span className="ring-tiny">WIN PROB</span>
      </div>
    </div>
  );
}

// ── ScenarioOutcome (unchanged) ──────────────────────────────

function ScenarioOutcome({ isFootball, scenario, match }) {
  if (isFootball) {
    const homeGoals = Number.isFinite(Number(scenario.home_goals)) ? Number(scenario.home_goals) : null;
    const awayGoals = Number.isFinite(Number(scenario.away_goals)) ? Number(scenario.away_goals) : null;
    return (
      <div className="scenario-outcomes" style={{ marginBottom: "0.4rem" }}>
        <p className="scoreline-text">
          {match.team_a} {homeGoals ?? "–"} · {match.team_b} {awayGoals ?? "–"}
        </p>
        <p className="meta-text">Winner: {scenario.likely_result || "n/a"}</p>
      </div>
    );
  }

  const primaryWinner = scenario.winner || "n/a";
  const scorePair =
    typeof scenario.team_a_score === "number" && typeof scenario.team_b_score === "number"
      ? `${match.team_a} ${scenario.team_a_score} – ${match.team_b} ${scenario.team_b_score}`
      : null;

  return (
    <div className="scenario-outcomes" style={{ marginBottom: "0.4rem" }}>
      {scorePair && <p className="scoreline-text">{scorePair}</p>}
      <p className="meta-text">Winner: {primaryWinner}</p>
    </div>
  );
}

// ── MAIN COMPONENT ───────────────────────────────────────────

function PredictionDashboard({ prediction, modelStatus, matchEvaluation }) {
  const match = prediction?.match;
  if (!match?.team_a || !match?.team_b) {
    return (
      <div className="error-banner anim-fade" role="alert">
        Forecast response is missing match details. Try running the forecast again.
      </div>
    );
  }

  const isFootball = match.sport === "football";
  const scenarios = Array.isArray(prediction?.predictions) ? prediction.predictions : [];
  const playerBuckets = prediction?.players || {};
  const topBatsmen = Array.isArray(playerBuckets.top_batsmen) ? playerBuckets.top_batsmen : [];
  const topBowlers = Array.isArray(playerBuckets.top_bowlers) ? playerBuckets.top_bowlers : [];
  const topImpact = Array.isArray(playerBuckets.top_match_impact) ? playerBuckets.top_match_impact : [];
  const topScorers = Array.isArray(playerBuckets.top_goal_scorers) ? playerBuckets.top_goal_scorers : [];
  const topStandout = Array.isArray(playerBuckets.top_standout) ? playerBuckets.top_standout : [];
  const scenarioProbabilities = prediction?.metadata?.scenario_probabilities || [];

  const dataMode = match.state || prediction?.metadata?.data_mode || modelStatus?.data || modelStatus?.data_mode || "historical";
  const normalizedDataMode = String(dataMode || "historical").toLowerCase();
  const modeLabel =
    normalizedDataMode === "historical" ? "Historical Backtest"
    : normalizedDataMode === "fallback" ? "Historical Backtest"
    : normalizedDataMode === "hybrid" ? "Upcoming"
    : normalizedDataMode.charAt(0).toUpperCase() + normalizedDataMode.slice(1);

  const isHistoricalMatch = String(match.state || "").toLowerCase() === "historical";

  const hasPlayerSignals = (isFootball ? [topScorers, topStandout] : [topBatsmen, topBowlers, topImpact]).some(
    (group) => Array.isArray(group) && group.some(hasStrongPlayerSignal)
  );

  const forecastSummary = deriveForecastSummary(prediction);
  const performanceSummary = prediction?.performance_summary || {};
  const summaryConfidence = Number(forecastSummary.confidence || 0);
  const summaryBandLow = Number.isFinite(Number(forecastSummary.bandLow)) ? Number(forecastSummary.bandLow) : null;
  const summaryBandHigh = Number.isFinite(Number(forecastSummary.bandHigh)) ? Number(forecastSummary.bandHigh) : null;
  const riskLevel = String(forecastSummary.riskLevel || "Medium");
  const riskClass = riskLevel.toLowerCase();
  const reliability = String(performanceSummary.reliability || "Unknown");

  const primaryScenario = scenarios.reduce((best, row) => {
    if (!best) return row;
    return scenarioWeight(row) > scenarioWeight(best) ? row : best;
  }, null);

  const aFirst = primaryScenario?.team_a_first || null;
  const bFirst = primaryScenario?.team_b_first || null;
  const battingOrderFlip =
    aFirst?.winner && bFirst?.winner &&
    String(aFirst.winner).trim().toLowerCase() !== String(bFirst.winner).trim().toLowerCase();

  const evaluationCard = matchEvaluation || {
    available: false,
    message: "Run forecast on a completed match to view evaluation."
  };

  // Confidence → uncertainty band position
  const bandPos = `${Math.round(summaryConfidence * 100)}%`;

  // Chart data
  const chartData = scenarios.map((scenario, idx) => {
    const teamAScore = isFootball ? Number(scenario.home_goals || 0) : Number(scenario.team_a_score || 0);
    const teamBScore = isFootball ? Number(scenario.away_goals || 0) : Number(scenario.team_b_score || 0);
    return {
      scenario: scenario.scenario,
      team_a_score: teamAScore,
      team_b_score: teamBScore,
      label: isFootball ? scenario.scoreline : scenario.score,
      probability: scenarioProbabilities[idx] || scenario.confidence
    };
  });

  // Max score for bar width normalization
  const maxScore = Math.max(...chartData.flatMap((d) => [d.team_a_score, d.team_b_score]), 1);

  return (
    <div className={`dashboard-grid ${isHistoricalMatch ? "historical-mode" : "forecast-mode"}`}>

      {/* ── FORECAST HERO ─────────────────────────────────── */}
      <div className="forecast-hero anim-scale delay-1">
        <div className="hero-header">
          <span className="hero-context-label">
            {isHistoricalMatch ? "Forecast Snapshot" : "Primary Forecast"}
            {" · "}
            {match.team_a} vs {match.team_b}
            {" · "}
            <span className={`mode-pill ${normalizedDataMode}`} style={{ fontSize: "10px" }}>
              {modeLabel}
            </span>
          </span>
          {performanceSummary.reliability && (
            <span className="reliability-badge">{reliability}</span>
          )}
        </div>

        <div className="hero-main">
          <div>
            <div className="hero-favored-team">
              {forecastSummary.favoredTeam || "Balanced"}
            </div>
            <div className="hero-sub">
              favored to win · {match.venue} · {match.match_date || match.tournament}
            </div>
          </div>
          <WinProbRing confidence={summaryConfidence} />
        </div>

        <div className="hero-metrics">
          <div className="metric-cell">
            <div className="metric-label">Score Band</div>
            <div className="metric-value">
              {forecastSummary.rangeText || (summaryBandLow !== null && summaryBandHigh !== null
                ? `${summaryBandLow.toFixed(0)}–${summaryBandHigh.toFixed(0)}`
                : "n/a")}
            </div>
          </div>
          <div className="metric-cell">
            <div className="metric-label">Risk Level</div>
            <div className={`metric-value ${riskClass === "low" ? "ok" : riskClass === "high" ? "danger" : "warn"}`}>
              {riskLevel}
            </div>
          </div>
          <div className="metric-cell">
            <div className="metric-label">Spread</div>
            <div className="metric-value">
              {summaryBandLow !== null && summaryBandHigh !== null
                ? `±${Math.round((summaryBandHigh - summaryBandLow) / 2)}`
                : "n/a"}
            </div>
          </div>
        </div>

        {/* Uncertainty band — visual confidence indicator */}
        <div className="uncertainty-band" style={{ "--pos": bandPos }} />

        {/* Risk summary */}
        {(forecastSummary.riskExplanation || forecastSummary.keyRisk) && (
          <div className="risk-row">
            <span className="risk-label">Key risk —</span>
            {forecastSummary.riskExplanation || forecastSummary.keyRisk}
          </div>
        )}

        {/* Final summary */}
        {forecastSummary.finalSummary && (
          <p className="meta-text" style={{ marginTop: "0.6rem", lineHeight: "1.5" }}>
            {forecastSummary.finalSummary}
          </p>
        )}
      </div>

      {/* ── MODEL PERFORMANCE ─────────────────────────────── */}
      {performanceSummary.accuracy !== undefined && (
        <div className="performance-card anim-up delay-2">
          <div className="card-title">Model Reliability</div>
          <div className="performance-metrics-grid">
            {[
              { label: "Winner Acc.", value: Number.isFinite(Number(performanceSummary.accuracy))
                ? `${Math.round(Number(performanceSummary.accuracy) * 100)}%` : "n/a" },
              { label: "Avg Error", value: Number.isFinite(Number(performanceSummary.avg_error))
                ? Number(performanceSummary.avg_error).toFixed(1) : "n/a" },
              { label: "In Range", value: Number.isFinite(Number(performanceSummary.in_range_pct))
                ? `${Math.round(Number(performanceSummary.in_range_pct) * 100)}%` : "n/a" },
              { label: "Samples", value: Number.isFinite(Number(performanceSummary.samples))
                ? Number(performanceSummary.samples) : "n/a" }
            ].map(({ label, value }) => (
              <div key={label} className="metric-cell">
                <div className="metric-label">{label}</div>
                <div className="metric-value" style={{ fontSize: "14px" }}>{value}</div>
              </div>
            ))}
          </div>
          {performanceSummary.interpretation && (
            <p className="meta-text" style={{ marginTop: "0.6rem" }}>
              {performanceSummary.interpretation}
            </p>
          )}
        </div>
      )}

      {/* ── SCENARIO CARDS ────────────────────────────────── */}
      {scenarios.length > 0 && (
        <>
          <div className="section-divider"><span>Scenario Breakdown</span></div>
          <div className="scenario-list">
            {scenarios.map((scenario, idx) => {
              const color = scenarioColor(scenario.scenario);
              const conf = Number(scenario.confidence || 0);
              const weight = (scenario.scenario_probability ?? scenarioProbabilities[idx]) || 0;
              const isPrimary = scenario === primaryScenario;
              const delayClass = `delay-${Math.min(idx + 2, 6)}`;

              return (
                <article
                  key={`${scenario.scenario}-${idx}`}
                  className={`scenario-card anim-up ${delayClass} ${isPrimary ? "primary-scenario" : ""}`}
                  style={{ "--stagger": idx }}
                >
                  <div className="sc-accent-bar" style={{ background: color }} />
                  <div className="sc-label">{scenario.scenario}</div>
                  <div className="sc-score">
                    {isFootball ? scenario.scoreline : scenario.score}
                  </div>

                  {scenario.story && (
                    <p className="meta-text" style={{ marginTop: "0.3rem", fontSize: "11px" }}>
                      {scenario.story}
                    </p>
                  )}

                  <div className="sc-conf-wrap">
                    <div className="sc-conf-bar" style={{
                      width: `${conf * 100}%`,
                      background: color
                    }} />
                  </div>
                  <div className="sc-conf-pct">
                    <span>{(conf * 100).toFixed(0)}% model confidence</span>
                    <span className="sc-weight">{(weight * 100).toFixed(0)}% scenario weight</span>
                  </div>

                  <ScenarioOutcome isFootball={isFootball} scenario={scenario} match={match} />
                  <ScenarioReasonList reasons={scenario.reason || []} limit={3} />
                </article>
              );
            })}
          </div>
        </>
      )}

      {/* ── CHART ─────────────────────────────────────────── */}
      {chartData.length > 0 && (
        <div className="card chart-card anim-up delay-3">
          <div className="card-title">
            {isFootball ? "Scenario Goals by Team" : "Scenario Runs by Team"}
          </div>

          <div className="bar-chart-rows">
            {chartData.map((row) => (
              <div key={row.scenario} className="bar-chart-block">
                <div className="bar-chart-scenario">{row.scenario}</div>
                <div className="bar-row">
                  <div className="bar-team-name">{match.team_a}</div>
                  <div className="bar-track">
                    <div
                      className="bar-fill bar-fill--team-a"
                      style={{
                        width: `${(row.team_a_score / maxScore) * 100}%`
                      }}
                    >
                      {row.team_a_score > 0 ? row.team_a_score : ""}
                    </div>
                  </div>
                  <div className="bar-score-label bar-score-label--a">{row.team_a_score}</div>
                </div>
                <div className="bar-row">
                  <div className="bar-team-name">{match.team_b}</div>
                  <div className="bar-track">
                    <div
                      className="bar-fill bar-fill--team-b"
                      style={{
                        width: `${(row.team_b_score / maxScore) * 100}%`
                      }}
                    >
                      {row.team_b_score > 0 ? row.team_b_score : ""}
                    </div>
                  </div>
                  <div className="bar-score-label bar-score-label--b">{row.team_b_score}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="chart-legend">
            <span className="chart-legend-item">
              <span className="legend-swatch legend-swatch--a" />
              {match.team_a}
            </span>
            <span className="chart-legend-item">
              <span className="legend-swatch legend-swatch--b" />
              {match.team_b}
            </span>
          </div>
        </div>
      )}

      {/* ── BATTING ORDER IMPACT ──────────────────────────── */}
      {!isFootball && primaryScenario && (
        <div className="card anim-up delay-4">
          <div className="card-title">Batting-Order Impact</div>
          <div className="batting-grid">
            <div className="batting-branch">
              <div className="batting-branch-label">{match.team_a} bats first</div>
              <div className="batting-score">
                {match.team_a} {formatWhole(aFirst?.batting_score)} · {match.team_b} {formatWhole(aFirst?.chase_score)}
              </div>
              <div className="batting-winner" style={{ color: "var(--primary)" }}>
                → {aFirst?.winner || "n/a"}
              </div>
            </div>
            <div className="batting-branch">
              <div className="batting-branch-label">{match.team_b} bats first</div>
              <div className="batting-score">
                {match.team_b} {formatWhole(bFirst?.batting_score)} · {match.team_a} {formatWhole(bFirst?.chase_score)}
              </div>
              <div className="batting-winner" style={{ color: "var(--ok)" }}>
                → {bFirst?.winner || "n/a"}
              </div>
            </div>
          </div>
          <div className="batting-flip-warning">
            {battingOrderFlip
              ? "⚠ Batting order can flip the winner — toss is critical."
              : "⟳ Same winner across batting orders — toss is less critical."}
          </div>
        </div>
      )}

      {/* ── EVALUATION (historical) ───────────────────────── */}
      {isHistoricalMatch && (
        <div className="context-card evaluation-panel anim-up delay-3">
          <div className="card-title">Evaluation</div>
          {evaluationCard.available ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.6rem" }}>
              <div className="metric-cell">
                <div className="metric-label">Predicted</div>
                <div style={{ fontSize: "13px", fontWeight: 600 }}>{evaluationCard.predicted_winner || "n/a"}</div>
              </div>
              <div className="metric-cell">
                <div className="metric-label">Actual</div>
                <div style={{ fontSize: "13px", fontWeight: 600 }}>{evaluationCard.actual_winner || "n/a"}</div>
              </div>
              <div className="metric-cell">
                <div className="metric-label">Winner Correct</div>
                <div className={`metric-value ${evaluationCard.winner_correct ? "ok" : "danger"}`} style={{ fontSize: "13px" }}>
                  {typeof evaluationCard.winner_correct === "boolean" ? (evaluationCard.winner_correct ? "Yes ✓" : "No ✗") : "n/a"}
                </div>
              </div>
              <div className="metric-cell">
                <div className="metric-label">In Range</div>
                <div className={`metric-value ${evaluationCard.interval_covered ? "ok" : "warn"}`} style={{ fontSize: "13px" }}>
                  {typeof evaluationCard.interval_covered === "boolean" ? (evaluationCard.interval_covered ? "Yes ✓" : "No ✗") : "n/a"}
                </div>
              </div>
              <div className="metric-cell">
                <div className="metric-label">Score Error</div>
                <div style={{ fontSize: "13px", fontWeight: 600, fontFamily: "DM Mono, monospace" }}>
                  {Number(evaluationCard.best_match_error || 0).toFixed(2)}
                </div>
              </div>
              <div className="metric-cell">
                <div className="metric-label">Best Scenario</div>
                <div style={{ fontSize: "12px", color: "var(--ink-2)" }}>
                  {evaluationCard.best_matching_scenario || evaluationCard.winner_scenario || "n/a"}
                </div>
              </div>
            </div>
          ) : (
            <p className="meta-text">{evaluationCard.message || "Actual result not available."}</p>
          )}
          <div className="eval-verdict">{evaluationInterpretation(evaluationCard)}</div>
        </div>
      )}

      {/* ── PLAYER PANELS ─────────────────────────────────── */}
      {hasPlayerSignals && (
        <>
          <div className="section-divider"><span>Player Signals</span></div>
          <div className="player-panels">
            {isFootball ? (
              <>
                <PlayerGroup title="Top Goal Scorers" players={topScorers} accent="var(--accent)" />
                <PlayerGroup title="Top Impact Players" players={topStandout} accent="var(--primary)" />
              </>
            ) : (
              <>
                <PlayerGroup title="Top Batsmen"       players={topBatsmen} accent="var(--accent)" />
                <PlayerGroup title="Top Bowlers"       players={topBowlers} accent="var(--ok)" />
                <PlayerGroup title="Top Impact"        players={topImpact}  accent="var(--purple)" />
              </>
            )}
          </div>
        </>
      )}

    </div>
  );
}

export default PredictionDashboard;
