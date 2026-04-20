import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000
});

export const fetchSports = async () => {
  const response = await api.get("/sports");
  return response.data;
};

export const fetchTournaments = async (sport = "cricket") => {
  const response = await api.get("/tournaments", { params: { sport } });
  return response.data;
};

export const fetchMatches = async (filters = {}) => {
  const response = await api.get("/matches", { params: filters });
  return response.data;
};

export const fetchHeadToHead = async (params = {}) => {
  const response = await api.get("/matches/head-to-head", { params });
  return response.data;
};

export const fetchPrediction = async (payload) => {
  const response = await api.post("/predict", payload);
  return response.data;
};

export const fetchBatchPrediction = async (payload) => {
  const response = await api.post("/predict/batch", payload);
  return response.data;
};

export const fetchForecastScenarios = async (params = {}) => {
  const response = await api.get("/forecast/scenarios", { params });
  return response.data;
};

export const fetchForecastUncertainty = async (params = {}) => {
  const response = await api.get("/forecast/uncertainty", { params });
  return response.data;
};

export const fetchMatchEvaluation = async (matchId, params = {}) => {
  const response = await api.get(`/matches/${matchId}/evaluation`, { params });
  return response.data;
};

export const fetchLiveInsights = async (params = {}) => {
  const response = await api.get("/insights/live", { params });
  return response.data;
};

export const fetchTopPlayers = async (params = {}) => {
  const response = await api.get("/players/top", { params });
  return response.data;
};

export const fetchModelStatus = async () => {
  const response = await api.get("/model/status");
  return response.data;
};

export const fetchSystemStatus = async () => {
  const response = await api.get("/system/status");
  return response.data;
};

export const refreshLiveData = async (payload = { sport: "cricket" }) => {
  const response = await api.post("/admin/refresh-live-data", payload);
  return response.data;
};
