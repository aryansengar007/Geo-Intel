import { useEffect, useMemo, useState } from 'react';

const API_BASE = 'http://localhost:8000';
const formatDate = (value: string) => value.slice(0, 10);

const formatCloud = (value: number) =>
  Number(value).toFixed(value < 1 ? 1 : 0);
type HealthState = {
status: string;
service: string;
dataset_status: string;
};

type AoiInfo = {
name: string;
description: string;
crs: { type: string; properties: { name: string } };
source: string;
source_method: string;
bbox: number[];
area_km2: number;
};

type Scene = {
scene_id: string;
acquisition_date: string;
cloud_cover: number;
tile_id?: string;
};

type ChangeClass = {
change_class: string;
pixel_count: number;
percentage_of_valid_pixels: number;
};

type SearchResult = {
  rank: number;
  score: number;
  scene_id: string;
  image_path: string;
  cloud_masked: boolean;
  acquisition_datetime: string;
  tile_id: string;
  mission: string;
  relative_orbit: string;
};

type SemanticSearchResponse = {
  query: string;
  model: string;
  device: string;
  total_indexed_vectors: number;
  results: SearchResult[];
};

type ChangeAnalysis = {
before_scene: string;
after_scene: string;
before_date: string;
after_date: string;
valid_pixel_count: number;
changed_pixel_count: number;
changed_percentage: number;
mean_spectral_change: number;
mean_delta_ndvi: number;
mean_delta_ndwi: number;
mean_delta_ndbi: number;
sensitivity: number;
change_raster_path: string;
change_class_raster_path: string;
change_classes: ChangeClass[];
};

const fallbackScenes: Scene[] = [
{
scene_id:
'S2B_MSIL2A_20221205T053209_N0510_R105_T43RGM_20240807T114821',
acquisition_date: '2022-12-05',
cloud_cover: 0,
tile_id: '43RGM',
},
{
scene_id:
'S2B_MSIL2A_20231210T053219_N0510_R105_T43RGM_20241101T190133',
acquisition_date: '2023-12-10',
cloud_cover: 0,
tile_id: '43RGM',
},
{
scene_id:
'S2C_MSIL2A_20241209T053251_N0511_R105_T43RGM_20260523T091305',
acquisition_date: '2024-12-09',
cloud_cover: 0,
tile_id: '43RGM',
},
{
scene_id:
'S2C_MSIL2A_20251224T053241_N0511_R105_T43RGM_20251224T091910',
acquisition_date: '2025-12-24',
cloud_cover: 0,
tile_id: '43RGM',
},
{
scene_id:
'S2C_MSIL2A_20260712T052651_N0512_R105_T43RGM_20260712T104114',
acquisition_date: '2026-07-12',
cloud_cover: 0.1,
tile_id: '43RGM',
},
];

const classLabels: Record<string, string> = {
vegetation_loss: 'Vegetation loss',
vegetation_growth: 'Vegetation growth',
water_expansion: 'Water expansion',
water_contraction: 'Water contraction',
built_up_construction: 'Built-up / construction',
other_surface_change: 'Other surface change',
};

export default function App() {
const [health, setHealth] = useState<HealthState | null>(null);
const [aoi, setAoi] = useState<AoiInfo | null>(null);
const [scenes, setScenes] = useState<Scene[]>(fallbackScenes);

const [beforeScene, setBeforeScene] = useState(fallbackScenes[0].scene_id);
const [afterScene, setAfterScene] = useState(fallbackScenes[1].scene_id);
const [sensitivity, setSensitivity] = useState(0.2);

const [analysis, setAnalysis] = useState<ChangeAnalysis | null>(null);
const [loading, setLoading] = useState(true);
const [analyzing, setAnalyzing] = useState(false);
const [error, setError] = useState('');

const [searchQuery, setSearchQuery] = useState('urban area and buildings');
const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
const [searching, setSearching] = useState(false);
const [searchError, setSearchError] = useState('');

useEffect(() => {
const fetchInitialData = async () => {
try {
const [healthRes, aoiRes, scenesRes] = await Promise.all([
  fetch(`${API_BASE}/health`),
  fetch(`${API_BASE}/api/aoi`),
  fetch(`${API_BASE}/api/scenes`),
]);

    if (healthRes.ok) {
      setHealth(await healthRes.json());
    }

    if (aoiRes.ok) {
      setAoi(await aoiRes.json());
    }

    if (scenesRes.ok) {
      const discoveredScenes: Scene[] = await scenesRes.json();

      if (discoveredScenes.length > 0) {
        const benchmarkScenes = discoveredScenes.filter((scene) =>
          fallbackScenes.some(
            (benchmark) => benchmark.scene_id === scene.scene_id,
          ),
        );

        if (benchmarkScenes.length >= 2) {
          setScenes(benchmarkScenes);
          setBeforeScene(benchmarkScenes[0].scene_id);
          setAfterScene(benchmarkScenes[1].scene_id);
        }
      }
    }
  } catch {
    setHealth({
      status: 'offline',
      service: 'GeoIntel',
      dataset_status: 'local benchmark available',
    });
  } finally {
    setLoading(false);
  }
};

fetchInitialData();

}, []);

const selectedBefore = useMemo(
  () => scenes.find((scene) => scene.scene_id === beforeScene),
  [beforeScene, scenes],
);

const selectedAfter = useMemo(
  () => scenes.find((scene) => scene.scene_id === afterScene),
  [afterScene, scenes],
);

const beforeImageUrl =
  `${API_BASE}/api/scenes/${beforeScene}/preview`;

const afterImageUrl =
  `${API_BASE}/api/scenes/${afterScene}/preview`;

const changeOverlayUrl =
  analysis && beforeScene && afterScene
    ? `${API_BASE}/api/change-analysis/overlay?before_scene=${encodeURIComponent(
        beforeScene
      )}&after_scene=${encodeURIComponent(afterScene)}`
    : null;
const runAnalysis = async () => {
if (beforeScene === afterScene) {
setError('Before and after scenes must be different.');
return;
}

setAnalyzing(true);
setError('');
setAnalysis(null);

try {
  const response = await fetch(`${API_BASE}/api/change-analysis`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      before_scene: beforeScene,
      after_scene: afterScene,
      sensitivity,
    }),
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      result?.detail
        ? JSON.stringify(result.detail)
        : 'Change analysis failed.',
    );
  }

  setAnalysis(result);
} catch (requestError) {
  setError(
    requestError instanceof Error
      ? requestError.message
      : 'Unable to run change analysis.',
  );
} finally {
  setAnalyzing(false);
}

};
const runSemanticSearch = async (query = searchQuery) => {
  const trimmedQuery = query.trim();

  if (!trimmedQuery) {
    setSearchError('Enter a semantic search query.');
    return;
  }

  setSearching(true);
  setSearchError('');

  try {
    const response = await fetch(`${API_BASE}/api/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: trimmedQuery,
        top_k: 5,
      }),
    });

    const result: SemanticSearchResponse = await response.json();

    if (!response.ok) {
      throw new Error(
        result?.results
          ? 'Semantic search failed.'
          : 'Semantic search service is unavailable.',
      );
    }

    setSearchResults(result.results);
  } catch (requestError) {
    setSearchResults([]);
    setSearchError(
      requestError instanceof Error
        ? requestError.message
        : 'Unable to run semantic search.',
    );
  } finally {
    setSearching(false);
  }
};

const timelineYears = [2022, 2023, 2024, 2025, 2026];

return (
<div className="app-shell">
<header className="topbar">
<div>
<div className="brand">GeoIntel</div>
<div className="text-muted">
Semantic Retrieval & Multi-Temporal Satellite Analysis
</div>
</div>

    <div className="topbar-right">
      <span className="pill">System</span>
      <span
        className={`status ${
          health?.status === 'ok'
            ? 'status-ok'
            : 'status-offline'
        }`}
      >
        {health ? health.status : loading ? 'LOADING' : 'OFFLINE'}
      </span>

      <span className="pill">Dataset</span>
      <span className="text-muted">
        {health?.dataset_status ?? 'checking...'}
      </span>
    </div>
  </header>

  <main className="main-layout">
    <aside className="panel left-panel">
      <h3>Analysis Controls</h3>

      <div className="field-group">
        <label>Area of Interest</label>
        <div className="stat-box">
          {aoi?.name ?? 'Gurugram study area'}
        </div>
      </div>

      <div className="field-group">
        <label>Before Scene</label>
        <select
          className="select-control"
          value={beforeScene}
          onChange={(event) => setBeforeScene(event.target.value)}
        >
          {scenes.map((scene) => (
            <option key={scene.scene_id} value={scene.scene_id}>
              {formatDate(scene.acquisition_date)}
            </option>
          ))}
        </select>

        {selectedBefore && (
          <span className="field-hint">
            Cloud: {formatCloud(selectedBefore.cloud_cover)}% · Tile:{' '}
            {selectedBefore.tile_id ?? 'N/A'}
          </span>
        )}
      </div>

      <div className="field-group">
        <label>After Scene</label>
        <select
          className="select-control"
          value={afterScene}
          onChange={(event) => setAfterScene(event.target.value)}
        >
          {scenes.map((scene) => (
            <option key={scene.scene_id} value={scene.scene_id}>
              {formatDate(scene.acquisition_date)}
            </option>
          ))}
        </select>

        {selectedAfter && (
          <span className="field-hint">
            Cloud: {formatCloud(selectedAfter.cloud_cover)}% · Tile:{' '}
            {selectedAfter.tile_id ?? 'N/A'}
          </span>
        )}
      </div>

      <div className="field-group">
        <label>Sensitivity: {sensitivity.toFixed(2)}</label>
        <input
          className="range-control"
          type="range"
          min="0.05"
          max="0.50"
          step="0.05"
          value={sensitivity}
          onChange={(event) =>
            setSensitivity(Number(event.target.value))
          }
        />
        <span className="field-hint">
          Higher values detect stronger changes only.
        </span>
      </div>

      <button
        className="primary-button"
        onClick={runAnalysis}
        disabled={analyzing || beforeScene === afterScene}
      >
        {analyzing ? 'Running Analysis...' : 'Run Change Analysis'}
      </button>

      {error && <div className="error-box">{error}</div>}

      <div className="field-group">
        <label>Derived Layers</label>
        <ul className="layer-list">
          <li>NDVI — vegetation response</li>
          <li>NDWI — water response</li>
          <li>NDBI — built-up response</li>
          <li>Quality-masked reflectance</li>
        </ul>
      </div>
    </aside>

    <section className="panel map-panel">
      <div className="imagery-workspace-header">
  <div className="imagery-title-group">
    <div className="imagery-title-row">
      <span className="eyebrow">Temporal Analysis</span>
      <span className="comparison-badge">BEFORE / AFTER</span>
    </div>

    <div className="map-title">
      {selectedBefore
        ? formatDate(selectedBefore.acquisition_date)
        : '—'}{' '}
      <span className="date-arrow">→</span>{' '}
      {selectedAfter
        ? formatDate(selectedAfter.acquisition_date)
        : '—'}
    </div>
  </div>

  <div className="imagery-toolbar">
    <div className="toolbar-item">
      <span className="toolbar-label">AOI</span>
      <strong>{aoi?.name ?? 'GURUGRAM'}</strong>
    </div>

    <div className="toolbar-divider" />

    <div className="toolbar-item">
      <span className="toolbar-label">MODE</span>
      <strong>TRUE COLOR</strong>
    </div>

    <div className="toolbar-divider" />

    <div className="toolbar-item">
      <span className="toolbar-label">RESOLUTION</span>
      <strong>10 m</strong>
    </div>

    {changeOverlayUrl && (
      <>
        <div className="toolbar-divider" />

        <div className="toolbar-status">
          <span className="toolbar-status-dot" />
          CHANGE OVERLAY
        </div>
      </>
    )}
  </div>
</div>

      <div className="analysis-map real-imagery-map">
        <div className="temporal-badge">
          MULTI-TEMPORAL COMPARISON · {selectedBefore ? formatDate(selectedBefore.acquisition_date) : '—'} → {selectedAfter ? formatDate(selectedAfter.acquisition_date) : '—'}
        </div>
      <div className="imagery-grid">
        <div className="imagery-panel">
          <div className="imagery-label">
            <span className="imagery-label-main">BEFORE</span>
            <span className="imagery-label-date">
              {selectedBefore ? formatDate(selectedBefore.acquisition_date) : '—'}
            </span>
          </div>

          <img
            src={beforeImageUrl}
            alt={`Sentinel-2 imagery from ${
              selectedBefore
                ? formatDate(selectedBefore.acquisition_date)
                : 'before date'
            }`}
            className="satellite-image"
          />
          <div className="imagery-footer">
            <span>Sentinel-2 L2A</span>
            <span>10 m</span>
            <span>{selectedBefore?.tile_id ?? '43RGM'}</span>
          </div>
        </div>

        <div className="imagery-panel">
          <div className="imagery-label">
            <span className="imagery-label-main">AFTER</span>
            <span className="imagery-label-date">
              {selectedAfter ? formatDate(selectedAfter.acquisition_date) : '—'}
            </span>

            {changeOverlayUrl && (
              <span className="overlay-active-badge">
                  OVERLAY
              </span>
            )}
          </div>

          <div className="overlay-image-container">
            <img
              src={afterImageUrl}
              alt={`Sentinel-2 imagery from ${
                selectedAfter
                  ? formatDate(selectedAfter.acquisition_date)
                  : 'after date'
              }`}
              className="satellite-image"
            />
            <div className="imagery-footer">
              <span>Sentinel-2 L2A</span>
              <span>10 m</span>
              <span>{selectedAfter?.tile_id ?? '43RGM'}</span>
            </div>

            {changeOverlayUrl && (
              <img
                src={changeOverlayUrl}
                alt="Spatial change classification overlay"
                className="change-overlay-image"
              />
            )}
          </div>
        </div>
      </div>

  {analysis && (
  <div className="change-legend">
    <div className="change-legend-header">
  <div>
    <span className="change-legend-title">
      CHANGE CLASSIFICATION
    </span>
    <span className="change-legend-subtitle">
      Algorithmic candidate changes requiring analyst review
    </span>
  </div>

  <span className="change-legend-count">
    {analysis.changed_pixel_count.toLocaleString()} candidates
  </span>
</div>

<div className="change-legend-items">
  {analysis.change_classes.map((item) => (
    <div
      key={item.change_class}
      className={`change-legend-item ${
        item.change_class === 'vegetation_loss'
          ? 'vegetation-loss'
          : item.change_class === 'vegetation_growth'
            ? 'vegetation-growth'
            : item.change_class === 'water_expansion'
              ? 'water-expansion'
              : item.change_class === 'water_contraction'
                ? 'water-contraction'
                : item.change_class === 'built_up_construction'
                  ? 'built-up'
                  : 'other-change'
      }`}
    >
      <span className="legend-dot" />

      <span className="legend-label">
        {classLabels[item.change_class] ?? item.change_class}
      </span>

      <span className="legend-count">
        {item.pixel_count.toLocaleString()}
      </span>
    </div>
  ))}
</div>
  </div>
)}

  {!analysis && (
    <div className="imagery-hint">
      Select two dates and run change analysis
    </div>
  )}
</div>

      <div className="map-legend">
        <span>
          <i className="dot blue" /> AOI
        </span>
        <span>
          <i className="dot orange" /> Change candidates
        </span>
        <span>
          <i className="dot green" /> Vegetation / water response
        </span>
      </div>

      <div className="result-path">
        <span>Raster outputs:</span>
        <code>
          {analysis
            ? analysis.change_raster_path
            : 'Generated after analysis'}
        </code>
      </div>
    </section>

    <aside className="panel right-panel">
      <h3>Evidence / Analysis</h3>

      <div className="evidence-grid">
        <div className="mini-stat">
          <span className="label">Location</span>
          <strong>{aoi?.name ?? 'Gurugram'}</strong>
        </div>

        <div className="mini-stat">
          <span className="label">Valid Pixels</span>
          <strong>
            {analysis
              ? analysis.valid_pixel_count.toLocaleString()
              : '—'}
          </strong>
        </div>

        <div className="mini-stat">
          <span className="label">Changed</span>
          <strong>
            {analysis
              ? `${analysis.changed_percentage.toFixed(3)}%`
              : '—'}
          </strong>
        </div>

        <div className="mini-stat">
          <span className="label">Sensitivity</span>
          <strong>
            {analysis ? analysis.sensitivity.toFixed(2) : '—'}
          </strong>
        </div>
      </div>

      <div className="metadata-card">
        <h4>Change Metrics</h4>

        <div className="metric-row">
          <span>Spectral change</span>
          <strong>
            {analysis
              ? analysis.mean_spectral_change.toFixed(4)
              : '—'}
          </strong>
        </div>

        <div className="metric-row">
          <span>ΔNDVI</span>
          <strong>
            {analysis
              ? analysis.mean_delta_ndvi.toFixed(4)
              : '—'}
          </strong>
        </div>

        <div className="metric-row">
          <span>ΔNDWI</span>
          <strong>
            {analysis
              ? analysis.mean_delta_ndwi.toFixed(4)
              : '—'}
          </strong>
        </div>

        <div className="metric-row">
          <span>ΔNDBI</span>
          <strong>
            {analysis
              ? analysis.mean_delta_ndbi.toFixed(4)
              : '—'}
          </strong>
        </div>
      </div>

      <div className="metadata-card">
        <h4>Change Classes</h4>

        {analysis ? (
          <div className="class-list">
            {analysis.change_classes.map((item) => (
              <div className="class-row" key={item.change_class}>
                <div>
                  <span className="class-name">
                    {classLabels[item.change_class] ??
                      item.change_class}
                  </span>
                  <span className="class-percent">
                    {item.percentage_of_valid_pixels.toFixed(4)}%
                  </span>
                </div>

                <strong>{item.pixel_count.toLocaleString()}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="empty-state">
            No temporal analysis has been run yet.
          </p>
        )}
      </div>

      <div className="metadata-card">
        <h4>Provenance</h4>
        <p>
          <strong>Source:</strong>{' '}
          {aoi?.source ?? 'study_area_polygon'}
        </p>
        <p>
          <strong>CRS:</strong>{' '}
          {aoi?.crs?.properties?.name ?? 'EPSG:4326'}
        </p>
        <p>
          <strong>Area:</strong>{' '}
          {aoi?.area_km2
            ? `${aoi.area_km2.toFixed(2)} km²`
            : 'Not available'}
        </p>
        <p className="provenance-note">
          Spectral change classes are algorithmic candidates and require
          analyst review before being treated as confirmed events.
        </p>
      </div>
    </aside>
  </main>

    <section className="panel semantic-panel">
    <div className="semantic-header">
      <div>
        <span className="eyebrow">Semantic Retrieval</span>
        <h3>Search Satellite Scenes by Meaning</h3>
        <p className="text-muted">
          Natural-language retrieval over indexed Sentinel-2 imagery using
          RemoteCLIP semantic embeddings.
        </p>
      </div>

      <div className="semantic-meta">
        <span className="pill">RemoteCLIP ViT-B/32</span>
        <span className="pill">5 indexed scenes</span>
        <span className="pill">43RGM</span>
        <span className="pill">CUDA</span>
      </div>
    </div>

    <div className="semantic-search-row">
      <input
        className="search-input"
        type="text"
        value={searchQuery}
        onChange={(event) => setSearchQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            runSemanticSearch();
          }
        }}
        placeholder="e.g. urban area and buildings"
      />

      <button
        className="primary-button search-button"
        onClick={() => runSemanticSearch()}
        disabled={searching}
      >
        {searching ? 'Searching...' : 'Search Scenes'}
      </button>
    </div>

    <div className="semantic-presets">
      <span className="field-hint">Try:</span>

      {[
        'urban area and buildings',
        'vegetation and trees',
        'water bodies',
        'roads and transport',
      ].map((query) => (
        <button
          key={query}
          className="preset-button"
          onClick={() => {
            setSearchQuery(query);
            runSemanticSearch(query);
          }}
          disabled={searching}
        >
          {query}
        </button>
      ))}
    </div>

    {searchError && (
      <div className="error-box">{searchError}</div>
    )}

    {searchResults.length > 0 && (
      <div className="semantic-results">
        {searchResults.map((result) => {
          const acquisitionDate = `${result.acquisition_datetime.slice(
            0,
            4,
          )}-${result.acquisition_datetime.slice(
            4,
            6,
          )}-${result.acquisition_datetime.slice(6, 8)}`;

          const imageUrl =
            `${API_BASE}/api/scenes/${encodeURIComponent(
              result.scene_id,
            )}/preview`;

          return (
            <article
              className="semantic-result-card"
              key={result.scene_id}
            >
              <img
                src={imageUrl}
                alt={`Satellite scene from ${acquisitionDate}`}
                className="semantic-result-image"
              />

              <div className="semantic-result-content">
                <div className="semantic-result-top">
                  <span className="result-rank">
                    #{result.rank}
                  </span>

                  <span className="result-date">
                    {acquisitionDate}
                  </span>
                </div>

                <strong>
                  Similarity {result.score.toFixed(3)}
                </strong>

                <span className="field-hint">
                  {result.mission} · Tile {result.tile_id} ·
                  Orbit {result.relative_orbit}
                </span>
              </div>
            </article>
          );
        })}
      </div>
    )}

    {searchResults.length === 0 && !searching && !searchError && (
      <div className="imagery-hint semantic-empty">
        Enter a natural-language query or select a preset to retrieve
        relevant satellite scenes.
      </div>
    )}

    <p className="provenance-note semantic-note">
      Retrieval scores represent cosine similarity between the text query
      and indexed image embeddings. They are ranking scores, not
      classification probabilities.
    </p>
  </section>

  <footer className="timeline-panel">
    <span className="timeline-label">Temporal coverage</span>

    <div className="timeline">
      {timelineYears.map((year) => (
        <div
          key={year}
          className={`year-pill ${
            scenes.some((scene) =>
              scene.acquisition_date.startsWith(String(year)),
            )
              ? 'year-active'
              : ''
          }`}
        >
          {year}
        </div>
      ))}
    </div>

    <span className="text-muted">
      Sentinel-2 L2A · 2022–2026
    </span>
  </footer>
</div>

);
}