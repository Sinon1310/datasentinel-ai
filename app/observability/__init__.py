"""UI-side integration boundary for Pipeline Observability.

The Streamlit page depends only on the typed contracts in :mod:`app.observability.contracts`:

* :class:`ObservabilityDataSource` — read-only queries (display).
* :class:`ObservabilityCommands` — the two human actions allowed by the approved design
  (request a record-only retry; cancel a PENDING run).

Implementations:

* :class:`~app.observability.unavailable.UnavailableBackend` — the "live" source until the
  observability backend (Milestone 3, Phases 1–3) exists. Returns no data and refuses actions.
* :class:`~app.observability.demo.DemoBackend` — deterministic, in-memory synthetic data for visual
  development and demos. Never persisted, never mixed with real records, always labelled.

A future adapter will wrap the real ``ObservabilityReader`` (read) and a narrow command service,
without changes to the page. UI code must never import a writer/tracker/store.
"""
