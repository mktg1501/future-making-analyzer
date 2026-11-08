"""
Consumer Future-Making Analyzer - interfaz (Streamlit).
La lógica de análisis está en motor.py.
"""
import hashlib
import hmac
import html
import json
import logging
import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Consumer Future-Making Analyzer",
    page_icon="FM",
    layout="wide",
)

from motor import (
    CHALLENGE_DEFINITIONS, CHALLENGE_POLICY_GUIDANCE, CONTEXT_TYPE_CHOICES,
    CROSS_ORIENTATION_NOTE, DEFAULT_THREAD, DEMO_BY_ID, DEMO_CONVERSATION,
    DEMO_CONVERSATION_REFERENCE, DEMO_EXAMPLES, EST_COST_PER_COMMENT,
    EXAMPLE_DISCLAIMER, HOMEPAGE_DESCRIPTION, INTERPRETIVE_USE_NOTE,
    LABELED_THREAD, MANAGER_MESSAGING_NOTE, MANAGER_ORIENTATION_GUIDANCE,
    MAX_COMMENTS, ORIENTATION_ORDER, ORIENTATIONS, PAPER_TITLE, PAPER_URL, PF_EV,
    POLICY_ORIENTATION_GUIDANCE, analyze_comment, analyze_document,
    build_comment_records_from_consultation, build_comment_records_from_csv,
    build_comment_records_from_paragraphs, build_comment_records_from_user_labels,
    compute_challenge_review_candidates, compute_dominant_distributions,
    ensure_unique_ids, has_comment_text_column, orientation_coexistence,
    orientations_by_thread, prepare_records, pseudonymize, select_within_limit,
    _clean_enum,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("future_making_app")

CTX_CODE = {label: code for label, code, _ in CONTEXT_TYPE_CHOICES}
CTX_LABEL = {code: label for label, code, _ in CONTEXT_TYPE_CHOICES}
CTX_DESC = {code: desc for _, code, desc in CONTEXT_TYPE_CHOICES}

SOURCE_CSV = "Archivo CSV (columna comment_text, text o comment)"
SOURCE_PASTE = "Texto pegado (comentarios separados por una línea en blanco)"
SOURCE_CONV = "Conversación con etiquetas 'User 1:', 'User 2:'"
SOURCE_DEMO = "Ejemplo del paper: conversación en un foro (4 usuarios)"

ORI_LABEL = {o: o.title() for o in ORIENTATION_ORDER}


# ─────────────────────────────────────────
# CONTRASEÑA
# ─────────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return
    st.title("Consumer Future-Making Analyzer")
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        st.error("Falta la variable APP_PASSWORD en el servidor. Pide al administrador configurarla.")
        st.stop()
    entered = st.text_input("Contraseña de acceso", type="password")
    if st.button("Entrar"):
        if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()


# ─────────────────────────────────────────
# COMPONENTES COMUNES
# ─────────────────────────────────────────
def _guidance_expander(title, g, with_avoid=False):
    with st.expander(title):
        st.write(g["lens"])
        st.write(g["implications"])
        st.markdown("**Qué monitorear:**")
        for m in g["monitor"]:
            st.markdown(f"- {m}")
        st.markdown(f"**Objetivo:** {g['objective']}")
        if with_avoid and g.get("avoid"):
            st.markdown(f"**Evitar:** {g['avoid']}")


def show_result(r: dict, reference: dict = None):
    ori = _clean_enum(r.get("main_orientation", "")).upper()
    act = _clean_enum(r.get("main_activity", "")).upper()
    sub = _clean_enum(r.get("activity_subtype", "")).upper()
    meta = ORIENTATIONS.get(ori, {})
    color = html.escape(meta.get("color", "#333333"))
    bg = html.escape(meta.get("bg", "#FFFFFF"))

    st.markdown(
        f"<div style='background:{bg};border-left:6px solid {color};"
        f"padding:12px 16px;border-radius:6px'>"
        f"<b>Orientación:</b> {html.escape(ori)} &nbsp;|&nbsp; "
        f"<b>Actividad dominante:</b> {html.escape(act)} / {html.escape(sub)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if reference:
        match = (act, sub, ori) == (reference["activity"], reference["subtype"], reference["orientation"])
        texto = "Coincide" if match else "No coincide"
        st.info(
            f"{texto} con la referencia del paper: {reference['orientation'].title()} · "
            f"{reference['activity'].title()} / {reference['subtype'].title()}. "
            f"{EXAMPLE_DISCLAIMER}"
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Justificación de la actividad**")
        st.write(str(r.get("activity_rationale", "")))
        st.markdown("**Justificación de la orientación**")
        st.write(str(r.get("orientation_rationale", "")))
    with col2:
        st.markdown("**Narrativa identificada:** " + str(r.get("narrative_identified", "")))
        st.markdown("**Emociones:** " + str(r.get("emotions_identified", "")))
        st.markdown("**Temporalidad:** " + str(r.get("temporality_identified", "")))
        st.markdown("**Condiciones de adopción:** " + str(r.get("notable_conditions_of_adoption", "")))
        st.markdown("**Texto de apoyo:** " + str(r.get("supporting_text", "")))

    secondary = r.get("secondary_classifications") or []
    if secondary:
        st.markdown("**Clasificaciones secundarias**")
        for s in secondary:
            if isinstance(s, dict):
                st.markdown(
                    f"- {s.get('activity', '')} / {s.get('activity_subtype', '')} "
                    f"({s.get('orientation', '')}): {s.get('rationale', '')}"
                )

    if r.get("context_note"):
        st.markdown("**Nota de contexto:** " + str(r.get("context_note")))
    if r.get("input_scope_warning"):
        st.warning("Nota sobre el alcance de la entrada: " + str(r.get("input_scope_warning")))

    if ori in POLICY_ORIENTATION_GUIDANCE:
        _guidance_expander("Guía de política para esta orientación (Pasos 2 y 4)",
                           POLICY_ORIENTATION_GUIDANCE[ori])
        _guidance_expander("Guía gerencial para esta orientación (Pasos 2 y 4)",
                           MANAGER_ORIENTATION_GUIDANCE[ori], with_avoid=True)

    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# PESTAÑA 1: UN COMENTARIO
# ─────────────────────────────────────────
def _demo_label(demo_id: str) -> str:
    if not demo_id:
        return "— Escribir mi propio comentario —"
    return DEMO_BY_ID[demo_id]["titulo"]


def _load_single_demo():
    demo_id = st.session_state.get("demo_single", "")
    st.session_state.pop("single_result", None)
    ex = DEMO_BY_ID.get(demo_id)
    if ex:
        st.session_state["pf_single"] = PF_EV
        st.session_state["ctx_single"] = CTX_LABEL[ex["context_type"]]
        st.session_state["context_single"] = ex["context"]
        st.session_state["comment_single"] = ex["comment"]


def tab_single():
    st.subheader("Analizar un comentario")
    st.caption("Escribe tu propio comentario o carga uno de los ejemplos del paper para ver cómo funciona.")

    st.selectbox(
        "Ejemplo del paper (opcional)",
        [""] + list(DEMO_BY_ID),
        format_func=_demo_label,
        key="demo_single",
        on_change=_load_single_demo,
    )
    pf = st.text_area("Futuro prescrito", value=PF_EV, height=100, key="pf_single")
    label = st.selectbox("Tipo de contexto", list(CTX_CODE), key="ctx_single")
    ctype = CTX_CODE[label]
    st.caption(CTX_DESC[ctype])
    context = st.text_area("Contexto (opcional)", height=120, key="context_single")
    comment = st.text_area("Comentario a clasificar", height=160, key="comment_single")
    with_secondary = st.checkbox(
        "Permitir clasificaciones secundarias (si el comentario tiene una segunda práctica separable)",
        key="secondary_single",
    )

    demo = DEMO_BY_ID.get(st.session_state.get("demo_single", ""))
    reference = demo["referencia"] if demo and comment.strip() == demo["comment"] else None
    if demo and reference:
        with st.expander("Clasificación de referencia del paper (sin llamada al modelo)", expanded=True):
            st.markdown(
                f"**Orientación:** {reference['orientation'].title()} · "
                f"**Actividad:** {reference['activity'].title()} / {reference['subtype'].title()}"
            )
            st.write(demo["nota_paper"])
            st.caption(EXAMPLE_DISCLAIMER)

    if st.button("Clasificar", type="primary", key="btn_single"):
        if not pf.strip():
            st.warning("Escribe el futuro prescrito antes de clasificar.")
            return
        if not comment.strip():
            st.warning("Escribe o carga un comentario antes de clasificar.")
            return
        effective_ctype = ctype if context.strip() else "NONE"
        with st.spinner("Analizando el comentario..."):
            try:
                r = analyze_comment(
                    pf.strip(), comment.strip(), context.strip(), effective_ctype,
                    ctype == "CONSULTATION_PROMPT", allow_secondary=with_secondary,
                )
            except (ValueError, RuntimeError) as e:
                st.error(str(e))
                return
            except Exception:
                logger.exception("Error al clasificar un comentario individual")
                st.error("No fue posible completar el análisis. Intenta de nuevo.")
                return
        st.session_state["single_result"] = {
            "r": r,
            "key": (pf.strip(), comment.strip(), context.strip(), ctype, with_secondary),
        }

    stored = st.session_state.get("single_result")
    if not stored:
        return
    current_key = (pf.strip(), comment.strip(), context.strip(), ctype, with_secondary)
    if stored["key"] != current_key:
        st.info("La entrada cambió desde la última clasificación. Vuelve a clasificar para actualizar el resultado.")
        return
    show_result(stored["r"], reference)
    st.download_button(
        "Descargar resultado (JSON)",
        data=json.dumps(stored["r"], ensure_ascii=False, indent=2),
        file_name="resultado.json",
        mime="application/json",
    )


# ─────────────────────────────────────────
# PESTAÑA 2: VARIOS COMENTARIOS Y PANEL
# ─────────────────────────────────────────
def _read_records(source, is_consult, uploaded, text):
    """Devuelve (registros, mensajes[(nivel, texto)])."""
    msgs = []
    if source == SOURCE_DEMO:
        return build_comment_records_from_user_labels(DEMO_CONVERSATION), msgs

    if source == SOURCE_CSV:
        if uploaded is None:
            return [], msgs
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            logger.exception("No se pudo leer el CSV")
            msgs.append(("error", "No se pudo leer el CSV. Verifica que sea un archivo CSV válido."))
            return [], msgs
        if not has_comment_text_column(df):
            msgs.append(("error", "El CSV necesita una columna llamada comment_text, text o comment."))
            return [], msgs
        return build_comment_records_from_csv(df), msgs

    if not text.strip():
        return [], msgs

    if source == SOURCE_CONV:
        records = build_comment_records_from_user_labels(text)
        if not records:
            msgs.append(("warning", "Se encontraron menos de dos etiquetas 'User N:'. Revisa el formato."))
        return records, msgs

    if is_consult:
        msgs.append(("info", "Para consultas, un CSV con comment_id es más confiable que el texto pegado."))
        return build_comment_records_from_consultation(text), msgs

    records, discarded = build_comment_records_from_paragraphs(text)
    if discarded:
        msgs.append(("info", f"Se descartaron {discarded} bloque(s) con menos de dos palabras."))
    return records, msgs


def _signature(pf, consult_q, is_consult, records):
    payload = json.dumps({
        "pf": pf,
        "cq": consult_q if is_consult else "",
        "recs": [(r["comment_id"], r["comment_text"], r["thread_id"], r.get("parent_comment_id"))
                 for r in records],
    }, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _author_label(r: dict, pseudo: bool) -> str:
    author = r.get("_author", "")
    return pseudonymize(author) if pseudo else author


def _pair_label(r: dict) -> str:
    return (f"{_clean_enum(r.get('main_orientation', '')).title()} · "
            f"{_clean_enum(r.get('activity_subtype', '')).title()}")


def _render_pie(coex: dict):
    values = [{
        "orientacion": ORI_LABEL[o],
        "comentarios": coex["counts"][o],
        "color": ORIENTATIONS[o]["color"],
    } for o in coex["present"]]
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Distribución de orientaciones de futuro-making en los comentarios analizados.",
        "data": {"values": values},
        "mark": {"type": "arc", "tooltip": True},
        "encoding": {
            "theta": {"field": "comentarios", "type": "quantitative"},
            "color": {
                "field": "orientacion", "type": "nominal",
                "scale": {"domain": [v["orientacion"] for v in values],
                          "range": [v["color"] for v in values]},
                "legend": {"title": "Orientación"},
            },
            "tooltip": [
                {"field": "orientacion", "type": "nominal", "title": "Orientación"},
                {"field": "comentarios", "type": "quantitative", "title": "Comentarios"},
            ],
        },
        "width": 320,
        "height": 320,
    }
    st.vega_lite_chart(spec)


def _coexistence_sentence(coex: dict) -> str:
    if coex["n_present"] == 0:
        return "No hay orientaciones clasificadas."
    if coex["n_present"] == 1:
        return (f"Solo aparece una orientación ({ORI_LABEL[coex['present'][0]]}) en este conjunto, "
                "por lo que no se observa coexistencia.")
    names = ", ".join(ORI_LABEL[o] for o in coex["present"])
    return (f"Coexisten {coex['n_present']} de 4 orientaciones en este conjunto: {names}. "
            "La coexistencia describe el patrón del corpus; no clasifica a las personas.")


def _render_demo_comparison(valid: list):
    rows = []
    for r in valid:
        ref = DEMO_CONVERSATION_REFERENCE.get(r["_comment_id"])
        if not ref:
            continue
        pred = (_clean_enum(r.get("main_activity", "")).upper(),
                _clean_enum(r.get("activity_subtype", "")).upper(),
                _clean_enum(r.get("main_orientation", "")).upper())
        exp = (ref["activity"], ref["subtype"], ref["orientation"])
        rows.append({
            "Participante": r["_comment_id"],
            "Referencia del paper": f"{exp[2].title()} · {exp[0].title()} / {exp[1].title()}",
            "Modelo": f"{pred[2].title()} · {pred[0].title()} / {pred[1].title()}",
            "Coincide": "Sí" if pred == exp else "No",
        })
    if rows:
        st.markdown("**Comparación con la referencia del paper**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(EXAMPLE_DISCLAIMER)


def _render_pairs(pairs, pseudo: bool):
    for pair in pairs:
        p, r = pair["parent"], pair["reply"]
        st.markdown(
            f"**{_author_label(p, pseudo) or 'Comentario previo'} ({_pair_label(p)}) → "
            f"{_author_label(r, pseudo) or 'Respuesta'} ({_pair_label(r)})**"
        )
        st.markdown("> " + p["_comment_text"].replace("\n", " "))
        st.markdown("> " + r["_comment_text"].replace("\n", " "))
        st.divider()


def _show_dashboard(stored: dict, pseudo: bool):
    results = stored["results"]
    valid = [r for r in results if "_error" not in r]
    errors = len(results) - len(valid)

    st.success(f"Análisis completo: {len(valid)} de {len(results)} comentarios clasificados.")
    if errors:
        st.warning(f"{errors} comentario(s) con error. No se cuentan en la distribución. "
                   "Revisa la columna Error de la tabla.")
    if stored.get("omitted"):
        st.warning(f"Se omitieron {stored['omitted']} comentarios por el límite de {MAX_COMMENTS}.")
    if not valid:
        return

    st.markdown("### Coexistencia de orientaciones")
    coex = orientation_coexistence(valid)
    col_chart, col_text = st.columns([1, 1])
    with col_chart:
        _render_pie(coex)
    with col_text:
        st.write(_coexistence_sentence(coex))
        table = pd.DataFrame([{
            "Orientación": ORI_LABEL[o],
            "Comentarios": coex["counts"][o],
            "Porcentaje": f"{coex['shares'][o]:.0%}",
        } for o in ORIENTATION_ORDER])
        st.dataframe(table, hide_index=True)
        st.caption("El gráfico muestra la misma información que la tabla, para lectura sin color.")

    by_thread = orientations_by_thread(valid)
    if by_thread:
        st.markdown("**Coexistencia dentro de cada conversación**")
        rows = []
        for tid, info in by_thread.items():
            present = [o for o in ORIENTATION_ORDER if info["counts"][o]]
            rows.append({
                "Conversación": "Conversación con etiquetas" if tid == LABELED_THREAD else str(tid),
                "Comentarios": info["n"],
                "Orientaciones presentes": f"{len(present)} de 4: " + ", ".join(ORI_LABEL[o] for o in present),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("### Distribución de actividades")
    dist = compute_dominant_distributions(valid)
    if dist["activity_counts"]:
        st.dataframe(pd.DataFrame(list(dist["activity_counts"].items()),
                                  columns=["Actividad", "Comentarios"]), hide_index=True)
    if dist["subtype_counts"]:
        st.dataframe(pd.DataFrame(list(dist["subtype_counts"].items()),
                                  columns=["Actividad / subtipo", "Comentarios"]), hide_index=True)

    if stored.get("demo"):
        _render_demo_comparison(valid)

    st.markdown("### Revisión humana de desafíos")
    st.caption("Son pistas para revisión manual, no diagnósticos automáticos. Solo se interpretan pares enlazados.")
    candidates = compute_challenge_review_candidates(results)
    for key, meta in CHALLENGE_DEFINITIONS.items():
        pairs = candidates.get(key, [])
        with st.expander(f"{meta['label']}: {len(pairs)} par(es) enlazado(s) para revisar"):
            st.write(CHALLENGE_POLICY_GUIDANCE[key]["diagnostic_question"])
            st.write(CHALLENGE_POLICY_GUIDANCE[key]["action"])
            if pairs:
                st.markdown("**Pares enlazados (orden de la conversación)**")
                _render_pairs(pairs, pseudo)

    with st.expander("Notas de interpretación"):
        st.write(MANAGER_MESSAGING_NOTE)
        st.write(CROSS_ORIENTATION_NOTE)

    st.markdown("### Tabla de resultados")
    rows = []
    for r in results:
        rows.append({
            "ID": r.get("_comment_id", ""),
            "Autor": _author_label(r, pseudo),
            "Comentario": r.get("_comment_text", ""),
            "Orientación": _clean_enum(r.get("main_orientation", "")),
            "Actividad": _clean_enum(r.get("main_activity", "")),
            "Subtipo": _clean_enum(r.get("activity_subtype", "")),
            "Texto de apoyo": r.get("supporting_text", ""),
            "Error": r.get("_error", ""),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    st.download_button(
        "Descargar resultados (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="resultados_orientaciones.csv",
        mime="text/csv",
    )
    st.caption("Los comentarios pueden contener nombres u otros datos personales. Revísalos antes de compartir el archivo.")


def tab_multiple():
    st.subheader("Mapear orientaciones en varios comentarios")
    st.caption("Puedes subir un CSV, pegar comentarios o usar la conversación de ejemplo del paper.")

    pf = st.text_area("Futuro prescrito", value=PF_EV, height=100, key="pf_multi")
    source = st.radio(
        "Fuente de los comentarios",
        [SOURCE_CSV, SOURCE_PASTE, SOURCE_CONV, SOURCE_DEMO],
        key="source_multi",
    )
    uploaded, text = None, ""
    if source == SOURCE_CSV:
        uploaded = st.file_uploader("Sube el archivo CSV", type=["csv"], key="csv_upload")
    elif source != SOURCE_DEMO:
        text = st.text_area("Pega los comentarios aquí", height=250, key="text_multi")

    is_consult = st.checkbox("Son respuestas a una consulta pública", key="is_consult")
    consult_q = ""
    if is_consult:
        consult_q = st.text_area(
            "Pregunta o propuesta de la consulta (contexto de cada respuesta)",
            height=100, key="consult_question",
        )
    pseudo = st.checkbox("Seudonimizar autores en la tabla y la descarga", value=True, key="pseudo_multi")

    if not pf.strip():
        st.warning("Escribe el futuro prescrito antes de analizar.")
        return
    if is_consult and not consult_q.strip():
        st.warning("Escribe la pregunta de la consulta antes de analizar.")
        return
    if source == SOURCE_DEMO:
        st.info(EXAMPLE_DISCLAIMER)

    records, msgs = _read_records(source, is_consult, uploaded, text)
    for level, msg in msgs:
        getattr(st, level)(msg)
    if not records:
        st.info("Aún no hay comentarios para analizar.")
        return

    renamed = ensure_unique_ids(records)
    if renamed:
        st.warning(f"Hay {renamed} identificador(es) repetido(s); se renombraron para no perder comentarios.")

    selected, omitted = select_within_limit(records, MAX_COMMENTS)
    prepare_records(selected, consult_q.strip() if is_consult else "", is_consult)

    st.write(f"Comentarios detectados: **{len(records)}**. Se analizarán: **{len(selected)}**.")
    if omitted:
        st.warning(f"Se omitirán {omitted} comentarios para controlar el costo (límite {MAX_COMMENTS}). "
                   "Se conservan los hilos completos.")
    st.info(f"Costo estimado: ~USD {len(selected) * EST_COST_PER_COMMENT:.2f}.")

    with st.expander("Vista previa de los comentarios"):
        st.dataframe(pd.DataFrame([{
            "ID": r["comment_id"],
            "Autor": pseudonymize(r.get("author", "")) if pseudo else r.get("author", ""),
            "Comentario": r["comment_text"],
        } for r in selected]), hide_index=True)

    signature = _signature(pf.strip(), consult_q.strip(), is_consult, selected)

    if st.button("Analizar comentarios", type="primary", key="btn_multi"):
        progress = st.progress(0, text="Iniciando análisis...")
        try:
            results = analyze_document(
                selected, pf.strip(),
                lambda done, total: progress.progress(done / total, text=f"Analizados {done}/{total}"),
            )
        except Exception:
            logger.exception("Error durante el análisis por lotes")
            st.error("Ocurrió un error durante el análisis. Revisa el registro del servidor.")
            return
        progress.empty()
        st.session_state["multi"] = {
            "signature": signature,
            "results": results,
            "omitted": omitted,
            "demo": source == SOURCE_DEMO,
        }

    stored = st.session_state.get("multi")
    if not stored:
        return
    if stored["signature"] != signature:
        st.info("La entrada cambió desde el último análisis. Vuelve a analizar para actualizar el panel.")
        return
    _show_dashboard(stored, pseudo)


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
check_password()

st.title("Consumer Future-Making Analyzer")
with st.expander("Acerca de esta herramienta"):
    st.markdown(HOMEPAGE_DESCRIPTION)
    st.caption(f'"{PAPER_TITLE}" | Lee el artículo: {PAPER_URL}')
    st.caption(INTERPRETIVE_USE_NOTE)

tab1, tab2 = st.tabs(["Analizar un comentario", "Varios comentarios y panel"])
with tab1:
    tab_single()
with tab2:
    tab_multiple()
