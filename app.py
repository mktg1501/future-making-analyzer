"""
Interfaz de Consumer Future-Making Analyzer.
Usa el motor de análisis de motor.py.
"""
import os
import hmac
import html
import json

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Consumer Future-Making Analyzer",
    page_icon="FM",
    layout="wide",
)

from motor import (
    PAPER_TITLE, HOMEPAGE_DESCRIPTION, INTERPRETIVE_USE_NOTE, CONTEXT_TYPE_CHOICES,
    ORIENTATIONS, CHALLENGE_DEFINITIONS, PF_EV, EXAMPLES,
    POLICY_ROADMAP_STEPS, MANAGER_ROADMAP_STEPS,
    POLICY_ORIENTATION_GUIDANCE, MANAGER_ORIENTATION_GUIDANCE,
    CHALLENGE_POLICY_GUIDANCE, CHALLENGE_MANAGER_GUIDANCE,
    MANAGER_MESSAGING_NOTE, CROSS_ORIENTATION_NOTE,
    analyze_comment, analyze_document, index_threads, build_context,
    build_comment_records_from_csv, build_comment_records_from_paragraphs,
    build_comment_records_from_user_labels, build_comment_records_from_consultation,
    compute_dominant_distributions, compute_challenge_review_candidates,
    run_consistency_suite, run_framework_validation, _clean_enum,
)

MAX_COMENTARIOS = int(os.environ.get("MAX_COMENTARIOS", "300"))


# ─────────────────────────────────────────
# ACCESO CON CONTRASEÑA
# ─────────────────────────────────────────
def check_password():
    if st.session_state.get("autenticado"):
        return
    st.title("Consumer Future-Making Analyzer")
    clave_correcta = os.environ.get("APP_PASSWORD", "")
    if not clave_correcta:
        st.error("Falta la variable APP_PASSWORD en el servidor. Pide al administrador que la configure.")
        st.stop()
    ingresada = st.text_input("Contraseña de acceso", type="password")
    if st.button("Entrar"):
        if hmac.compare_digest(ingresada.encode("utf-8"), clave_correcta.encode("utf-8")):
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()


# ─────────────────────────────────────────
# PRESENTACIÓN DE UN RESULTADO
# ─────────────────────────────────────────
def mostrar_resultado(r: dict):
    ori = _clean_enum(r.get("main_orientation", "")).upper()
    act = _clean_enum(r.get("main_activity", "")).upper()
    sub = _clean_enum(r.get("activity_subtype", "")).upper()
    meta = ORIENTATIONS.get(ori, {})
    color = meta.get("color", "#333333")
    bg = meta.get("bg", "#FFFFFF")

    st.markdown(
        f"<div style='background:{bg};border-left:6px solid {color};"
        f"padding:12px 16px;border-radius:6px'>"
        f"<b>Orientación:</b> {html.escape(ori)} &nbsp;|&nbsp; "
        f"<b>Actividad principal:</b> {html.escape(act)} / {html.escape(sub)}"
        f"</div>",
        unsafe_allow_html=True,
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

    secundarias = r.get("secondary_classifications") or []
    if secundarias:
        st.markdown("**Clasificaciones secundarias**")
        for s in secundarias:
            if isinstance(s, dict):
                st.markdown(
                    f"- {s.get('activity', '')} / {s.get('activity_subtype', '')} "
                    f"({s.get('orientation', '')}): {s.get('rationale', '')}"
                )

    if r.get("context_note"):
        st.markdown("**Nota sobre el contexto:** " + str(r.get("context_note")))
    if r.get("input_scope_warning"):
        st.warning("Advertencia: " + str(r.get("input_scope_warning")))

    if ori in POLICY_ORIENTATION_GUIDANCE:
        with st.expander("Orientación para la política (Paso 2 y 4 del marco)"):
            g = POLICY_ORIENTATION_GUIDANCE[ori]
            st.write(g["lens"])
            st.write(g["implications"])
            st.markdown("**Qué monitorear:**")
            for m in g["monitor"]:
                st.markdown(f"- {m}")
            st.markdown(f"**Objetivo:** {g['objective']}")

        with st.expander("Orientación para la gestión (Paso 2 y 4 del marco)"):
            g = MANAGER_ORIENTATION_GUIDANCE[ori]
            st.write(g["lens"])
            st.write(g["implications"])
            st.markdown("**Qué monitorear:**")
            for m in g["monitor"]:
                st.markdown(f"- {m}")
            st.markdown(f"**Objetivo:** {g['objective']}")
            st.markdown(f"**Evitar:** {g['avoid']}")

    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# PESTAÑA 1: UN COMENTARIO
# ─────────────────────────────────────────
def pestana_individual():
    st.subheader("Analizar un comentario")
    pf = st.text_area("Futuro prescrito por la intervención", value=PF_EV, height=100, key="pf_ind")

    etiquetas = [label for label, _, _ in CONTEXT_TYPE_CHOICES]
    etiqueta = st.selectbox("Tipo de contexto", etiquetas, key="ctx_ind")
    tipo = next(code for label, code, _ in CONTEXT_TYPE_CHOICES if label == etiqueta)
    descripcion = next(desc for label, _, desc in CONTEXT_TYPE_CHOICES if label == etiqueta)
    st.caption(descripcion)

    contexto = st.text_area("Contexto (opcional)", height=120, key="contexto_ind")
    comentario = st.text_area("Comentario a clasificar", height=160, key="coment_ind")

    if st.button("Clasificar", type="primary", key="btn_ind"):
        if not comentario.strip():
            st.warning("Escribe o pega un comentario antes de clasificar.")
            return
        ctype = tipo if contexto.strip() else "NONE"
        with st.spinner("Analizando el comentario..."):
            try:
                r = analyze_comment(pf, comentario.strip(), contexto.strip(), ctype, False, None, True)
            except Exception as e:
                st.error(f"No se pudo completar el análisis: {e}")
                return
        mostrar_resultado(r)
        st.download_button(
            "Descargar resultado (JSON)",
            data=json.dumps(r, ensure_ascii=False, indent=2),
            file_name="resultado.json",
            mime="application/json",
        )


# ─────────────────────────────────────────
# PESTAÑA 2: VARIOS COMENTARIOS
# ─────────────────────────────────────────
def pestana_documento():
    st.subheader("Mapear orientaciones en varios comentarios")
    pf = st.text_area("Futuro prescrito por la intervención", value=PF_EV, height=100, key="pf_doc")

    origen = st.radio(
        "Origen de los comentarios",
        [
            "Archivo CSV (columna comment_text, text o comment)",
            "Texto pegado (comentarios separados por una línea en blanco)",
            "Conversación con etiquetas 'User 1:', 'User 2:', etc.",
        ],
        key="origen_doc",
    )
    es_consulta = st.checkbox("Son respuestas a una consulta pública", key="es_consulta")
    pregunta = ""
    if es_consulta:
        pregunta = st.text_area("Pregunta o propuesta de la consulta", height=100, key="pregunta")

    records = []
    if origen.startswith("Archivo"):
        archivo = st.file_uploader("Sube el archivo CSV", type=["csv"], key="csv")
        if archivo is not None:
            try:
                df = pd.read_csv(archivo)
                records = build_comment_records_from_csv(df)
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
    else:
        texto = st.text_area("Pega aquí los comentarios", height=250, key="texto_doc")
        if origen.startswith("Texto") and texto.strip():
            if es_consulta:
                records = build_comment_records_from_consultation(texto)
            else:
                records = build_comment_records_from_paragraphs(texto)
        elif origen.startswith("Conversación") and texto.strip():
            records = build_comment_records_from_user_labels(texto)

    if not records:
        st.info("Aún no hay comentarios para analizar.")
        return

    st.write(f"Comentarios detectados: **{len(records)}**")
    if len(records) > MAX_COMENTARIOS:
        st.warning(
            f"Se analizarán solo los primeros {MAX_COMENTARIOS} comentarios "
            "para proteger el uso de la API. Divide el archivo en partes si necesitas más."
        )
        records = records[:MAX_COMENTARIOS]

    with st.expander("Vista previa de los comentarios"):
        st.dataframe(pd.DataFrame([
            {"ID": r["comment_id"], "Autor": r.get("author", ""), "Comentario": r["comment_text"]}
            for r in records
        ]), use_container_width=True)

    if not st.button("Analizar comentarios", type="primary", key="btn_doc"):
        return

    # Contexto para cada comentario (hilo, respuesta padre, consulta)
    by_id, order = index_threads(records)
    for rec in records:
        ctx, ctype, _ = build_context(rec, by_id, order, pregunta, es_consulta)
        rec["context_text"] = ctx
        rec["context_type"] = ctype
        rec["is_consultation"] = es_consulta

    barra = st.progress(0, text="Iniciando análisis...")
    try:
        results = analyze_document(records, pf, None, barra)
    except Exception as e:
        st.error(f"Ocurrió un error durante el análisis: {e}")
        return
    barra.empty()

    autor_por_id = {r["comment_id"]: r.get("author", "") for r in records}
    filas = []
    for r in results:
        filas.append({
            "ID": r.get("_comment_id", ""),
            "Autor": autor_por_id.get(r.get("_comment_id"), ""),
            "Comentario": r.get("_comment_text", ""),
            "Orientación": _clean_enum(r.get("main_orientation", "")),
            "Actividad": _clean_enum(r.get("main_activity", "")),
            "Subtipo": _clean_enum(r.get("activity_subtype", "")),
            "Texto de apoyo": r.get("supporting_text", ""),
            "Error": r.get("_error", ""),
        })
    df_result = pd.DataFrame(filas)

    st.success(f"Análisis terminado: {len(results)} comentarios.")
    st.dataframe(df_result, use_container_width=True)

    st.download_button(
        "Descargar resultados (CSV)",
        data=df_result.to_csv(index=False).encode("utf-8-sig"),
        file_name="orientaciones_resultados.csv",
        mime="text/csv",
    )

    dist = compute_dominant_distributions(results)
    st.markdown("### Distribución")
    if dist["orientation_counts"]:
        st.bar_chart(pd.Series(dist["orientation_counts"], name="Comentarios"))
    if dist["activity_counts"]:
        st.write("Actividades:", dist["activity_counts"])

    st.markdown("### Revisión humana de desafíos (vinculados entre comentarios)")
    st.caption(
        "Estas son pistas para revisar a mano, no diagnósticos automáticos. "
        "Los desafíos solo se interpretan en patrones entre comentarios relacionados."
    )
    candidatos = compute_challenge_review_candidates(results)
    for clave, meta in CHALLENGE_DEFINITIONS.items():
        n = len(candidatos.get(clave, []))
        with st.expander(f"{meta['label']}: {n} pares vinculados para revisar"):
            st.write(CHALLENGE_POLICY_GUIDANCE[clave]["diagnostic_question"])
            st.write(CHALLENGE_POLICY_GUIDANCE[clave]["action"])

    with st.expander("Notas de interpretación"):
        st.write(MANAGER_MESSAGING_NOTE)
        st.write(CROSS_ORIENTATION_NOTE)
    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# PESTAÑA 3: MARCO (roadmaps)
# ─────────────────────────────────────────
def pestana_marco():
    st.subheader("Hoja de ruta para la política (7 pasos)")
    for num, titulo, texto in POLICY_ROADMAP_STEPS:
        st.markdown(f"**{num}. {titulo}**")
        st.write(texto)

    st.subheader("Hoja de ruta para la gestión (6 pasos)")
    for num, titulo, texto in MANAGER_ROADMAP_STEPS:
        st.markdown(f"**{num}. {titulo}**")
        st.write(texto)

    st.caption(INTERPRETIVE_USE_NOTE)


# ─────────────────────────────────────────
# PESTAÑA 4: HERRAMIENTAS DE DESARROLLADOR (protegidas)
# ─────────────────────────────────────────
def pestana_desarrollador():
    clave_dev = os.environ.get("DEV_PASSWORD", "")
    if not clave_dev:
        st.info("Herramientas de desarrollador desactivadas. Falta la variable DEV_PASSWORD.")
        return
    if not st.session_state.get("dev_ok"):
        p = st.text_input("Contraseña de desarrollador", type="password", key="dev_pw")
        if st.button("Desbloquear", key="dev_btn"):
            if hmac.compare_digest(p.encode("utf-8"), clave_dev.encode("utf-8")):
                st.session_state["dev_ok"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta.")
        return

    st.subheader("Pruebas de estructura del marco (sin usar la API)")
    if st.button("Ejecutar pruebas de estructura", key="dev_val"):
        for nombre, ok, detalle in run_framework_validation():
            st.write(f"{'✅' if ok else '❌'} {nombre} — {detalle}")

    st.subheader("Consistencia con ejemplos de referencia (usa la API)")
    st.caption(
        "Verificación interna con ejemplos ilustrativos. No constituye validación empírica "
        "ni fiabilidad entre codificadores."
    )
    if st.button("Ejecutar consistencia", key="dev_cons"):
        with st.spinner("Ejecutando ejemplos..."):
            resumen = run_consistency_suite(None)
        st.write(f"Acuerdo general: {resumen['overall_agreement']:.0%}")
        st.dataframe(pd.DataFrame([
            {
                "Ejemplo": r["example"],
                "Esperado": " / ".join(r["expected"]),
                "Predicho": " / ".join(str(x) for x in r["predicted"]),
                "Coincide": r["match"],
                "Error": r.get("error", ""),
            }
            for r in resumen["results"]
        ]), use_container_width=True)


# ─────────────────────────────────────────
# APLICACIÓN
# ─────────────────────────────────────────
check_password()

st.title("Consumer Future-Making Analyzer")
with st.expander("Acerca de esta herramienta"):
    st.markdown(HOMEPAGE_DESCRIPTION)
    st.caption(INTERPRETIVE_USE_NOTE)

tab1, tab2, tab3, tab4 = st.tabs([
    "Comentario individual",
    "Varios comentarios",
    "Hojas de ruta",
    "Desarrollador",
])
with tab1:
    pestana_individual()
with tab2:
    pestana_documento()
with tab3:
    pestana_marco()
with tab4:
    pestana_desarrollador()
