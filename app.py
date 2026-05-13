import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Procesador RSP PRO", layout="wide")
st.title("🛣️ Procesador Vial Dynatest Mark III")
st.markdown("Automatización de inventarios de rugosidad con auditoría de singularidades.")

def procesar_streamlit(uploaded_file, ruta, sentido, intervalo_str, umbral_puntual, max_puente_m):
    data_iri, data_gps, data_eventos = [], [], []
    fecha_registro = "Sin Fecha"
    lineas = uploaded_file.getvalue().decode('latin-1').splitlines()
    
    for line in lineas:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3: continue
        cod = parts[0]
        try:
            if cod == '5011' and len(parts) >= 6: fecha_registro = f"{parts[3].zfill(2)}/{parts[4].zfill(2)}/{parts[5]}"
            elif cod == '5406': data_iri.append({'Progresiva': float(parts[1]), 'IRI_Full': float(parts[4]) if len(parts) > 4 else float(parts[-1])})
            elif cod == '5280': data_gps.append({'Progresiva': float(parts[1]), 'Lat': float(parts[5]), 'Lon': float(parts[6])})
            elif cod == '5416':
                ev = parts[2].replace('"', '').replace("'", "").strip().upper()
                if ev: data_eventos.append({'Progresiva': float(parts[1]), 'Evento': ev})
        except: continue

    df_iri = pd.DataFrame(data_iri).sort_values('Progresiva')
    df_gps = pd.DataFrame(data_gps).sort_values('Progresiva')
    df_evt_raw = pd.DataFrame(data_eventos).sort_values('Progresiva') if data_eventos else pd.DataFrame(columns=['Progresiva', 'Evento'])

    if df_iri.empty or df_gps.empty: return None, None, None

    eventos_tramo = ['P', 'O']
    eventos_puntuales = ['G', 'L', 'S', 'B']
    eventos_objetivo = eventos_tramo + eventos_puntuales
    
    auditoria = {'rebotes': 0, 'puentes': 0, 'obras': 0, 'alertas': [], 'detalle_rebotes': {}}

    # 1. FILTRO CON MEMORIA E INDIFERENCIA
    if not df_evt_raw.empty:
        u_puntual = umbral_puntual / 1000.0
        u_tramo = 0.005 
        evt_validos = []
        memoria = {}
        for _, row in df_evt_raw.iterrows():
            letra, prog = row['Evento'], row['Progresiva']
            
            if letra not in eventos_objetivo:
                evt_validos.append(row)
                continue

            umbral = u_tramo if letra in eventos_tramo else u_puntual
            
            if letra not in memoria:
                evt_validos.append(row); memoria[letra] = prog
            else:
                dist = prog - memoria[letra]
                if dist > umbral:
                    evt_validos.append(row); memoria[letra] = prog
                else: 
                    auditoria['rebotes'] += 1
                    auditoria['detalle_rebotes'][letra] = auditoria['detalle_rebotes'].get(letra, 0) + 1
        df_evt = pd.DataFrame(evt_validos)
    else: 
        df_evt = df_evt_raw

    # SINCRONIZAR
    df_final = pd.merge_asof(df_iri, df_gps, on='Progresiva', direction='nearest').drop_duplicates('Progresiva')
    df_final = pd.merge_asof(df_final, df_evt, on='Progresiva', direction='nearest', tolerance=0.01)

    # 2. MÁQUINA DE ESTADOS
    mask_tramo = pd.Series(False, index=df_final.index)
    limite_p_km = max_puente_m / 1000.0
    auto_cierre = 0.030 
    
    for tipo in eventos_tramo:
        indices = df_final[df_final['Evento'] == tipo].index
        en_tramo, p_ini = False, 0
        for idx in indices:
            p_act = df_final.loc[idx, 'Progresiva']
            if not en_tramo:
                p_ini, en_tramo = p_act, True
            else:
                dist = p_act - p_ini
                if tipo == 'P' and dist > limite_p_km:
                    mask_tramo |= (df_final['Progresiva'] >= p_ini) & (df_final['Progresiva'] <= p_ini + auto_cierre)
                    auditoria['alertas'].append(f"⚠️ Puente en km {p_ini:.3f} excedió {max_puente_m}m. Auto-cerrado.")
                    p_ini = p_act
                else:
                    mask_tramo |= (df_final['Progresiva'] >= p_ini) & (df_final['Progresiva'] <= p_act)
                    en_tramo = False
                    if tipo == 'P': auditoria['puentes'] += 1
                    if tipo == 'O': 
                        auditoria['obras'] += 1
                        if dist > 2.0: auditoria['alertas'].append(f"⚠️ Obra extensa ({dist:.2f} km) detectada desde km {p_ini:.3f}.")
        if en_tramo:
            fin = p_ini + auto_cierre if tipo == 'P' else df_final['Progresiva'].max()
            mask_tramo |= (df_final['Progresiva'] >= p_ini) & (df_final['Progresiva'] <= fin)
            auditoria['alertas'].append(f"🔴 {'Puente' if tipo=='P' else 'Obra'} sin cerrar desde km {p_ini:.3f}.")

    mask_p = df_final['Evento'].isin(eventos_puntuales)
    mask_p_ext = mask_p | mask_p.shift(-1).fillna(False) | mask_p.shift(1).fillna(False)
    mask_t_ext = mask_tramo | mask_tramo.shift(-1).fillna(False) | mask_tramo.shift(1).fillna(False)
    
    df_final['Estado'] = np.where(mask_p_ext | mask_t_ext, 'Excluido', '')
    df_final['Ruta'], df_final['Sentido'], df_final['Fecha_Registro'] = ruta, sentido, fecha_registro

    df_agrupado = None
    if str(intervalo_str).strip().isdigit() and int(intervalo_str) > 0:
        int_km = int(intervalo_str) / 1000.0
        df_calc = df_final[df_final['Estado'] != 'Excluido'].copy()
        df_calc['Bloque'] = (df_calc['Progresiva'] // int_km) * int_km
        df_agrupado = df_calc.groupby('Bloque').agg(
            Progresiva=('Bloque', 'first'), Tramo_Final=('Progresiva', 'max'),
            Lat=('Lat', 'first'), Lon=('Lon', 'first'), IRI_Full=('IRI_Full', 'mean'),
            Ruta=('Ruta', 'first'), Sentido=('Sentido', 'first'), Fecha_Registro=('Fecha_Registro', 'first')
        ).reset_index(drop=True)

    return df_final, df_agrupado, auditoria

# --- UI STREAMLIT ---
with st.form("form_main"):
    c1, c2, c3, c4 = st.columns(4)
    with c1: arc = st.file_uploader("1. Archivo RSP", type=["rsp", "txt"])
    with c2: rut = st.text_input("2. Ruta (ej. PE-3N)")
    with c3: sen = st.selectbox("3. Sentido", ["Creciente", "Decreciente"])
    with c4: int_rep = st.text_input("4. Intervalo (m)", value="200")
    
    with st.expander("⚙️ Configuraciones de Tolerancia y Auditoría"):
        col_a, col_b = st.columns(2)
        with col_a: umbral_rebote = st.number_input("Umbral Anti-rebote (m)", 0, 100, 30)
        with col_b: limite_puente = st.number_input("Límite max. Puente (m)", 100, 2000, 800)

    # Capturamos el botón aquí adentro
    submit_btn = st.form_submit_button("Procesar Datos")

# Todo el procesamiento visual y de descargas está aquí afuera para evitar el error
if submit_btn and arc and rut:
    with st.spinner('Aplicando algoritmos de exclusión...'):
        df_cru, df_agr, audit = procesar_streamlit(arc, rut, sen, int_rep, umbral_rebote, limite_puente)
        
        if df_cru is not None:
            # REPORTE DE AUDITORÍA VISUAL
            st.subheader("📊 Reporte de Auditoría")
            m1, m2, m3 = st.columns(3)
            m1.metric("Puentes Procesados", audit['puentes'])
            m2.metric("Obras Procesadas", audit['obras'])
            m3.metric("Rebotes Filtrados", audit['rebotes'])
            
            # Detalle de rebotes (Solo se muestra si hubo errores del operador)
            if audit['rebotes'] > 0:
                detalle = ", ".join([f"{cant} en '{letra}'" for letra, cant in audit['detalle_rebotes'].items()])
                st.caption(f"🔍 Detalle de rebotes corregidos: {detalle}")
            
            if audit['alertas']:
                for al in audit['alertas']:
                    if "🔴" in al: st.error(al)
                    else: st.warning(al)
            else:
                st.success("✅ Tramos lógicos consistentes. Sin alertas de desfase.")

            # DESCARGAS
            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            nb = arc.name.split('.')[0]
            with col_d1:
                st.download_button("⬇️ Descargar Crudo (SIG)", data=df_cru.to_csv(index=False).encode('utf-8'), file_name=f"{nb}_crudo.csv", mime='text/csv')
            if df_agr is not None:
                with col_d2:
                    st.download_button(f"⬇️ Descargar Resumen ({int_rep}m)", data=df_agr.to_csv(index=False).encode('utf-8'), file_name=f"{nb}_resumen.csv", mime='text/csv')
