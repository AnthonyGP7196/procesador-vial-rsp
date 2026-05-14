import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="RSP Multi-Archivo PRO", layout="wide")
st.title("🛣️ Procesador Multitrack Dynatest Mark III")
st.markdown("Carga masiva, auditoría de eventos y unificación de inventarios georreferenciados.")

def procesar_lote_streamlit(lista_archivos, ruta, sentido, intervalo_str, umbral_puntual, max_puente_m):
    lista_dfs = []
    audit = {'rebotes': 0, 'puentes': 0, 'obras': 0, 'alertas': [], 'detalle_rebotes': {}}
    
    # EL CAMBIO CLAVE: 'O' pasa a ser puntual
    eventos_tramo = ['P']
    eventos_puntuales = ['G', 'L', 'S', 'B', 'O']
    eventos_objetivo = eventos_tramo + eventos_puntuales

    for arc in lista_archivos:
        nombre_arch = arc.name
        data_iri, data_gps, data_eventos = [], [], []
        fecha_reg = "Sin Fecha"
        lineas = arc.getvalue().decode('latin-1').splitlines()
        
        for line in lineas:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3: continue
            cod = parts[0]
            try:
                if cod == '5011' and len(parts) >= 6: fecha_reg = f"{parts[3].zfill(2)}/{parts[4].zfill(2)}/{parts[5]}"
                elif cod == '5406': data_iri.append({'Progresiva': float(parts[1]), 'IRI_Full': float(parts[4]) if len(parts) > 4 else float(parts[-1])})
                elif cod == '5280': data_gps.append({'Progresiva': float(parts[1]), 'Lat': float(parts[5]), 'Lon': float(parts[6])})
                elif cod == '5416':
                    ev = parts[2].replace('"', '').replace("'", "").strip().upper()
                    if ev: data_eventos.append({'Progresiva': float(parts[1]), 'Evento': ev})
            except: continue

        df_iri = pd.DataFrame(data_iri).sort_values('Progresiva')
        df_gps = pd.DataFrame(data_gps).sort_values('Progresiva')
        df_evt_raw = pd.DataFrame(data_eventos).sort_values('Progresiva') if data_eventos else pd.DataFrame(columns=['Progresiva', 'Evento'])

        if df_iri.empty or df_gps.empty: continue

        # 1. FILTRO CON MEMORIA
        if not df_evt_raw.empty:
            u_p, u_t = umbral_puntual / 1000.0, 0.005 
            evt_ok, memoria = [], {}
            for _, row in df_evt_raw.iterrows():
                l, p = row['Evento'], row['Progresiva']
                if l not in eventos_objetivo: 
                    evt_ok.append(row); continue
                umbral = u_t if l in eventos_tramo else u_p
                if l not in memoria or (p - memoria[l]) > umbral:
                    evt_ok.append(row); memoria[l] = p
                else: 
                    audit['rebotes'] += 1
                    audit['detalle_rebotes'][l] = audit['detalle_rebotes'].get(l, 0) + 1
            df_evt = pd.DataFrame(evt_ok)
        else: df_evt = df_evt_raw

        # Auditoría de Obras
        if not df_evt.empty:
            audit['obras'] += df_evt[df_evt['Evento'] == 'O'].shape[0]

        df_f = pd.merge_asof(df_iri, df_gps, on='Progresiva', direction='nearest').drop_duplicates('Progresiva')
        df_f = pd.merge_asof(df_f, df_evt, on='Progresiva', direction='nearest', tolerance=0.01)

        # 2. MÁQUINA DE ESTADOS (Solo Puentes)
        mask_t = pd.Series(False, index=df_f.index)
        limite_p_km = max_puente_m / 1000.0
        
        for tipo in eventos_tramo:
            indices = df_f[df_f['Evento'] == tipo].index
            en_t, p_i = False, 0
            for idx in indices:
                p_a = df_f.loc[idx, 'Progresiva']
                if not en_t: p_i, en_t = p_a, True
                else:
                    dist = p_a - p_i
                    if tipo == 'P' and dist > limite_p_km:
                        mask_t |= (df_f['Progresiva'] >= p_i) & (df_f['Progresiva'] <= p_i + 0.030)
                        audit['alertas'].append(f"[{nombre_arch}] ⚠️ Puente en km {p_i:.3f} auto-cerrado (> {max_puente_m}m).")
                        p_i = p_a
                    else:
                        mask_t |= (df_f['Progresiva'] >= p_i) & (df_f['Progresiva'] <= p_a)
                        en_t = False
                        if tipo == 'P': audit['puentes'] += 1
            if en_t:
                fin = p_i + 0.030 if tipo == 'P' else df_f['Progresiva'].max()
                mask_t |= (df_f['Progresiva'] >= p_i) & (df_f['Progresiva'] <= fin)
                audit['alertas'].append(f"[{nombre_arch}] 🔴 {'Puente' if tipo=='P' else 'Obra'} sin cerrar.")

        mask_p = df_f['Evento'].isin(eventos_puntuales)
        mask_p_ext = mask_p | mask_p.shift(-1).fillna(False) | mask_p.shift(1).fillna(False)
        mask_t_ext = mask_t | mask_t.shift(-1).fillna(False) | mask_t.shift(1).fillna(False)
        
        df_f['Estado'] = np.where(mask_p_ext | mask_t_ext, 'Excluido', '')
        df_f['Ruta'], df_f['Sentido'], df_f['Archivo_Origen'], df_f['Fecha_Registro'] = ruta, sentido, nombre_arch, fecha_reg
        lista_dfs.append(df_f)

    if not lista_dfs: return None, None, None

    # UNIFICAR Y ORDENAR
    df_consolidado = pd.concat(lista_dfs).sort_values(['Progresiva', 'Archivo_Origen'])

    df_agrupado = None
    if str(intervalo_str).strip().isdigit() and int(intervalo_str) > 0:
        int_km = int(intervalo_str) / 1000.0
        df_calc = df_consolidado[df_consolidado['Estado'] != 'Excluido'].copy()
        df_calc['Bloque'] = (df_calc['Progresiva'] // int_km) * int_km
        df_agrupado = df_calc.groupby('Bloque').agg(
            Progresiva=('Bloque', 'first'), Tramo_Final=('Progresiva', 'max'),
            Lat=('Lat', 'first'), Lon=('Lon', 'first'), IRI_Full=('IRI_Full', 'mean'),
            Ruta=('Ruta', 'first'), Sentido=('Sentido', 'first'), Archivo_Referencia=('Archivo_Origen', 'first')
        ).reset_index(drop=True)

    return df_consolidado, df_agrupado, audit

# --- UI STREAMLIT ---
with st.form("form_main"):
    c1, c2, c3, c4 = st.columns(4)
    with c1: arcs = st.file_uploader("1. Archivos RSP", type=["rsp", "txt"], accept_multiple_files=True)
    with c2: rut = st.text_input("2. Ruta (ej. PE-3N)")
    with c3: sen = st.selectbox("3. Sentido", ["Creciente", "Decreciente"])
    with c4: int_rep = st.text_input("4. Intervalo (m)", value="200")
    
    with st.expander("⚙️ Calibración de Máquina de Estados"):
        col_a, col_b = st.columns(2)
        with col_a: umbral_rebote = st.number_input("Umbral Anti-rebote (m)", 0, 100, 30)
        with col_b: limite_puente = st.number_input("Límite max. Puente (m)", 100, 2000, 800)

    submit_btn = st.form_submit_button("Procesar y Unificar Datos")

if submit_btn and arcs and rut:
    with st.spinner(f'Procesando {len(arcs)} archivos y unificando progresivas...'):
        df_cru, df_agr, audit = procesar_lote_streamlit(arcs, rut, sen, int_rep, umbral_rebote, limite_puente)
        
        if df_cru is not None:
            st.subheader(f"📊 Auditoría Global ({len(arcs)} Archivos)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Puentes Totales", audit['puentes'])
            m2.metric("Marcas de Obra (O)", audit['obras'])
            m3.metric("Rebotes Filtrados", audit['rebotes'])
            
            if audit['rebotes'] > 0:
                detalle = ", ".join([f"{cant} en '{letra}'" for letra, cant in audit['detalle_rebotes'].items()])
                st.caption(f"🔍 Detalle de rebotes: {detalle}")
            
            if audit['alertas']:
                for al in audit['alertas']:
                    if "🔴" in al: st.error(al)
                    else: st.warning(al)
            else:
                st.success("✅ Todos los archivos procesados sin alertas de desfase en puentes.")

            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            n_salida = f"{rut}_{sen}_Consolidado"
            with col_d1:
                st.download_button("⬇️ Descargar Crudo Unificado", data=df_cru.to_csv(index=False).encode('utf-8'), file_name=f"{n_salida}_Crudo.csv", mime='text/csv')
            if df_agr is not None:
                with col_d2:
                    st.download_button(f"⬇️ Descargar Resumen Unificado ({int_rep}m)", data=df_agr.to_csv(index=False).encode('utf-8'), file_name=f"{n_salida}_{int_rep}m.csv", mime='text/csv')
