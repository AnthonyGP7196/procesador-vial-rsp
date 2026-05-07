import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Dynatest RSP a GIS", layout="wide")
st.title("🛣️ Procesador Vial Dynatest Mark III")
st.markdown("Herramienta de gabinete para filtrar singularidades estructurales y exportar tablas de rugosidad georreferenciadas.")

def procesar_streamlit(uploaded_file, ruta_vial, sentido_vial, intervalo_str):
    data_iri, data_gps, data_eventos = [], [], []
    fecha_reg = "Sin Fecha"
    eventos_excluyentes = ['G', 'L', 'P', 'B', 'S']
    
    lineas = uploaded_file.getvalue().decode('latin-1').splitlines()

    for line in lineas:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3: continue
        cod = parts[0]
        try:
            if cod == '5011' and len(parts) >= 6: 
                fecha_reg = f"{parts[3].zfill(2)}/{parts[4].zfill(2)}/{parts[5]}"
            elif cod == '5406': 
                data_iri.append({'Progresiva': float(parts[1]), 'IRI_Full': float(parts[4]) if len(parts) > 4 else float(parts[-1])})
            elif cod == '5280': 
                data_gps.append({'Progresiva': float(parts[1]), 'Lat': float(parts[5]), 'Lon': float(parts[6])})
            elif cod == '5416':
                ev = parts[2].replace('"', '').replace("'", "").strip().upper()
                if ev: data_eventos.append({'Progresiva': float(parts[1]), 'Evento': ev})
        except: continue

    df_iri = pd.DataFrame(data_iri).sort_values('Progresiva')
    df_gps = pd.DataFrame(data_gps).sort_values('Progresiva')
    df_evt = pd.DataFrame(data_eventos).sort_values('Progresiva') if data_eventos else pd.DataFrame(columns=['Progresiva', 'Evento'])

    if df_iri.empty or df_gps.empty: return None, None

    # Sincronización
    df_final = pd.merge_asof(df_iri, df_gps, on='Progresiva', direction='nearest').drop_duplicates('Progresiva')
    if not df_evt.empty:
        df_final = pd.merge_asof(df_final, df_evt, on='Progresiva', direction='nearest', tolerance=0.01)
    else:
        df_final['Evento'] = np.nan

    # Exclusión
    es_sing = df_final['Evento'].isin(eventos_excluyentes)
    mask = es_sing | es_sing.shift(-1).fillna(False) | es_sing.shift(1).fillna(False)
    df_final['Estado'] = np.where(mask, 'Excluido', '')
    
    # Metadatos
    df_final['Ruta'] = ruta_vial
    df_final['Sentido'] = sentido_vial
    df_final['Fecha_Registro'] = fecha_reg

    # 1. Preparar el Crudo
    cols_crudas = ['Ruta', 'Sentido', 'Progresiva', 'Lat', 'Lon', 'IRI_Full', 'Evento', 'Estado', 'Fecha_Registro']
    df_crudo = df_final[cols_crudas]

    # 2. Preparar el Agrupado (si aplica)
    df_agrupado_resumen = None
    intervalo_limpio = str(intervalo_str).lower().replace('m', '').strip()
    
    if intervalo_limpio and intervalo_limpio != '0':
        try:
            intervalo_m = int(intervalo_limpio)
            intervalo_km = intervalo_m / 1000.0
            
            df_calc = df_final[df_final['Estado'] != 'Excluido'].copy()
            df_calc['Bloque_Nominal'] = (df_calc['Progresiva'] // intervalo_km) * intervalo_km
            
            df_agrup = df_calc.groupby('Bloque_Nominal').agg(
                Progresiva=('Bloque_Nominal', 'first'),
                Tramo_Final=('Progresiva', 'max'), 
                Lat=('Lat', 'first'),
                Lon=('Lon', 'first'),
                IRI_Full=('IRI_Full', 'mean'),
                Ruta=('Ruta', 'first'),
                Sentido=('Sentido', 'first'),
                Fecha_Registro=('Fecha_Registro', 'first')
            ).reset_index(drop=True)
            
            cols_resumen = ['Ruta', 'Sentido', 'Progresiva', 'Tramo_Final', 'Lat', 'Lon', 'IRI_Full', 'Fecha_Registro']
            df_agrupado_resumen = df_agrup(cols_resumen)
        except ValueError:
            pass

    return df_crudo, df_agrupado_resumen

# --- INTERFAZ STREAMLIT ---
with st.form("procesador_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1: arc = st.file_uploader("1. Archivo RSP", type=["rsp", "txt"])
    with c2: rut = st.text_input("2. Ruta (ej. PE-3N)")
    with c3: sen = st.selectbox("3. Sentido", ["Creciente", "Decreciente"])
    with c4: int_rep = st.text_input("4. Intervalo en metros (Dejar vacío para solo crudo)", value="200")
    
    submit_btn = st.form_submit_button("Procesar Datos")

if submit_btn and arc and rut:
    with st.spinner('Evaluando progresivas y aplicando filtros de exclusión...'):
        df_crudo, df_agrup = procesar_streamlit(arc, rut, sen, int_rep)
        
        if df_crudo is not None:
            st.success(f"✅ Procesamiento completado. Fecha de medición extraída: {df_crudo['Fecha_Registro'].iloc[0]}")
            
            # --- SECCIÓN DE DESCARGAS ---
            st.markdown("### Descarga de Archivos para GIS")
            col_desc1, col_desc2 = st.columns(2)
            
            nombre_base = arc.name.split('.')[0]
            
            with col_desc1:
                csv_crudo = df_crudo.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar Perfil Crudo", data=csv_crudo, file_name=f"{nombre_base}_crudo.csv", mime='text/csv')
            
            if df_agrup is not None:
                with col_desc2:
                    csv_agrup = df_agrup.to_csv(index=False).encode('utf-8')
                    int_limpio = int_rep.lower().replace('m', '').strip()
                    st.download_button(f"⬇️ Descargar Resumen @{int_limpio}m", data=csv_agrup, file_name=f"{nombre_base}_{int_limpio}m.csv", mime='text/csv')

            # --- GRÁFICA ---
            st.markdown("### Control de Calidad del Perfil")
            fig, ax = plt.subplots(figsize=(12, 4))
            
            ax.plot(df_crudo['Progresiva'], df_crudo['IRI_Full'], color='lightblue', alpha=0.6, label='IRI Crudo')
            rojos = df_crudo[df_crudo['Evento'].isin(['G', 'L', 'P', 'B', 'S'])]
            ax.scatter(rojos['Progresiva'], rojos['IRI_Full'], color='red', s=40, label='Singularidad (Excluida)')
            
            if df_agrup is not None:
                ax.step(df_agrup['Progresiva'], df_agrup['IRI_Full'], where='post', color='darkblue', linewidth=2, label=f'Tendencia ({int_rep}m)')
            
            ax.set_title(f"Ruta: {rut} ({sen})")
            ax.set_xlabel("Progresiva")
            ax.set_ylabel("IRI (m/km)")
            ax.grid(True, linestyle='--')
            ax.legend()
            st.pyplot(fig)
