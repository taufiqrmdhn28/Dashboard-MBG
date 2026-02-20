import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import pyvis.network as net
import streamlit.components.v1 as components
import matplotlib.colors as mcolors
import math
import json
import numpy as np

# Cek Ketersediaan Library Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ----------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Socio-Ekonomi I-O", 
    layout="wide",
    page_icon="📈"
)

st.title("Analisis Data Socio-Ekonomi Input-Output")
st.caption("Tools Analisis Socio-Ekonomi Berdasarkan Data Input-Output")
st.write("Unggah file Excel (xlsx) atau CSV (Matriks N x N) untuk memulai.")

# ----------------------------------------------------------------------
# 2. FUNGSI-FUNGSI INTI
# ----------------------------------------------------------------------

@st.cache_data
def load_raw_data(uploaded_file, header_row_index):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=header_row_index, sep=None, engine='python')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, header=header_row_index)
        else: return None
        return df
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None

def get_smart_column_index(df):
    best_idx = 0
    found_text_col = False
    for i, col in enumerate(df.columns):
        sample_values = df[col].astype(str).head(5).tolist()
        non_empty = [s for s in sample_values if s.lower() != 'nan' and s.strip() != '']
        if not non_empty: continue
        is_number = all(s.replace('.','').replace(',','').isdigit() for s in non_empty)
        if not is_number:
            col_str = str(col).lower()
            if any(k in col_str for k in ['produk', 'nama', 'sektor', 'komoditas', 'category', 'code']):
                return i
            if not found_text_col:
                best_idx = i
                found_text_col = True
    return best_idx

def process_data_matrix(df, label_column_index):
    try:
        raw_names = df.iloc[:, label_column_index].astype(str).str.strip().tolist()
        valid_indices = [i for i, name in enumerate(raw_names) if name.lower() != 'nan' and name != '']
        sector_names = [raw_names[i] for i in valid_indices]
        
        seen = {}
        unique_names = []
        for name in sector_names:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        
        n_sectors = len(unique_names)
        start_col = label_column_index + 1
        df_numeric = df.iloc[valid_indices, start_col:]
        
        if df_numeric.shape[1] >= n_sectors:
            df_numeric = df_numeric.iloc[:, :n_sectors]
        
        df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if df_numeric.shape[0] == n_sectors and df_numeric.shape[1] == n_sectors:
            df_numeric.index = [str(x) for x in unique_names]
            df_numeric.columns = [str(x) for x in unique_names]
            df_numeric[df_numeric < 0] = 0 
            return df_numeric
        else:
            st.error(f"Gagal sinkronisasi Matriks. Baris: {n_sectors}, Kolom: {df_numeric.shape[1]}.")
            return None
    except Exception as e:
        st.error(f"Error proses data: {e}")
        return None

@st.cache_resource
def build_graph(_df, graph_type, use_weighted_mode, binarize_threshold, remove_loops):
    df_processed = _df.copy()
    if graph_type == 'Undirected':
        df_processed = df_processed.add(df_processed.T, fill_value=0)

    if not use_weighted_mode:
        df_processed = (df_processed >= binarize_threshold).astype(int)

    create_using = nx.DiGraph() if graph_type == 'Directed' else nx.Graph()
    G = nx.from_pandas_adjacency(df_processed, create_using=create_using)
    
    if remove_loops:
        G.remove_edges_from(nx.selfloop_edges(G))
    return G

def generate_distinct_colors(n):
    colors = list(mcolors.CSS4_COLORS.values())
    bad_colors = ['#FFFFFF', '#F0F8FF', '#F5F5F5', '#000000', '#FFFFF0', '#FFF5EE', '#FFFAFA', '#F8F8FF', '#DCDCDC', '#A9A9A9', '#F0FFF0', '#FFFFE0', '#D3D3D3', '#C0C0C0']
    colors = [c for c in colors if c not in bad_colors]
    if n <= len(colors): return colors[:n]
    return (colors * (n // len(colors) + 1))[:n]

def run_shock_simulation(G, trigger_node, shock_percentage):
    impact_scores = {node: 0.0 for node in G.nodes()}
    impact_scores[trigger_node] = shock_percentage
    
    if G.is_directed():
        downstream_nodes = list(G.successors(trigger_node))
        for node in downstream_nodes:
            try:
                edge_data = G.get_edge_data(trigger_node, node) if G.has_edge(trigger_node, node) else {}
            except: edge_data = {}
            
            input_from_trigger = edge_data.get('weight', 0)
            try: total_input = G.in_degree(node, weight='weight')
            except: total_input = 0
            
            if total_input > 0:
                dependency_ratio = input_from_trigger / total_input
                impact_scores[node] = dependency_ratio * shock_percentage
    else:
        neighbors = list(G.neighbors(trigger_node))
        for node in neighbors:
            try: edge_data = G.get_edge_data(trigger_node, node)
            except: edge_data = {}
            w = edge_data.get('weight', 0)
            total_w = G.degree(node, weight='weight')
            if total_w > 0:
                impact_scores[node] = (w / total_w) * shock_percentage
    return impact_scores

# ----------------------------------------------------------------------
# 3. FUNGSI AI
# ----------------------------------------------------------------------
def generate_data_context(df, G):
    if df is None or G is None: return "Belum ada data."
    num_nodes = G.number_of_nodes(); num_edges = G.number_of_edges()
    density = nx.density(G)
    deg = nx.degree_centrality(G)
    top_5 = sorted(deg.items(), key=lambda x:x[1], reverse=True)[:5]
    top_text = "\n".join([f"- {s}: {v:.4f}" for s,v in top_5])
    return f"""DATA EKONOMI:\n- Sektor: {num_nodes}\n- 5 Sektor Kunci:\n{top_text}"""

def ask_ollama(model_name, messages):
    try:
        response = ollama.chat(model=model_name, messages=messages)
        if hasattr(response, 'message'): return response.message.content
        elif 'message' in response: return response['message']['content']
        else: return str(response)
    except Exception as e: return f"Error AI: {str(e)}"

# ----------------------------------------------------------------------
# 4. SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.header("1. Unggah Data")

def reset_all_state():
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.cache_data.clear(); st.cache_resource.clear()

uploaded_file = st.sidebar.file_uploader("Pilih file", type=["xlsx", "csv"], on_change=reset_all_state)

G = None; df_matrix = None; use_weighted = True 

if uploaded_file is not None:
    st.sidebar.header("2. Pengaturan Data")
    header_row = st.sidebar.number_input("Baris Header (0-based)", min_value=0, value=0)
    df_raw = load_raw_data(uploaded_file, header_row)
    
    if df_raw is not None:
        c_idx = get_smart_column_index(df_raw)
        label_col = st.sidebar.selectbox("Pilih Kolom Nama Kategori:", df_raw.columns, index=c_idx)
        df_matrix = process_data_matrix(df_raw, df_raw.columns.get_loc(label_col))
        
        if df_matrix is not None:
            st.sidebar.success(f"✅ Matriks: {df_matrix.shape}")
            st.sidebar.header("3. Opsi Pra-Pemrosesan")
            graph_type = st.sidebar.radio("Tipe Graf", ('Directed', 'Undirected'), index=0)
            use_weighted = st.sidebar.checkbox("Weighted", True)
            max_val = float(df_matrix.values.max())
            bin_thresh = 0.0
            if not use_weighted: bin_thresh = st.sidebar.slider("Threshold", 0.0, max_val, 0.0)
            remove_loops = st.sidebar.checkbox("Hapus Self-Loops", value=True)
            
            with st.spinner("Membangun Graf..."):
                G = build_graph(df_matrix, graph_type, use_weighted, bin_thresh, remove_loops)
                st.session_state['graph'] = G
                st.session_state['is_weighted'] = use_weighted

st.sidebar.divider()
st.sidebar.header("🤖 AI Config")
selected_model = None
if OLLAMA_AVAILABLE:
    try:
        models = ollama.list()
        m_names = [m.model for m in models.models] if hasattr(models, 'models') else [m['name'] for m in models['models']]
        if m_names: selected_model = st.sidebar.selectbox("Model LLM:", m_names)
    except: st.sidebar.error("Gagal koneksi.")

st.sidebar.divider()
if st.sidebar.button("⚠️ HAPUS DATA & RESET", type="primary"):
    reset_all_state(); st.rerun()

# ----------------------------------------------------------------------
# 5. TAMPILAN UTAMA
# ----------------------------------------------------------------------
if 'graph' in st.session_state and df_matrix is not None:
    G = st.session_state['graph']
    is_weighted = st.session_state['is_weighted']
    
    tab_data, tab_centrality, tab_community, tab_viz, tab_impact, tab_ai = st.tabs([
        "Data", "Sentralitas", "Komunitas", "Visualisasi Network", "Laporan Dampak", "AI Agent"
    ])

    # --- TAB 1: DATA & HEATMAP ---
    with tab_data:
        st.subheader("Matriks Data (Preview)")
        st.dataframe(df_matrix.head(10))
        st.markdown("---")
        st.subheader("Heatmap Transaksi Antar Sektor")
        n_top = st.slider("Jumlah Sektor Teratas (Top-N):", 5, 50, 20)
        try:
            sector_sums = df_matrix.sum(axis=1)
            sorted_idx = sector_sums.sort_values(ascending=False).index
            top_n_idx = sorted_idx[:n_top]
            df_heatmap = df_matrix.loc[top_n_idx, top_n_idx]
            fig = px.imshow(df_heatmap, labels=dict(x="Pembeli", y="Penjual", color="Nilai"), aspect="auto", color_continuous_scale="Blues")
            fig.update_layout(height=600, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Gagal menampilkan heatmap: {e}")

    # --- TAB 2: SENTRALITAS (FIXED SORTING & GRAFIK) ---
    with tab_centrality:
        measure = st.selectbox("Pilih Metrik Sentralitas:", ["Degree Centrality", "Betweenness Centrality", "Eigenvector Centrality"])
        
        if st.button("Hitung Sentralitas"):
            w = 'weight' if is_weighted else None
            if measure == "Degree Centrality": res = nx.degree_centrality(G)
            elif measure == "Betweenness Centrality": res = nx.betweenness_centrality(G, weight=w)
            else: res = nx.eigenvector_centrality(G, max_iter=1000, weight=w)
            
            # 1. Pastikan Urut dari Terbesar ke Terkecil
            df_res = pd.DataFrame.from_dict(res, orient='index', columns=['Score']).sort_values('Score', ascending=False)
            
            # 2. Grafik dengan Plotly (Agar Urutan Sesuai Dataframe & Rapi)
            st.write("📊 **Grafik Visual**")
            fig = px.bar(
                df_res.head(20), 
                x=df_res.head(20).index, 
                y='Score',
                labels={'index': 'Sektor', 'Score': 'Skor Sentralitas'},
                color='Score'
            )
            # Paksa sumbu X agar mengikuti urutan Total Descending (Besar ke Kecil)
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # 3. Tabel Lengkap
            st.write("📋 **Tabel Peringkat Lengkap**")
            st.dataframe(df_res, use_container_width=True)

    # --- TAB 3: KOMUNITAS (FIXED: TABLE VIEW) ---
    with tab_community:
        if st.button("Cari Komunitas (Louvain)"):
            c = list(nx.community.louvain_communities(G, weight='weight' if is_weighted else None))
            st.success(f"Ditemukan {len(c)} Komunitas.")
            
            # Format Data Menjadi Tabel
            comm_data = []
            for i, comm in enumerate(c):
                anggota_sorted = sorted(list(comm))
                anggota_str = ", ".join([str(x) for x in anggota_sorted])
                comm_data.append({
                    "ID": i + 1,
                    "Jumlah Anggota": len(comm),
                    "Anggota Komunitas": anggota_str
                })
            
            df_comm = pd.DataFrame(comm_data)
            st.dataframe(df_comm, use_container_width=True, hide_index=True)

    # --- TAB 4: VISUALISASI NETWORK ---
    with tab_viz:
        st.subheader("Visualisasi Interaktif")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            node_list = sorted([str(n) for n in G.nodes])
            focus = st.selectbox("Fokus:", ["Semua"] + node_list)
            sz_scale = st.slider("Ukuran Node:", 10, 200, 60)
            show_labels = st.checkbox("Tampilkan Label", value=True)
            
        with c2:
            edge_scale = st.slider("Tebal Garis:", 0.1, 5.0, 0.2)
            w_vals = [d.get('weight', 0) for u, v, d in G.edges(data=True)] if is_weighted else []
            v_max = max(w_vals) if w_vals else 1.0
            v_thresh = st.slider("Filter Nilai <", 0.0, float(v_max), float(v_max*0.01))
        
        with c3:
            viz_mode = st.radio("Mode:", ["Normal", "Simulasi Kenaikan/Penurunan"])
            layout_mode = st.radio("Layout:", ["Fisika (Otomatis)", "Lingkaran Terurut"])
        
        impact_results = {}
        trigger = None
        shock_val = 0
        
        if viz_mode == "Simulasi Kenaikan/Penurunan":
            st.info("💡 Geser Kiri (-) untuk Penurunan, Geser Kanan (+) untuk Kenaikan.")
            c_sim1, c_sim2 = st.columns(2)
            with c_sim1: trigger = st.selectbox("Sektor Pemicu:", node_list)
            with c_sim2: shock_val = st.slider("Perubahan (%)", -100, 100, 0)
            
            if st.button("Jalankan Simulasi") and shock_val != 0:
                 impact_results = run_shock_simulation(G, trigger, shock_val)
                 st.session_state['last_sim'] = {'res': impact_results, 'trig': trigger, 'val': shock_val}
            
            if 'last_sim' in st.session_state:
                impact_results = st.session_state['last_sim']['res']
                trigger = st.session_state['last_sim']['trig']
                shock_val = st.session_state['last_sim']['val']

        with st.spinner("Rendering..."):
            try:
                G_viz = G.copy()
                if is_weighted and v_thresh > 0:
                    rem = [(u, v) for u, v, d in G_viz.edges(data=True) if d.get('weight', 0) < v_thresh]
                    G_viz.remove_edges_from(rem)
                
                if focus != "Semua":
                    nb = list(G_viz.neighbors(focus)) if not G.is_directed() else list(G_viz.successors(focus)) + list(G_viz.predecessors(focus))
                    G_viz = G_viz.subgraph(set(nb + [focus]))
                else:
                    G_viz.remove_nodes_from(list(nx.isolates(G_viz)))

                nt = net.Network(height="700px", width="100%", bgcolor="#FFFFFF", font_color="#262730", cdn_resources="in_line")
                
                # Layout Config
                positions = {}
                if layout_mode == "Lingkaran Terurut":
                    nt.toggle_physics(False)
                    deg_all = dict(G_viz.degree(weight='weight')) if is_weighted else dict(G_viz.degree())
                    sorted_nodes = sorted(G_viz.nodes(), key=lambda n: deg_all.get(n, 0), reverse=True)
                    pos_shell = nx.shell_layout(G_viz, nlist=[sorted_nodes])
                    SCALE_FACTOR = 2500 
                    positions = {k: (v[0]*SCALE_FACTOR, v[1]*SCALE_FACTOR) for k, v in pos_shell.items()}
                else:
                    nt.force_atlas_2based()
                
                # --- NODE RENDERING ---
                deg = dict(G_viz.degree(weight='weight')) if is_weighted else dict(G_viz.degree())
                vals = list(deg.values()); min_s, max_s = (min(vals), max(vals)) if vals else (0,1)
                
                cols = generate_distinct_colors(len(G.nodes))
                cmap = {n:c for n,c in zip(list(G.nodes()), cols)}
                
                for n in G_viz.nodes():
                    nid = str(n)
                    sc = deg.get(n, 0)
                    sz = 10 + ((sc - min_s)/(max_s - min_s + 0.001) * sz_scale)
                    
                    # 1. Tentukan Warna Dasar
                    if viz_mode == "Simulasi Kenaikan/Penurunan":
                        # Logic: Default Abu-abu, kecuali terdampak
                        col = "#e0e0e0" # Abu-abu
                        title = f"{nid}\nTotal: {sc:,.0f}"

                        if nid in impact_results:
                            imp = impact_results[nid]
                            if abs(imp) > 0.001:
                                if nid == str(trigger): 
                                    col = "#000000"; sz *= 1.3; title += f"\nPEMICU: {imp:+.1f}%" # Hitam
                                elif imp < 0: 
                                    col = "#FF0000"; title += f"\nDampak: {imp:.2f}%" # Merah
                                else: 
                                    col = "#28a745"; title += f"\nDampak: +{imp:.2f}%" # Hijau
                    else:
                        # Logic: Warna Normal (Warna-warni Kategori)
                        col = cmap.get(n, "#007bff")
                        title = f"{nid}\nTotal: {sc:,.0f}"

                    x, y = None, None
                    if nid in positions: x, y = positions[nid][0], positions[nid][1]
                    lbl = nid if show_labels else " "
                    nt.add_node(nid, label=lbl, size=sz, color=col, title=title, x=x, y=y)

                # --- EDGE RENDERING ---
                for u, v, d in G_viz.edges(data=True):
                    w = d.get('weight', 1)
                    wid = 0.5 + (math.log(w + 1) * edge_scale)
                    ecol = "#cccccc" # Default Abu-abu
                    nt.add_edge(u, v, width=wid, color=ecol)
                
                components.html(nt.generate_html(), height=750)

            except Exception as e: st.error(f"Gagal visualisasi: {e}")

    # --- TAB 5: LAPORAN DAMPAK (FIXED SORTING BY CHANGE %) ---
    with tab_impact:
        st.subheader("📋 Laporan Detail Dampak Simulasi")
        if 'last_sim' in st.session_state:
            sim = st.session_state['last_sim']
            impacts = sim['res']; trigger = sim['trig']; val = sim['val']
            try: totals = df_matrix.sum(axis=1)
            except: totals = pd.Series(0, index=df_matrix.index)
            rows = []
            for sec, pct in impacts.items():
                if abs(pct) > 0.001: 
                    base = totals.get(str(sec), 0); chg = base * (pct / 100.0)
                    stt = "Pemicu" if str(sec) == str(trigger) else ("Naik (+)" if pct > 0 else "Turun (-)")
                    rows.append({"Sektor": str(sec), "Change (%)": pct, "Nominal": chg, "Status": stt})
            
            df_imp = pd.DataFrame(rows)
            if not df_imp.empty:
                # SORTING PERBAIKAN: Urutkan berdasarkan 'Besaran Change' (Magnitude Absolut)
                # Agar sektor dengan % perubahan terbesar (Pemicu & Dampak Utama) selalu di atas
                df_imp['Abs_Change'] = df_imp['Change (%)'].abs()
                df_imp = df_imp.sort_values(by="Abs_Change", ascending=False).drop(columns=['Abs_Change'])
                
                judul = "Kenaikan" if val > 0 else "Penurunan"
                st.info(f"Skenario: **{judul}** sebesar **{val}%** pada sektor **{trigger}**")
                
                c1, c2 = st.columns(2)
                c1.metric("Sektor Terdampak", len(df_imp))
                tot = df_imp['Nominal'].sum()
                c2.metric("Total Dampak Ekonomi", f"{tot:,.2f}", delta_color="normal")
                
                st.dataframe(df_imp, use_container_width=True)
            else: st.warning("Tidak ada dampak signifikan.")
        else: st.info("Jalankan simulasi di Tab Visualisasi dulu.")
        
    with tab_ai:
        if not selected_model: st.warning("Pilih model di sidebar.")
        else:
            if "msgs" not in st.session_state: st.session_state.msgs = []
            ctx = generate_data_context(df_matrix, G)
            for m in st.session_state.msgs: 
                with st.chat_message(m["role"]): st.markdown(m["content"])
            if q := st.chat_input("Tanya AI..."):
                st.session_state.msgs.append({"role":"user", "content":q})
                with st.chat_message("user"): st.markdown(q)
                with st.spinner("..."):
                    payload = [{"role":"system","content":ctx}] + st.session_state.msgs
                    ans = ask_ollama(selected_model, payload)
                    st.session_state.msgs.append({"role":"assistant", "content":ans})
                    with st.chat_message("assistant"): st.markdown(ans)
            if st.button("Reset"): st.session_state.msgs = []; st.rerun()
else:
    st.info("Upload file di sidebar.")
