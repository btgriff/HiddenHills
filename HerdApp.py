import seaborn as sns
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statistics import mean
from math import pi
import streamlit as st
sns.set_style("white")
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Headless backend — faster server-side rendering, no GUI overhead
from PIL import Image
from highlight_text import fig_text
import urllib.request
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import plotly.express as px
import plotly.figure_factory as ff
from plotly.graph_objects import Layout
from vega_datasets import data
import pycountry
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import sqlite3
import threading
import io as _qio


acros = {
    'Calving Ease Direct':'(CED)\n',
    'Birth Weight':'(BW)\n',
    'Weaning Weight':'(WW)\n',
    'Yearling Weight':'(YW)\n',
    'Average Daily Gain':'(ADG)\n',
    'Dry Matter Intake':'(DMI)\n',
    "Daughter's Milk":'(MILK)\n',
    'Maintenance Energy':'(ME)\n',
    'Heifer Pregnancy':'(HPG)\n',
    'Calving Ease Maternal':'(CEM)\n',
    'Stayability':'(STAY)\n',
    'Marbling':'(MARB)\n',
    'Yield Grade':'(YG)\n',
    'Carcass Weight':'(CW)\n',
    'Rib Eye Area':'(REA)\n',
    'Fat':'',
    'ProS':'',
    'HerdBuilder':'',
    'GridMaster':'',
}

metrics = {
    'Aggregate Measures':[
        'ProS','HerdBuilder','GridMaster'
    ],
    'Maternal Qualities':[
        "Daughter's Milk",'Heifer Pregnancy','Calving Ease Maternal','Calving Ease Direct'
    ],
    'Weights':[
        'Birth Weight','Weaning Weight','Yearling Weight'
    ],
    'Herd Maintenance':[
        'Average Daily Gain','Dry Matter Intake','Maintenance Energy','Stayability'
    ],
    'Slaughter Outcomes':[
        'Marbling','Yield Grade','Carcass Weight','Rib Eye Area','Fat'
    ]
}


def clean_herd_dataset():
    herd_table_col_order = [
        'Field Tag','Reg #','Reg Type','Animal ID','Name','Sex','Birth Date','ProS','ProS_pct','HerdBuilder','HerdBuilder_pct','GridMaster','GridMaster_pct','Calving Ease Direct','Calving Ease Direct_pct','Birth Weight','Birth Weight_pct','Weaning Weight','Weaning Weight_pct','Yearling Weight','Yearling Weight_pct','Average Daily Gain','Average Daily Gain_pct','Dry Matter Intake','Dry Matter Intake_pct',"Daughter's Milk","Daughter's Milk_pct",'Maintenance Energy','Maintenance Energy_pct','Heifer Pregnancy','Heifer Pregnancy_pct','Calving Ease Maternal','Calving Ease Maternal_pct','Stayability','Stayability_pct','Marbling','Marbling_pct','Yield Grade','Yield Grade_pct','Carcass Weight','Carcass Weight_pct','Rib Eye Area','Rib Eye Area_pct','Fat','Fat_pct','$Profit','$Profit_pct','$Ranch','$Ranch_pct','Fertility','Fertility_pct','$Feeder','$Feeder_pct','BrdCds','CowBull',
    ]

    renames = {
        'CED':'Calving Ease Direct',
        'BW':'Birth Weight',
        'WW':'Weaning Weight',
        'YW':'Yearling Weight',
        'ADG':'Average Daily Gain',
        'DMI':'Dry Matter Intake',
        'Milk':"Daughter's Milk",
        'ME':'Maintenance Energy',
        'HPG':'Heifer Pregnancy',
        'CEM':'Calving Ease Maternal',
        'Stay':'Stayability',
        'Marb':'Marbling',
        'YG':'Yield Grade',
        'CW':'Carcass Weight',
        'RE':'Rib Eye Area',
        'BF':'Fat',
        'HB':'HerdBuilder',
        'GM':'GridMaster',
        'CE':'Calving Ease Direct',
        'MM':"Daughter's Milk",
        'STAY':'Stayability',
        'REA':'Rib Eye Area',
    }

    ### Cows
    # cows = pd.read_excel(dirty_data, sheet_name="Cows")
    cows = pd.read_excel("https://github.com/btgriff/HiddenHills/raw/refs/heads/main/Herd.xlsx", sheet_name='Cows')
    cow_cols = cows.iloc[1].values[1:]
    cows = cows.iloc[2:,1:]
    cows.columns = cow_cols
    cows.rename(columns=renames,inplace=True)
    
    cows_raw = cows[cows['Field Tag'].notna()]
    young_cows = cows_raw[cows_raw['ProS'].isna()]
    cows_raw = cows_raw[cows_raw['ProS'].notna()]

    cows_pct = cows.loc[cows_raw.index+1,:]
    
    cows_pct = cows_raw[['Field Tag','Reg #','Reg Type','Animal ID','Name','Sex','Birth Date']].reset_index(drop=True).join(cows_pct.drop(columns=['Field Tag','Reg #','Reg Type','Animal ID','Name','Sex','Birth Date']).reset_index(drop=True).add_suffix('_pct'))
    
    cows_all = cows_raw.merge(cows_pct)
    if len(cows_all)!=len(cows_raw):
        raise ValueError("A cow's row must be incomplete")
    
    cows_all['CowBull'] = 'Cow'
    
    ### Bulls
    
    # bulls = pd.read_excel(dirty_data, sheet_name="Bulls")
    bulls = pd.read_excel("https://github.com/btgriff/HiddenHills/raw/refs/heads/main/Herd.xlsx", sheet_name='Bulls')
    bull_cols = bulls.iloc[2].values[1:]
    bulls = bulls.iloc[4:,1:]
    bulls.columns = bull_cols
    bulls.rename(columns={'RAAA#':'Reg #','RegType':'Reg Type','AnimalID':'Animal ID','DOB':'Birth Date'},inplace=True)
    bulls.rename(columns=renames,inplace=True)
    
    bulls_raw = bulls[bulls['Field Tag'].notna()]
    young_bulls = bulls_raw[bulls_raw['ProS'].isna()]
    bulls_raw = bulls_raw[bulls_raw['ProS'].notna()]
    
    bulls_pct = bulls.loc[bulls_raw.index+1,:]
    
    bulls_pct = bulls_raw[['Field Tag','Reg #','Reg Type','Animal ID','Name','Sex','Birth Date','BrdCds']].reset_index(drop=True).join(bulls_pct.drop(columns=['Field Tag','Reg #','Reg Type','Animal ID','Name','Sex','Birth Date','BrdCds']).reset_index(drop=True).add_suffix('_pct'))
    
    bulls_all = bulls_raw.merge(bulls_pct)
    if len(bulls_all)!=len(bulls_raw):
        raise ValueError("A bull's row must be incomplete")
    
    bulls_all['CowBull'] = 'Bull'
    
    ### Combine
    
    herd = pd.concat([cows_all,bulls_all],ignore_index=True)
    herd['Birth Date'] = pd.to_datetime(herd['Birth Date']).dt.strftime('%Y-%m-%d')
    herd['Field Tag'] = herd['Field Tag'].astype(str)
    return herd[herd_table_col_order]
###########

def add_labels(angles, values, labels, offset, ax, text_colors):

    # This is the space between the end of the bar and the label
    padding = .05

    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, text_col in zip(angles, values, labels, text_colors):
        angle = angle

        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=1.11,
            s=f"{acros[label.replace('\n',' ')]}{label}", 
            ha=alignment, 
            weight = 'bold',
            va="center", 
            rotation=rotation,
            color=text_col,
        )

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)+90
    if angle <= np.pi/2:
        alignment = "center"
        rotation = rotation + 180
    elif 4.3 < angle < np.pi*2:  # 4.71239 is 270 degrees
        alignment = "center"
        rotation = rotation - 180
    else: 
        alignment = "center"
    return rotation, alignment



def scout_report(herd, field_tag):
    plt.close('all')

    ProS = 'ProS'
    HerdBuilder = 'HerdBuilder'
    GridMaster = 'GridMaster'
    
    milk = "Daughter's Milk"
    hpg = 'Heifer Pregnancy'
    cem = 'Calving Ease Maternal'
    ced = 'Calving Ease Direct'
    
    bw = 'Birth Weight'
    ww = 'Weaning Weight'
    yw = 'Yearling Weight'

    adg = 'Average Daily Gain'
    dmi = 'Dry Matter Intake'
    me = 'Maintenance Energy'
    stay = 'Stayability'

    marb = 'Marbling'
    yg = 'Yield Grade'
    cw = 'Carcass Weight'
    rea = 'Rib Eye Area'
    fat = 'Fat'
    
    dfProspect = herd

    raw_valsdf = dfProspect[dfProspect['Field Tag']==field_tag]
    cow_bull = raw_valsdf['CowBull'].values[0]
    dob = raw_valsdf['Birth Date'].values[0]
    reg_tag = raw_valsdf['Reg #'].values[0]
    raw_valsdf_full = dfProspect.copy()
    df_pros = dfProspect

    ######################################################################

    dfRadarMF = dfProspect[dfProspect['Field Tag']==field_tag].reset_index(drop=True)
    # dfRadarMF = dfRadarMF.fillna(0)
    player_full_name = field_tag
    # Define a dictionary to map old column names to new ones
    column_mapping = {
        "Daughter's Milk":'MILK',
        'Heifer Pregnancy':'HPG',
        'Calving Ease Maternal':'CEM',
        'Calving Ease Direct':'CED',
        
        'Birth Weight':'BW',
        'Weaning Weight':'WW',
        'Yearling Weight':'YW',
        
        'Average Daily Gain':'ADG',
        'Dry Matter Intake':'DMI',
        'Maintenance Energy':'ME',
        'Stayability':'STAY',
        
        'Marbling':'MARB',
        'Yield Grade':'YG',
        'Carcass Weight':'CW',
        'Rib Eye Area':'REA',
        'Fat':'FAT'
    }
    raw_vals = raw_valsdf[["Field Tag",
                       ProS,HerdBuilder,GridMaster,milk,hpg,cem,ced,bw,ww,yw,adg,dmi,me,stay,marb,yg,cw,rea,fat
                      ]]
    raw_vals_full = raw_valsdf_full[["Field Tag",
                       ProS,HerdBuilder,GridMaster,milk,hpg,cem,ced,bw,ww,yw,adg,dmi,me,stay,marb,yg,cw,rea,fat
                      ]]
    ll = list(column_mapping.keys())
    ll = [l + '_pct' for l in ll]
    dfRadarMF = dfRadarMF[['Field Tag'] + [ProS+'_pct',HerdBuilder+'_pct',GridMaster+'_pct'] + ll]
    dfRadarMF.rename(columns=column_mapping, inplace=True)
        
    ###########################################################################

    df1 = dfRadarMF.T.reset_index()

    df1.columns = df1.iloc[0] 

    df1 = df1[1:]
    df1 = df1.reset_index()
    df1 = df1.rename(columns={'Field Tag': 'Metric',
                        field_tag: 'Value',
                             'index': 'Group'})

    for i in range(len(df1)):
        if df1['Group'][i] <= 3:
            df1['Group'][i] = 'Aggregate'
        elif df1['Group'][i] <= 3+4:
            df1['Group'][i] = 'Maternal'
        elif df1['Group'][i] <= 3+4+3:
            df1['Group'][i] = 'Weights'
        elif df1['Group'][i] <= 3+4+3+4:
            df1['Group'][i] = 'Herd Maintenance'
        elif df1['Group'][i] <= 3+4+3+4+5:
            df1['Group'][i] = 'Slaughter'

    #####################################################################
    # Grab the group values
    GROUP = df1["Group"].values
    VALUES = df1["Value"].values
    LABELS = df1["Metric"].values
    LABELS = [l.replace('_pct','').replace(' ','\n') for l in LABELS]
    OFFSET = np.pi / 2

    PAD = 2
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)

    offset = 0
    IDXS = []

    GROUPS_SIZE = [3,4,3,4,5]

    print(df1)


    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-.5, 1)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


    COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

    ax.bar(
        ANGLES[IDXS], 1-VALUES, width=WIDTH, color=COLORS,
        edgecolor="#4A2E19", linewidth=1
    )

    GROUP_NAMES = ['\nAggregate', 'Maternal\n', 'Weight\n', 'Herd\nMaintenance\n', '\nSlaughter']
    
    offset = 0
    for group, size in zip(GROUP_NAMES, GROUPS_SIZE):
        # Add line below bars
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-.02] * 50, color="#4A2E19")
    
        # Add group name below the line
        mid = np.mean(x1)
        screen_deg = np.rad2deg(mid + OFFSET) % 360    # where the label sits on screen
        rotation = screen_deg - 90                     # follow the curve of the circle
        if 180 < screen_deg < 360:                     # bottom half would be upside down
            rotation += 180
        ax.text(
            mid, -0.12, group,
            color="#4A2E19", fontsize=10, fontweight="bold",
            ha="center", va="center",
            rotation=rotation, rotation_mode="anchor",
        )
    
        # Add reference lines at 20, 40, 60, 80, and 100
        x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
        for r in [.2, .4, .6, .8, 1]:
            ax.plot(x2, [r] * 50, color="#bebebe", lw=0.8)
    
        offset += size + PAD 
        
    text_cs = []
    text_inv_cs = []
    for i, bar in enumerate(ax.patches):
        pc = 1 - bar.get_height()

        if pc <= 0.1:
            color = ('#01349b', '#d9e3f6')  # Elite
        elif 0.1 < pc <= 0.35:
            color = ('#007f35', '#d9f0e3')  # Above Avg
        elif 0.35 < pc <= 0.66:
            color = ('#9b6700', '#fff2d9')  # Avg
        else:
            color = ('#b60918', '#fddbde')  # Below Avg

        bar.set_color(color[1])
        bar.set_edgecolor(color[0])

        text_cs.append(color[0])
        text_inv_cs.append(color[1])
        

    for i, bar in enumerate(ax.patches):
        value_format = f'{format(100-(bar.get_height() * 100), '.0f')}%\n{round(raw_vals.iloc[0][i+1], 2)}'
        color = text_inv_cs[i]
        face = text_cs[i]

        ax.annotate(value_format,
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.1),
                    ha='center', va='center', size=10, xytext=(0, 8),
                    textcoords='offset points', color=color, zorder=4,
                    bbox=dict(boxstyle="round", fc=face, ec="black", lw=1))

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax, text_cs)

    PAD = 0.02
    ax.text(0.15, 0 + PAD, "100", size=10, color='#4A2E19')
    ax.text(0.15, 0.2 + PAD, "80", size=10, color='#4A2E19')
    ax.text(0.15, 0.4 + PAD, "60", size=10, color='#4A2E19')
    ax.text(0.15, 0.6 + PAD, "40", size=10, color='#4A2E19')
    ax.text(0.15, 0.8 + PAD, "20", size=10, color='#4A2E19')
    ax.text(0.15, 1 + PAD, "0", size=10, color='#4A2E19')

    plt.suptitle(f'{field_tag} ({cow_bull}, {dob}) Percentiles\nReg # {int(reg_tag)}\n ',
                 fontsize=17,
                 fontfamily="DejaVu Sans",
                 va='center',
                color="#4A2E19", #4A2E19
                 fontweight="bold", fontname="DejaVu Sans",
                x=0.5,
                y=.97)

    plt.annotate("DATA CALLOUTS: [Percentile%, Raw Value]\n\nLonger bars indicate higher percentile rankings\nColors indicate broad percentile ranges, see legend",
                 xy = (0, -.09), xycoords='axes fraction',
                ha='left', va='center',
                fontsize=9, fontfamily="DejaVu Sans",
                color="#4A2E19", fontweight="regular", fontname="DejaVu Sans",
                ) 

    ax.set_facecolor('#fbf9f4')
    fig = plt.gcf()
    fig.patch.set_facecolor('#fbf9f4')
    fig.set_size_inches(12, (12*.9)) #length, height
    
    fig_text(
        0.88, 0.055, "<Elite (Top 10%)>\n<Above Average (11-35%)>\n<Average (36-66%)>\n<Below Average (Bottom 35%)>", color="#4A2E19",
        highlight_textprops=[{"color": '#01349b'},
                             {'color' : '#007f35'},
                             {"color" : '#9b6700'},
                             {'color' : '#b60918'},
                            ],
        size=10, fig=fig, ha='right',va='center'
    )


    buf = _qio.BytesIO()
    fig.figure.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#fbf9f4')
    buf.seek(0)
    return fig, buf.getvalue(), None

def show_report(herd,field_tag):
    with st.spinner(f"Generating report for {field_tag}..."):
        _fig, _png, _err = scout_report(herd,field_tag)
    if _err:
        st.warning(_err)
        return
    st.pyplot(_fig)
    st.download_button("Download Image", _png,
        file_name=f"{field_tag.replace(' ','_')}_percentiles.png",
        mime="image/png", key=f"qr_dl_")

###############################################################################################################################

herd = clean_herd_dataset()
all_herd_list, herd_compare_tab = st.tabs(['Full Herd','Comparison'])

with all_herd_list:
    st.subheader("Click the box next to a cow or bull to visualize its percentile rankings. Below the image is a button to download it.")
    
    _pl_selection = st.dataframe(
        herd,
        width='stretch',
        on_select="rerun",
        selection_mode="single-row",
        key="pl_df_select",
    )
    _pl_selected_rows = _pl_selection.selection.rows if _pl_selection and hasattr(_pl_selection, 'selection') else []
    if _pl_selected_rows:
        _pl_clicked = herd.iloc[_pl_selected_rows[0]]
        _pl_clicked_name = _pl_clicked['Field Tag']
        st.divider()
        show_report(
            herd, _pl_clicked_name
        )


with herd_compare_tab:
    st.subheader("Select 2 cows to see their percentiles plotted together")
    
    _ZONE_COLORS = [
        (0, 10, '#01349b', 'Elite (Top 10%)'),        # Elite
        (10, 35, '#007f35', 'Above Average (11-35%)'), # Above Average
        (35, 66, '#9b6700', 'Average (36-66%)'),        # Average
        (66, 100, '#b60918', 'Below Average (Bottom 35%)'), # Below Average
    ]

    def _dot_color(pct):
        """Return the zone colour for a single percentile value (0-100)."""
        for lo, hi, col, _ in _ZONE_COLORS:
            if lo <= pct <= hi:
                return col
        return '#b60918'

    def _make_comparison_chart(
        cow1, cow2,
        color1='#4C94F6', color2='#EE5454',
        bg_color='#fbf9f4',
    ):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        from matplotlib.lines import Line2D

        column_mapping = {
            "ProS":"ProS",
            "HerdBuilder":"HerdB",
            "GridMaster":"GridM",
            
            "Daughter's Milk":'MILK',
            'Heifer Pregnancy':'HPG',
            'Calving Ease Maternal':'CEM',
            'Calving Ease Direct':'CED',
            
            'Birth Weight':'BW',
            'Weaning Weight':'WW',
            'Yearling Weight':'YW',
            
            'Average Daily Gain':'ADG',
            'Dry Matter Intake':'DMI',
            'Maintenance Energy':'ME',
            'Stayability':'STAY',
            
            'Marbling':'MARB',
            'Yield Grade':'YG',
            'Carcass Weight':'CW',
            'Rib Eye Area':'REA',
            'Fat':'FAT'
        }

        labels = list(column_mapping.keys())
        cols = [f"{l}_pct" for l in labels]
        labels = [column_mapping[l] for l in labels]

        reg1 = herd[herd['Field Tag']==cow1]['Reg #'].values[0]
        reg2 = herd[herd['Field Tag']==cow2]['Reg #'].values[0]

        pcts1 = herd[herd['Field Tag']==cow1][cols].values.tolist()[0]
        pcts1 = [p*100 for p in pcts1]
        pcts2 = herd[herd['Field Tag']==cow2][cols].values.tolist()[0]
        pcts2 = [p*100 for p in pcts2]

        n = len(labels)
        # Extra bottom margin for the notes block (same position as scout_report)
        fig_h = max(7, n * 0.52 + 3.5)
        fig, ax = plt.subplots(figsize=(12, fig_h), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)

        y_pos = np.arange(n)

        # ── Percentile zone background bands (full width, like scout_report) ──
        zone_bg = {
            (0, 10): '#dce8fb',   # Elite
            (10, 35): '#dff2e9',   # Above Average
            (35, 66): '#fff6e0',   # Average
            (66, 100): '#fde8ea',   # Below Average
        }
        for (lo, hi), bg in zone_bg.items():
            ax.axvspan(lo, hi, color=bg, alpha=0.35, zorder=0)

        # Subtle alternating row bands on top of zone colours
        for i in range(n):
            if i % 2 == 0:
                ax.barh(i, 100, left=0, height=0.85, color='#00000008', zorder=1)

        # Zone divider lines
        for x in [10, 66, 35]:
            ax.axvline(x, color='#f1e9d8', linewidth=0.8, linestyle='--', zorder=2)

        # Connecting lines — coloured by the leading player
        for i, (p1, p2) in enumerate(zip(pcts1, pcts2)):
            if p1 < p2:
                line_col = color1
            elif p2 < p1:
                line_col = color2
            else:
                line_col = 'grey'
            ax.plot([p1, p2], [i, i], color='#4a2e19', linewidth=2.2, zorder=3,
                    solid_capstyle='round', alpha=0.55)

        # Dots coloured by percentile zone (matching scout_report bar colours)
        for i, (p1, p2) in enumerate(zip(pcts1, pcts2)):
            ax.scatter(p1, i, color=color1, s=200, zorder=5,
                       edgecolors='w', linewidths=.5)
            ax.scatter(p2, i, color=color2, s=200, zorder=5,
                       edgecolors='w', linewidths=.5)

        # Per-90 value labels next to each dot
        for i, (p1, p2) in enumerate(zip(pcts1, pcts2)):
            offset = 2.5
            ha1 = 'right' if p1 <= p2 else 'left'
            ha2 = 'left'  if p1 <= p2 else 'right'
            dx1 = 0 #-offset if ha1 == 'right' else offset
            dx2 = 0 # offset if ha2 == 'left'  else -offset
            ax.text(p1 + dx1, i+.275, f'{int(p1)}', va='center', ha='center', zorder=6,
                    fontsize=12, color='#4a2e19', fontweight='bold')
            ax.text(p2 + dx2, i+.275, f'{int(p2)}', va='center', ha='center', zorder=6,
                    fontsize=12, color='#4a2e19', fontweight='bold')

        # Zone labels along the top
        for lo, hi, col, lbl_short in [
            (66, 100, '#b60918', 'Below Average'),
            (35, 66, '#9b6700', 'Average'),
            (10, 35, '#007f35', 'Above Avg'),
            (0, 10, '#01349b', 'Top 10%'),
        ]:
            ax.text((lo+hi)/2, -0.75, lbl_short, ha='center', va='top',
                    fontsize=10, color=col, fontweight='bold')

        # Y-axis metric labels — coloured by the leading player
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=14, color='#333333')
        ax.tick_params(axis='y', length=0)
        for tick, p1, p2 in zip(ax.get_yticklabels(), pcts1, pcts2):
            if p1 < p2:
                tick.set_color(color1)
                tick.set_fontweight('bold')
            elif p2 < p1:
                tick.set_color(color2)
                tick.set_fontweight('bold')
            else:
                tick.set_color('#888888')
        ax.set_xlim(0, 100)
        ax.set_ylim(-1.3, n - 0.2)
        ax.set_xlabel(f'Percentile Rankings', fontsize=12, color='#fbf9f4', labelpad=8)
        ax.xaxis.set_tick_params(labelsize=8, colors='#fbf9f4')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.invert_yaxis()

        # ── Title ─────────────────────────────────────────────────────────────
        fig.text(0.5, 1.0, f'{cow1} (#{reg1})                    ', fontsize=13, fontweight='bold', color=color1, ha='right')
        fig.text(0.5, 1.0, f'Percentile Comparison', fontsize=13, fontweight='bold', color='#4a2e19', ha='center')
        fig.text(0.5, 1.0, f'                    {cow2} (#{reg2})', fontsize=13, fontweight='bold', color=color2, ha='left')


        notes_lines = [
            "Points & numbers are each cow's percentiles. Metrics are colored based on the better-scoring cow.",
        ]
        notes_text = '\n'.join(notes_lines)
        fig.text(0.5, 0.08, notes_text, fontsize=10, color='#4A2E19',
                 va='bottom', ha='center', linespacing=1.6)

        plt.tight_layout(rect=[0, 0.06, 0.88, 0.995])
        return fig


    with st.form('Cow Selection'):
        _cc1, _cc2 = st.columns(2)
        with _cc1:
            _cow_sel1 = st.selectbox('Cow 1', herd['Field Tag'].unique(), key='cmp_p1')
        with _cc2:
            _cow_sel2 = st.selectbox('Cow 2', herd['Field Tag'].unique(), key='cmp_p2')
        _cmp_submitted = st.form_submit_button("Generate Image", width='stretch')

    if _cmp_submitted and _cow_sel1 and _cow_sel2:
        if _cow_sel1 == _cow_sel2:
            st.warning("Please select two different cows")
        else:
                _cmp_fig = _make_comparison_chart(
                    cow1=_cow_sel1, cow2=_cow_sel2,
                )
                st.pyplot(_cmp_fig)

                # Download
                import io as _cio
                _cbuf = _cio.BytesIO()
                _cmp_fig.savefig(_cbuf, format='png', dpi=150, bbox_inches='tight', facecolor='#fbf9f4')
                _cbuf.seek(0)
                st.download_button("Download Chart", _cbuf,
                    file_name=f"compare_{_cow_sel1.replace(' ','_')}_vs_{_cow_sel2.replace(' ','_')}.png",
                    mime="image/png", key="dl_compare")

    elif _cmp_submitted:
        st.info("Please select both cows before comparing.")

