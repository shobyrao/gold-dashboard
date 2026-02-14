# ============================================
# 🥇 GOLD TRADING DASHBOARD - WEB VERSION
# ============================================
# Auto-refresh every 60 seconds!

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os

# ============================================
# DATA FUNCTIONS
# ============================================

def get_gold_price():
    try:
        gold = yf.Ticker("GC=F")
        data = gold.history(period="5d")
        if data.empty:
            return {'price':0,'change':0,'pct':0,'high':0,'low':0,'open':0}
        current = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2] if len(data)>1 else current
        ch = current - prev
        return {
            'price': round(current,2),
            'change': round(ch,2),
            'pct': round((ch/prev)*100, 3),
            'high': round(data['High'].iloc[-1],2),
            'low': round(data['Low'].iloc[-1],2),
            'open': round(data['Open'].iloc[-1],2),
        }
    except:
        return {'price':0,'change':0,'pct':0,'high':0,'low':0,'open':0}

def get_gold_chart(period="3mo"):
    try:
        gold = yf.Ticker("GC=F")
        return gold.history(period=period)
    except:
        return pd.DataFrame()

def get_factor(ticker, name, icon, relation="inverse"):
    try:
        asset = yf.Ticker(ticker)
        data = asset.history(period="1mo")
        if data.empty:
            return None
        current = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2] if len(data)>1 else current
        ch = current - prev
        pct = (ch/prev)*100

        week = data['Close'].iloc[-5] if len(data)>=5 else data['Close'].iloc[0]
        wpct = ((current-week)/week)*100

        if relation == "inverse":
            if pct > 0.2:
                impact, strength = "bearish", min(abs(pct)*15, 100)
            elif pct < -0.2:
                impact, strength = "bullish", min(abs(pct)*15, 100)
            else:
                impact, strength = "neutral", abs(pct)*5
        else:
            if pct > 0.2:
                impact, strength = "bullish", min(abs(pct)*15, 100)
            elif pct < -0.2:
                impact, strength = "bearish", min(abs(pct)*15, 100)
            else:
                impact, strength = "neutral", abs(pct)*5

        return {
            'name':name, 'icon':icon, 'value':round(current,2),
            'change':round(ch,2), 'pct':round(pct,3),
            'wpct':round(wpct,3), 'impact':impact,
            'strength':round(min(strength,100),1)
        }
    except:
        return None

def get_all_factors():
    factors = {}
    configs = [
        ("DX-Y.NYB", "US Dollar (DXY)", "💵", "inverse"),
        ("^GSPC", "S&P 500", "📊", "inverse"),
        ("^TNX", "10Y Bond Yield", "📜", "inverse"),
        ("^VIX", "VIX Fear Index", "😨", "direct"),
        ("CL=F", "Crude Oil", "🛢️", "direct"),
        ("SI=F", "Silver", "🥈", "direct"),
        ("BTC-USD", "Bitcoin", "₿", "inverse"),
        ("EURUSD=X", "EUR/USD", "💶", "direct"),
    ]
    for ticker, name, icon, rel in configs:
        key = name.lower().replace(" ","_").replace("(","").replace(")","")
        factors[key] = get_factor(ticker, name, icon, rel)
    return factors

def get_technicals():
    try:
        data = yf.Ticker("GC=F").history(period="3mo")
        if data.empty or len(data)<50:
            return None
        close = data['Close']
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        delta = close.diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        rs = gain/loss
        rsi = (100-(100/(1+rs))).iloc[-1]
        cur = close.iloc[-1]
        return {
            'sma20':round(sma20,2), 'sma50':round(sma50,2),
            'rsi':round(rsi,1),
            'above20': cur>sma20, 'above50': cur>sma50,
            'trend': 'BULLISH' if sma20>sma50 else 'BEARISH',
            'rsi_sig': 'OVERBOUGHT' if rsi>70 else ('OVERSOLD' if rsi<30 else 'NEUTRAL'),
        }
    except:
        return None

# ============================================
# DASH APP
# ============================================

app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}],
    title="🥇 Gold Dashboard"
)

server = app.server

app.layout = html.Div([
    # Auto refresh har 60 second
    dcc.Interval(id='timer', interval=60*1000, n_intervals=0),

    # Header
    dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("🥇 GOLD TRADING DASHBOARD",
                    className="text-warning fw-bold mb-0"), width="auto"),
                dbc.Col(html.Span(id="clock", className="text-muted"),
                    width="auto", className="ms-auto"),
            ], align="center", className="w-100"),
        ], fluid=True),
        color="dark", dark=True, className="mb-3"
    ),

    dbc.Container([
        # Gold Price
        html.Div(id="gold-price-box"),

        # Signal Row
        html.Div(id="signal-box", className="mb-3"),

        # Sentiment Bar
        html.Div(id="sentiment-bar", className="mb-3"),

        # Factors: Bearish LEFT | Bullish RIGHT
        dbc.Row(id="factors-row", className="mb-3"),

        # Technicals
        html.Div(id="tech-box", className="mb-3"),

        # Charts
        dcc.Graph(id="gold-chart", config={'displayModeBar':False}),
        
        dbc.Row([
            dbc.Col(dcc.Graph(id="gauge-chart", config={'displayModeBar':False}), lg=4, md=12),
            dbc.Col(dcc.Graph(id="bar-chart", config={'displayModeBar':False}), lg=4, md=12),
            dbc.Col(dcc.Graph(id="heatmap-chart", config={'displayModeBar':False}), lg=4, md=12),
        ], className="mb-3"),

        # Footer
        html.Hr(),
        html.P("⚠️ Educational purpose only. Not financial advice.",
               className="text-muted text-center small"),
    ], fluid=True),

], style={"backgroundColor":"#0a0a1a","minHeight":"100vh"})

# ============================================
# MAIN CALLBACK - SAB KUCH UPDATE KARTA HAI
# ============================================

@app.callback(
    [Output("gold-price-box","children"),
     Output("signal-box","children"),
     Output("sentiment-bar","children"),
     Output("factors-row","children"),
     Output("tech-box","children"),
     Output("gold-chart","figure"),
     Output("gauge-chart","figure"),
     Output("bar-chart","figure"),
     Output("heatmap-chart","figure"),
     Output("clock","children")],
    [Input("timer","n_intervals")]
)
def update(n):
    gold = get_gold_price()
    factors = get_all_factors()
    tech = get_technicals()
    chart_data = get_gold_chart("3mo")

    # Separate bullish/bearish
    bullish = []
    bearish = []
    neutral_list = []
    for k,f in factors.items():
        if f is None: continue
        if f['impact']=='bullish': bullish.append(f)
        elif f['impact']=='bearish': bearish.append(f)
        else: neutral_list.append(f)

    bullish.sort(key=lambda x:x['strength'], reverse=True)
    bearish.sort(key=lambda x:x['strength'], reverse=True)

    bt = sum(f['strength'] for f in bullish)
    brt = sum(f['strength'] for f in bearish)
    total = bt+brt
    sentiment = (bt/total*100) if total>0 else 50

    if sentiment>65: sig,sigc = "🟢 STRONG BUY","#00ff88"
    elif sentiment>55: sig,sigc = "🟢 BUY","#00cc66"
    elif sentiment<35: sig,sigc = "🔴 STRONG SELL","#ff4444"
    elif sentiment<45: sig,sigc = "🔴 SELL","#ff6666"
    else: sig,sigc = "🟡 HOLD","#ffaa00"

    pc = "#00ff88" if gold['change']>=0 else "#ff4444"
    ar = "▲" if gold['change']>=0 else "▼"

    # 1. Gold Price Box
    price_box = dbc.Card([dbc.CardBody([
        html.Div("GOLD (XAU/USD)", className="text-warning text-center",
                style={"letterSpacing":"3px","fontSize":"14px"}),
        html.H1(f"${gold['price']:,.2f}",
            className="text-center fw-bold",
            style={"fontSize":"52px","color":"#FFD700"}),
        html.H4(f"{ar} ${abs(gold['change']):,.2f} ({gold['pct']:+.3f}%)",
            className="text-center", style={"color":pc}),
        html.Div([
            html.Span(f"H: ${gold['high']:,.2f}", className="text-success me-3"),
            html.Span(f"L: ${gold['low']:,.2f}", className="text-danger me-3"),
            html.Span(f"O: ${gold['open']:,.2f}", className="text-info"),
        ], className="text-center"),
    ])], className="border-warning mb-3",
        style={"backgroundColor":"rgba(255,215,0,0.05)"})

    # 2. Signal Box
    signal_box = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("SIGNAL", className="text-muted"),
            html.H3(sig, style={"color":sigc,"fontWeight":"bold"}),
        ]), className="text-center"), lg=4, md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("SENTIMENT", className="text-muted"),
            html.H3(f"{sentiment:.1f}%", className="text-warning"),
        ]), className="text-center"), lg=4, md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small("FACTORS", className="text-muted"),
            html.H4([
                html.Span(f"🟢{len(bullish)} ", style={"color":"#00ff88"}),
                html.Span("vs "),
                html.Span(f"🔴{len(bearish)}", style={"color":"#ff4444"}),
            ]),
        ]), className="text-center"), lg=4, md=4),
    ])

    # 3. Sentiment Bar
    bw = sentiment
    brw = 100 - sentiment
    sent_bar = dbc.Card(dbc.CardBody([
        html.Div([
            html.Span(f"🟢 Bullish {bw:.0f}%",
                style={"color":"#00ff88","fontSize":"13px"}),
            html.Span(f"Bearish {brw:.0f}% 🔴",
                style={"color":"#ff4444","fontSize":"13px","float":"right"}),
        ]),
        dbc.Progress([
            dbc.Progress(value=bw, color="success", bar=True,
                label=f"{bw:.0f}%", style={"fontSize":"12px"}),
            dbc.Progress(value=brw, color="danger", bar=True,
                label=f"{brw:.0f}%", style={"fontSize":"12px"}),
        ], style={"height":"28px"}, className="mt-1"),
    ]))

    # 4. Factor Cards
    def make_cards(flist, ftype):
        cards = []
        for f in flist:
            if ftype=="bullish":
                bc,arrow_f = "success","▲"
                bg = "rgba(0,255,0,0.05)"
                col = "#00ff88"
            else:
                bc,arrow_f = "danger","▼"
                bg = "rgba(255,0,0,0.05)"
                col = "#ff4444"

            cards.append(dbc.Card([dbc.CardBody([
                html.Div([
                    html.Span(f['icon'], style={"fontSize":"22px"}),
                    html.B(f" {f['name']}", style={"fontSize":"14px"}),
                ]),
                html.Div([
                    html.Span(f"Value: {f['value']} "),
                    html.Span(f"({f['pct']:+.2f}%)",
                        style={"color":col,"fontWeight":"bold"}),
                ], style={"fontSize":"13px","color":"#ccc"}),
                html.Div([
                    dbc.Progress(value=f['strength'],
                        color=bc, style={"height":"7px"},
                        className="mt-1"),
                    html.Small(f"Impact: {f['strength']:.0f}%",
                        style={"color":"#888"}),
                ]),
            ], className="p-2")],
                className=f"mb-2 border-{bc}",
                style={"backgroundColor":bg}))

        if not cards:
            cards = [html.P(f"No {ftype} factors active",
                className="text-muted text-center")]
        return cards

    factors_row = [
        # LEFT - BEARISH
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("🔴 BEARISH FACTORS",
                        className="text-danger text-center mb-0"),
                    html.Small("Gold neeche le jaane wale",
                        className="text-muted d-block text-center"),
                ], className="bg-dark"),
                dbc.CardBody(make_cards(bearish,"bearish"),
                    style={"maxHeight":"500px","overflowY":"auto"}),
            ], className="border-danger h-100",
               style={"backgroundColor":"rgba(255,0,0,0.02)"}),
        ], lg=6, md=12, className="mb-3"),

        # RIGHT - BULLISH
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("🟢 BULLISH FACTORS",
                        className="text-success text-center mb-0"),
                    html.Small("Gold upar le jaane wale",
                        className="text-muted d-block text-center"),
                ], className="bg-dark"),
                dbc.CardBody(make_cards(bullish,"bullish"),
                    style={"maxHeight":"500px","overflowY":"auto"}),
            ], className="border-success h-100",
               style={"backgroundColor":"rgba(0,255,0,0.02)"}),
        ], lg=6, md=12, className="mb-3"),
    ]

    # 5. Technicals
    if tech:
        tc = "#00ff88" if tech['trend']=="BULLISH" else "#ff4444"
        rc = "#ff4444" if tech['rsi']>70 else ("#00ff88" if tech['rsi']<30 else "#ffaa00")
        tech_box = dbc.Card([
            dbc.CardHeader(html.H5("📐 Technical Indicators",className="mb-0")),
            dbc.CardBody(dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Small("TREND",className="text-muted"),
                    html.H4(tech['trend'], style={"color":tc}),
                ]),className="text-center bg-dark"), lg=3, md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Small(f"RSI ({tech['rsi']})",className="text-muted"),
                    html.H4(tech['rsi_sig'], style={"color":rc}),
                ]),className="text-center bg-dark"), lg=3, md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Small("SMA 20",className="text-muted"),
                    html.H5(f"${tech['sma20']:,.2f}"),
                    html.Small("✅ Above" if tech['above20'] else "❌ Below",
                        style={"color":"#00ff88" if tech['above20'] else "#ff4444"}),
                ]),className="text-center bg-dark"), lg=3, md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Small("SMA 50",className="text-muted"),
                    html.H5(f"${tech['sma50']:,.2f}"),
                    html.Small("✅ Above" if tech['above50'] else "❌ Below",
                        style={"color":"#00ff88" if tech['above50'] else "#ff4444"}),
                ]),className="text-center bg-dark"), lg=3, md=6),
            ]))
        ])
    else:
        tech_box = html.Div()

    # 6. Gold Chart
    if not chart_data.empty:
        fig1 = make_subplots(rows=2,cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.8,0.2])
        fig1.add_trace(go.Candlestick(
            x=chart_data.index, open=chart_data['Open'],
            high=chart_data['High'], low=chart_data['Low'],
            close=chart_data['Close'], name='Gold',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'), row=1,col=1)
        if len(chart_data)>=20:
            s20 = chart_data['Close'].rolling(20).mean()
            fig1.add_trace(go.Scatter(x=chart_data.index, y=s20,
                name='SMA20', line=dict(color='#FFD700',width=1.5)),
                row=1,col=1)
        if len(chart_data)>=50:
            s50 = chart_data['Close'].rolling(50).mean()
            fig1.add_trace(go.Scatter(x=chart_data.index, y=s50,
                name='SMA50', line=dict(color='#6ec6ff',width=1.5)),
                row=1,col=1)
        vc = ['#00ff88' if c>=o else '#ff4444'
              for c,o in zip(chart_data['Close'],chart_data['Open'])]
        fig1.add_trace(go.Bar(x=chart_data.index, y=chart_data['Volume'],
            marker_color=vc, opacity=0.5, name='Vol'), row=2,col=1)
        fig1.update_layout(template='plotly_dark', height=450,
            xaxis_rangeslider_visible=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10,r=10,t=30,b=10),
            title={'text':'📈 Gold Price Chart (3 Months)',
                   'font':{'color':'#FFD700'}})
    else:
        fig1 = go.Figure()
        fig1.update_layout(template='plotly_dark')

    # 7. Gauge
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number", value=sentiment,
        title={'text':"Sentiment",'font':{'size':16,'color':'#FFD700'}},
        number={'suffix':'%','font':{'size':28}},
        gauge={'axis':{'range':[0,100]}, 'bar':{'color':sigc},
            'steps':[
                {'range':[0,20],'color':'rgba(255,0,0,0.3)'},
                {'range':[20,40],'color':'rgba(255,136,0,0.3)'},
                {'range':[40,60],'color':'rgba(255,255,0,0.3)'},
                {'range':[60,80],'color':'rgba(136,255,0,0.3)'},
                {'range':[80,100],'color':'rgba(0,255,0,0.3)'}],
            'threshold':{'line':{'color':'white','width':3},
                'thickness':0.75,'value':sentiment}}
    ))
    fig2.update_layout(template='plotly_dark', height=280,
        paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20,r=20,t=40,b=10))

    # 8. Bar Chart
    names,vals,cols = [],[],[]
    for f in bullish:
        names.append(f['icon']+' '+f['name'][:18])
        vals.append(f['strength'])
        cols.append('#00ff88')
    for f in bearish:
        names.append(f['icon']+' '+f['name'][:18])
        vals.append(-f['strength'])
        cols.append('#ff4444')

    fig3 = go.Figure(go.Bar(y=names, x=vals, orientation='h',
        marker_color=cols,
        text=[f"{abs(v):.0f}%" for v in vals],
        textposition='auto'))
    fig3.update_layout(template='plotly_dark', height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        title={'text':'⚖️ Bull vs Bear','font':{'color':'#FFD700','size':14}},
        margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'))

    # 9. Heatmap
    try:
        corr_tickers = {'Gold':'GC=F','Silver':'SI=F','DXY':'DX-Y.NYB',
            'S&P500':'^GSPC','Oil':'CL=F','BTC':'BTC-USD'}
        cd = {}
        for nm,tk in corr_tickers.items():
            try:
                d = yf.Ticker(tk).history(period="3mo")
                if not d.empty:
                    cd[nm] = d['Close'].pct_change().dropna()
            except: continue
        if len(cd)>=3:
            df = pd.DataFrame(cd)
            cm = df.corr()
            fig4 = go.Figure(go.Heatmap(
                z=cm.values, x=cm.columns, y=cm.columns,
                colorscale='RdYlGn', zmid=0,
                text=[[f'{v:.2f}' for v in r] for r in cm.values],
                texttemplate='%{text}', textfont={"size":11}))
            fig4.update_layout(template='plotly_dark', height=280,
                paper_bgcolor='rgba(0,0,0,0)',
                title={'text':'🔥 Correlation','font':{'color':'#FFD700','size':14}},
                margin=dict(l=10,r=10,t=40,b=10))
        else:
            fig4 = go.Figure()
            fig4.update_layout(template='plotly_dark', height=280)
    except:
        fig4 = go.Figure()
        fig4.update_layout(template='plotly_dark', height=280)

    clock = f"🔄 Auto-refresh: {datetime.now().strftime('%H:%M:%S')}"

    return (price_box, signal_box, sent_bar, factors_row,
            tech_box, fig1, fig2, fig3, fig4, clock)

# ============================================
# RUN
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
