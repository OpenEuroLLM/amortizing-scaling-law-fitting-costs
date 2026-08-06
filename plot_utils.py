N_palette = [
    "#ff62e0",
    "#ab68d8",
    "#5d63bb",
    "#19548e",
    "#003f5c",
    "#162C36",
]

D_palette = [
    "#10ffbf",
    "#23e4ac",
    "#2bca99",
    "#2fb186",
    "#309874",
    "#308063",
    "#2d6852",
    "#295242",
    "#243c32",
]

FORMAT_BILLIONS = lambda x: f"{float(x)/1e9:g}B"

Ns = [.05e9, .13e9, .3e9, .6e9, 1e9, 1.7e9]
Ds = [6e9, 12e9, 20e9, 30e9, 50e9, 80e9, 120e9, 200e9, 300e9]
N_color_map = dict(zip(Ns, N_palette[:len(Ns)]))
D_color_map = dict(zip(Ds, D_palette[:len(Ds)]))

N_ticks = [0.05e9, 0.1e9, 0.2e9, 0.4e9, 0.8e9, 1.6e9]
D_ticks = [5e9, 10e9, 20e9, 40e9, 80e9, 160e9, 320e9]
