"""Serie mensual sintética por tramo de mora (Banco Austral, entidad ficticia), oct 2023 a sep 2026.
Une cartera (saldo, roll rate, self-cure, Expected Loss) y gestión de cobranza (intentos, contactos,
promesas, recupero). Niveles calibrados con el modelo del notebook (roll 26% / 49% / 60%; self-cure 48% / 22% / 9%)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(11)
meses = pd.period_range("2023-10", "2026-09", freq="M")
LGD = 0.55
PD = {"1-30": 0.238, "31-60": 0.610, "61-90": 0.869}
#            clientes, saldo medio, roll, self-cure, intensidad, contactab. máx, k, efectividad, promesa
P = {"1-30":  (420, 2_300_000, 0.27, 0.47, 2.2, 0.80, 0.70, 0.60, 0.74),
     "31-60": (160, 2_600_000, 0.50, 0.21, 3.5, 0.74, 0.42, 0.44, 0.62),
     "61-90": (115, 2_900_000, 0.61, 0.09, 4.6, 0.64, 0.33, 0.30, 0.50)}
EST = {1: .020, 2: -.025, 3: -.020, 4: 0, 5: .005, 6: .015, 7: .025, 8: 0, 9: -.005, 10: 0, 11: -.005, 12: .020}
TEND = {2023: -.012, 2024: -.010, 2025: 0.0, 2026: .012}
filas = []
for i, m in enumerate(meses):
    for tramo, (c0, s0, roll0, cure0, it0, cmax, k, ef0, pr0) in P.items():
        clientes = int(rng.normal(c0 * (1 + .002 * i), c0 * .05))
        saldo = clientes * s0 * (1 + .025 * i) * rng.lognormal(0, .03)
        gestionados = int(round(clientes * rng.uniform(.88, .95)))
        intens = max(1.0, rng.normal(it0 + (.25 if m.year == 2026 else 0) + (.1 if m.month in (2, 3) else 0), .15))
        contact = np.clip(cmax * (1 - np.exp(-k * intens)) + .4 * TEND[m.year] + .3 * EST[m.month] + rng.normal(0, .012), .2, .92)
        contactados = int(round(gestionados * contact))
        promesas = int(round(contactados * np.clip(rng.normal(pr0 + TEND[m.year], .02), .2, .95)))
        efect = np.clip(rng.normal(ef0 + TEND[m.year] + .6 * EST[m.month], .02), .1, .9)
        cumplidas = min(promesas, int(round(contactados * efect)))
        cure = np.clip(rng.normal(cure0 + .8 * TEND[m.year] + .8 * EST[m.month], .015), .02, .8)
        roll = np.clip(rng.normal(roll0 - TEND[m.year] - .8 * EST[m.month], .018), .05, .9)
        recup = np.clip(cure * .9 + .25 * (cumplidas / gestionados) * (1 - cure) + rng.normal(0, .008), .02, .85)
        filas.append({"mes": str(m), "anio": m.year, "mes_num": m.month, "tramo": tramo,
                      "clientes": clientes, "saldo_inicial": round(saldo, -3),
                      "saldo_roleado": round(saldo * roll, -3), "clientes_self_cure": int(round(clientes * cure)),
                      "clientes_gestionados": gestionados, "intentos": int(round(gestionados * intens)),
                      "clientes_contactados": contactados, "promesas": promesas, "promesas_cumplidas": cumplidas,
                      "monto_recuperado": round(saldo * recup, -3),
                      "expected_loss": round(saldo * LGD * PD[tramo], -3)})
df = pd.DataFrame(filas)
df.to_csv("cartera_gestion_mensual.csv", index=False)
g = df.groupby(["anio", "tramo"]).sum(numeric_only=True)
print(pd.DataFrame({"roll": g.saldo_roleado / g.saldo_inicial, "self_cure": g.clientes_self_cure / g.clientes,
    "contact": g.clientes_contactados / g.clientes_gestionados, "efect": g.promesas_cumplidas / g.clientes_contactados,
    "intens": g.intentos / g.clientes_gestionados, "recupero": g.monto_recuperado / g.saldo_inicial}).round(3))
