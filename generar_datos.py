"""
Generador de datos sintéticos — Banco Austral (entidad ficticia)
=================================================================

Simula 18 meses de comportamiento de una cartera de consumo (préstamos
personales y tarjetas de crédito) para estudiar cobranza preventiva y
temprana (mora hasta 90 días).

Idea central del simulador
--------------------------
Cada cliente tiene un nivel de "estrés financiero" que NO se observa
directamente. Cuando el estrés sube (por ejemplo, pierde el empleo):

  1. Primero cambian las SEÑALES DE COMPORTAMIENTO: deja de acreditar el
     sueldo, bajan sus saldos, sube el uso de la tarjeta, se endeuda en
     otras entidades y entra menos a la app.
  2. Recién después aparece la MORA y el roleo entre tramos.

Ese desfase es justamente lo que un modelo predictivo puede aprovechar
para "llegar antes".

Todos los datos son ficticios. Montos en pesos argentinos (ARS).

Uso:  python generar_datos.py
Salida: clientes.csv y comportamiento_mensual.csv (en la misma carpeta)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 2026
N_CLIENTES = 12_000
N_MESES = 18
MES_INICIAL = "2025-04"

# Tramos de mora: 0 = al día, 1 = 1-30, 2 = 31-60, 3 = 61-90, 4 = 90+ (pasa a recupero)
TRAMOS = {0: "Al día", 1: "1-30", 2: "31-60", 3: "61-90", 4: "90+"}

rng = np.random.default_rng(SEMILLA)
sig = lambda x: 1 / (1 + np.exp(-x))


def generar_clientes(n: int) -> pd.DataFrame:
    edad = rng.integers(21, 71, n)
    antiguedad = np.clip(rng.gamma(2.2, 30, n), 3, 300).astype(int)
    ingreso = np.round(rng.lognormal(np.log(1_800_000), 0.55, n), -3)
    segmento = np.select(
        [ingreso >= 4_000_000, ingreso >= 2_200_000], ["Premium", "Plus"], "Masivo"
    )
    # Riesgo de base (no observable): peor en clientes jóvenes, nuevos y de menor ingreso
    riesgo_base = (
        rng.normal(0, 1, n)
        - 0.25 * (np.log(ingreso) - np.log(1_800_000))
        - 0.004 * (antiguedad - 60)
        - 0.01 * (edad - 40)
    )
    tiene_prestamo = rng.random(n) < 0.70
    tiene_tarjeta = (rng.random(n) < 0.85) | ~tiene_prestamo
    limite = np.where(tiene_tarjeta, np.round(ingreso * rng.uniform(1.0, 3.0, n), -3), 0)
    prestamo_inicial = np.where(
        tiene_prestamo, np.round(ingreso * rng.uniform(1.0, 6.0, n), -3), 0
    )
    tiene_inversiones = rng.random(n) < sig(-1.2 + 0.9 * (segmento == "Premium") - 0.6 * riesgo_base)
    afinidad_digital = np.clip(rng.normal(1.0, 0.35, n) - 0.008 * (edad - 40), 0.2, 2.0)

    return pd.DataFrame(
        {
            "cliente_id": [f"BA{i:06d}" for i in range(1, n + 1)],
            "edad": edad,
            "antiguedad_meses": antiguedad,
            "segmento": segmento,
            "ingreso_mensual": ingreso,
            "tiene_prestamo": tiene_prestamo.astype(int),
            "tiene_tarjeta": tiene_tarjeta.astype(int),
            "limite_tarjeta": limite,
            "monto_prestamo_inicial": prestamo_inicial,
            "tiene_inversiones": tiene_inversiones.astype(int),
            # columnas internas del simulador (no se exportan)
            "_riesgo_base": riesgo_base,
            "_afinidad_digital": afinidad_digital,
        }
    )


def simular(cli: pd.DataFrame) -> pd.DataFrame:
    n = len(cli)
    r = cli["_riesgo_base"].to_numpy()
    afin = cli["_afinidad_digital"].to_numpy()
    ingreso = cli["ingreso_mensual"].to_numpy()
    limite = cli["limite_tarjeta"].to_numpy()
    tiene_tc = cli["tiene_tarjeta"].to_numpy() == 1

    estres = rng.normal(0, 0.5, n)
    acredita = (rng.random(n) < sig(1.1 + 0.4 * (cli["segmento"] != "Masivo").to_numpy())).astype(int)
    deuda_otras = np.clip(0.5 + 0.3 * r + rng.normal(0, 0.2, n), 0, None)  # en veces el ingreso
    saldo_prestamo = cli["monto_prestamo_inicial"].to_numpy().astype(float)
    tramo = np.zeros(n, dtype=int)
    utiliz = np.clip(0.30 + 0.10 * r + rng.normal(0, 0.1, n), 0, 1)
    meses = pd.period_range(MES_INICIAL, periods=N_MESES, freq="M").astype(str)

    filas = []
    for t, mes in enumerate(meses):
        # 1) Dinámica del estrés: persistente, con shocks (pérdida de ingresos, etc.)
        p_shock = 0.012 * np.exp(0.5 * r)
        shock = rng.random(n) < p_shock
        estres = 0.80 * estres + 0.30 * rng.normal(0, 1, n) + 2.2 * shock

        # 2) Señales de comportamiento (reaccionan al estrés antes que la mora)
        corte = shock & (rng.random(n) < 0.6)
        acredita = np.where(corte, 0, acredita)
        vuelve = (acredita == 0) & (rng.random(n) < 0.06 * np.exp(-0.4 * np.clip(estres, 0, None)))
        acredita = np.where(vuelve, 1, acredita)

        saldo_cuentas = np.round(
            ingreso * 0.8 * np.exp(-0.35 * estres) * np.where(acredita == 1, 1.0, 0.45)
            * rng.lognormal(0, 0.25, n), -2
        )
        utiliz = np.clip(0.6 * utiliz + 0.4 * (0.30 + 0.10 * r + 0.13 * estres) + rng.normal(0, 0.04, n), 0, 1.05)
        utiliz = np.where(tiene_tc, utiliz, 0)
        deuda_otras = np.clip(
            0.92 * deuda_otras + 0.08 * (0.5 + 0.3 * r) + 0.10 * np.clip(estres, 0, None) + rng.normal(0, 0.03, n),
            0, None,
        )
        p_sit = sig(-4.0 + 1.1 * estres + 0.8 * deuda_otras)
        situacion = 1 + (rng.random(n) < p_sit) + (rng.random(n) < p_sit * 0.5) + (rng.random(n) < p_sit * 0.2)
        logins = rng.poisson(np.clip(10 * afin * np.exp(-0.30 * estres), 0.2, None))
        dias_anticipo = np.where(
            tramo == 0, np.clip(np.round(rng.normal(5 - 1.6 * estres - 0.8 * r, 2.5)), 0, 15), 0
        )
        pago_digital = (rng.random(n) < sig(-0.5 + 1.2 * afin)).astype(int)

        saldo_tarjeta = np.round(limite * utiliz, -2)
        saldo_prestamo = np.round(saldo_prestamo * 0.975, -2)
        exposicion = saldo_prestamo + saldo_tarjeta

        # Promesa de pago incumplida (solo clientes en mora)
        promesa_incumplida = ((tramo >= 1) & (tramo <= 3) & (rng.random(n) < sig(-1.0 + 0.6 * estres))).astype(int)

        dias_atraso = np.select(
            [tramo == 0, tramo == 1, tramo == 2, tramo == 3, tramo == 4],
            [0, rng.integers(1, 31, n), rng.integers(31, 61, n), rng.integers(61, 91, n), 91],
        )

        filas.append(
            pd.DataFrame(
                {
                    "cliente_id": cli["cliente_id"],
                    "mes": mes,
                    "tramo": tramo,
                    "dias_atraso": dias_atraso,
                    "acredita_sueldo": acredita,
                    "saldo_cuentas": saldo_cuentas,
                    "utilizacion_tarjeta": np.round(utiliz, 3),
                    "deuda_otras_entidades_x_ingreso": np.round(deuda_otras, 3),
                    "situacion_sistema": situacion,
                    "logins_app": logins,
                    "dias_pago_anticipado": dias_anticipo,
                    "pago_canal_digital": pago_digital,
                    "promesa_incumplida": promesa_incumplida,
                    "saldo_prestamo": saldo_prestamo,
                    "saldo_tarjeta": saldo_tarjeta,
                    "exposicion": exposicion,
                }
            )
        )

        # 3) Transición de tramo para el mes siguiente
        nuevo = tramo.copy()
        al_dia = tramo == 0
        p_entrada = sig(
            -4.05 + 0.95 * estres + 0.45 * r + 0.55 * (acredita == 0)
            + 1.2 * np.clip(utiliz - 0.85, 0, None) + 0.35 * (situacion >= 2)
        )
        nuevo[al_dia & (rng.random(n) < p_entrada)] = 1

        en_mora = (tramo >= 1) & (tramo <= 3)
        logit_roll = -1.35 + 0.30 * tramo + 0.85 * estres + 0.35 * r + 0.4 * promesa_incumplida + 0.3 * (situacion >= 3)
        logit_cura = 0.9 - 0.75 * tramo - 0.80 * estres - 0.30 * r + 0.035 * logins + 0.45 * acredita + 0.3 * pago_digital
        e_roll, e_cura = np.exp(logit_roll), np.exp(logit_cura)
        tot = 1 + e_roll + e_cura
        u = rng.random(n)
        cura = en_mora & (u < e_cura / tot)
        roll = en_mora & ~cura & (u < (e_cura + e_roll) / tot)
        nuevo[cura] = 0
        nuevo[roll] = tramo[roll] + 1
        # 90+ es absorbente: el caso pasa a recupero
        tramo = nuevo
        # Si se cura, el estrés cede un poco (resolvió su situación)
        estres = np.where(cura, estres * 0.6, estres)

    return pd.concat(filas, ignore_index=True)


def main():
    base = Path(__file__).resolve().parent
    base.mkdir(exist_ok=True)
    cli = generar_clientes(N_CLIENTES)
    panel = simular(cli)
    cli.drop(columns=[c for c in cli.columns if c.startswith("_")]).to_csv(base / "clientes.csv", index=False)
    panel.to_csv(base / "comportamiento_mensual.csv", index=False)
    print(f"Clientes: {len(cli):,} | Filas mensuales: {len(panel):,}")
    print(panel.groupby("mes")["tramo"].value_counts(normalize=True).unstack().round(3).tail(3))


if __name__ == "__main__":
    main()
