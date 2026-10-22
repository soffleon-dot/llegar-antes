# Llegar antes

### Modelo predictivo de cobranza preventiva y temprana

> Detectar el deterioro del cliente por su comportamiento y resolverlo con él, antes que la mora y antes que el resto del sistema financiero.

**Banco Austral** es una entidad ficticia. Todos los datos son sintéticos, generados por criterio de confidencialidad.

---

## Resultados en 30 segundos

Con **la misma capacidad de gestores** (350 llamados por mes), comparamos dos formas de gestionar la mora temprana en un mes que el modelo nunca vio:

| | Gestión tradicional<br/>(por días de atraso) | Llegar antes<br/>(por valor en juego) |
|---|---|---|
| Pérdida generada que pasó por un gestor | 29% | **49%** |
| Pérdida esperada evitada | $ 35,3 M | **$ 54,0 M (+53%)** |
| Pesos de pérdida evitados por peso de gestión | 16,4 | **23,4** |
| Contactos resueltos por canal digital | 46% | **79%** |

Las cifras de pérdida evitada dependen de supuestos de efectividad por canal, explicados abajo.

![Curva de captura](06_curva_captura.png)

---

## El problema

La cobranza tradicional actúa cuando el cliente ya está en mora y ordena el trabajo por días de atraso. Eso tiene tres costos:

1. **Llega tarde.** Cuando aparece el primer día de atraso, el cliente muchas veces ya se endeudó en otras entidades.
2. **Gasta gestión donde no hace falta.** Casi la mitad de los clientes en 1-30 días se regulariza al mes siguiente, y muchos lo harían con un simple recordatorio.
3. **No distingue cuánto hay en juego.** Un cliente con alta exposición y alta probabilidad de rolear vale más que uno con más días de atraso pero poca deuda.

Esta es la matriz de transición mensual de la cartera:

![Matriz de transición](02_matriz_transicion.png)

La mitad de lo que entra en 31-60 pasa a 61-90, y el 60% de 61-90 cae en 90+. Cada salto de tramo multiplica la pérdida esperada.

## La solución

```mermaid
flowchart LR
  A[Señales de<br/>comportamiento] --> B[Modelos<br/>predictivos]
  B --> C[Pérdida esperada<br/>en pesos]
  C --> D[Prioridad y canal<br/>por cliente]
  D --> E[Gestor · Digital ·<br/>Preventivo]
```

### 1. Señales de comportamiento

La mora es la consecuencia. Lo que se mide son los cambios previos:

- Deja de acreditar el sueldo.
- Bajan sus saldos.
- Usa cada vez más el límite de la tarjeta.
- Aumenta su deuda en otras entidades o empeora su situación en el sistema.
- Entra menos a la app.
- Paga cada vez más cerca del vencimiento.

### 2. Tres modelos predictivos

| Modelo | Población | Predice | Gini fuera de tiempo |
|---|---|---|---|
| Preventivo | Clientes al día | Entrada en mora el mes próximo | 0,47 |
| Roleo | Mora de 1 a 90 días | Pase al tramo siguiente | 0,59 |
| Self-cure | Mora de 1 a 90 días | Regularización sin intervención | 0,60 |

Cada uno compara una **regresión logística** (explicable) contra **LightGBM** (retador). El retador no mejora de forma relevante, así que en producción queda la logística: se puede defender variable por variable ante un comité, y cada gestor recibe el **motivo** de cada caso.

![Pesos del modelo preventivo](03_pesos_preventivo.png)

El modelo está bien calibrado: cuando predice 10%, entra en mora cerca del 10%.

![Calibración](04_calibracion_preventivo.png)

### 3. De probabilidad a pesos

**Pérdida esperada = PD × LGD × EAD**

- La **PD a 12 meses** de cada tramo se obtiene elevando la matriz de transición a la potencia 12 (cadena de Markov): 4% al día, 24% en 1-30, 61% en 31-60 y 87% en 61-90.
- **Valor en juego** de cada cliente = probabilidad de rolear × cuánto sube su pérdida esperada si rolea.

### 4. Prioridad y canal

| Canal | A quién va |
|---|---|
| Gestor humano | Los 350 clientes con mayor valor en juego, estén en mora o todavía al día |
| Digital | Clientes con alta probabilidad de self-cure, el 10% de clientes al día más riesgosos (preventivo) y el resto de la mora |

El resultado es una lista operativa (`lista_priorizada.csv`) con cliente, canal, prioridad y motivo principal en lenguaje de negocio.

---

## Decisiones de diseño

| Decisión | Por qué |
|---|---|
| Regresión logística en lugar de LightGBM | Rinden igual, y la logística es explicable y defendible ante un comité |
| Validación fuera de tiempo (8 meses de entrenamiento, 3 de prueba) | Así se usa en la realidad: entrenar con el pasado, aplicar al futuro |
| Excluir "días de atraso" del modelo de roleo | Está casi explicado por el tramo; juntos generaban pesos con signo engañoso |
| Priorizar por valor en pesos, no por probabilidad | Un 60% de probabilidad sobre $ 20 M importa más que un 90% sobre $ 500 mil |
| Contactar clientes al día | La forma más barata de evitar mora es que no ocurra |

## Supuestos y limitaciones

- **Datos sintéticos.** Las relaciones entre variables las define el simulador (`generar_datos.py`). Con datos reales las métricas cambian y hay que recalibrar.
- **Efectividad por canal (supuesto):** un gestor reduce 30% la probabilidad de rolear o entrar en mora; el canal digital, 10%. En producción se mide con un **grupo de control** antes de escalar.
- **Costos por contacto (supuesto):** gestor $ 6.000, digital $ 150.
- **LGD fija del 55%** para consumo sin garantía. Un modelo de LGD por producto y tramo mejoraría la estimación.
- El histórico no registra qué gestión recibió cada cliente. Con datos reales habría que aislar ese efecto.

## Próximos pasos

- [ ] Flujo en **n8n** que lea la lista priorizada, dispare mensajes por canal digital y alerte desvíos.
- [ ] Tablero de roll rates en **Looker Studio**, alimentado por `roll_rates_mensual.csv`.
- [ ] **Plan Anticipa:** refinanciación preventiva, sin bajar la tasa de entrada, para clientes al día de alto valor en juego, con premio al cumplir el plan.
- [ ] Agentes de IA que asignen estrategia a cada call center y reporten el cluster crítico.

---

## Cómo correrlo

```bash
pip install -r requirements.txt
python generar_datos.py                 # genera los datos sintéticos
jupyter notebook modelo_cobranza_temprana.ipynb
```

También se puede abrir en **Google Colab**: subí los archivos del repositorio a una carpeta y ejecutá las celdas en orden.

## Archivos

| Archivo | Qué es |
|---|---|
| `modelo_cobranza_temprana.ipynb` | El análisis completo: roll rates, modelos, validación, estrategia y backtest |
| `generar_datos.py` | Simulador de la cartera sintética |
| `clientes.csv`, `comportamiento_mensual.csv` | Datos sintéticos (12.000 clientes, 18 meses) |
| `lista_priorizada.csv` | Lista operativa del próximo mes: canal, prioridad y motivo |
| `roll_rates_mensual.csv` | Insumo para tableros |
| `resultados_clave.json` | Cifras del backtest |
| `01_…png` a `06_…png` | Gráficos del análisis |
| `construir_notebook.py` | Arma el notebook a partir de sus celdas |

## Stack

Python · pandas · scikit-learn · LightGBM · matplotlib

---

**Sofía León** · Riesgo de crédito, cobranzas y automatización
