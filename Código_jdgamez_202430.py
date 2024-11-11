import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

np.set_printoptions(precision=4, linewidth=1000)


def main():

    def coeficientes_potencial_propios(index):
        return (1 / (2 * np.pi * 8.8541878188e-12)) * np.log(
            float(line[index][0]) * 2 / (float(line[index][2]) / 2)
        )

    def coeficientes_potencial_mutuos(index1, index2):
        distancia_imagen = np.sqrt(
            (float(line[index1][0]) - (-float(line[index2][0]))) ** 2
            + (float(line[index1][1]) - float(line[index2][1])) ** 2
        )
        distancia_conductor = np.sqrt(
            (float(line[index1][0]) - float(line[index2][0])) ** 2
            + (float(line[index1][1]) - float(line[index2][1])) ** 2
        )
        return (1 / (2 * np.pi * 8.8541878188e-12)) * np.log(
            distancia_imagen / distancia_conductor
        )

    def campo_electrico_horizontal_fase(
        carga_fase,
        punto_horizontal,
        punto_vertical,
        distancia_horizontal_fase,
        distancia_vertical_fase,
    ):
        return (carga_fase * (punto_horizontal - distancia_horizontal_fase)) / (
            (2 * np.pi * 8.8541878188 * 1e-12)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase - punto_vertical) ** 2
            )
        ) - (carga_fase * (punto_horizontal - distancia_horizontal_fase)) / (
            (2 * np.pi * 8.8541878188 * 1e-12)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase + punto_vertical) ** 2
            )
        )

    def campo_electrico_vertical_fase(
        carga_fase,
        punto_horizontal,
        punto_vertical,
        distancia_horizontal_fase,
        distancia_vertical_fase,
    ):
        return (carga_fase * (punto_vertical - distancia_vertical_fase)) / (
            (2 * np.pi * 8.8541878188 * 1e-12)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase - punto_vertical) ** 2
            )
        ) - (carga_fase * (punto_vertical + distancia_vertical_fase)) / (
            (2 * np.pi * 8.8541878188 * 1e-12)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase + punto_vertical) ** 2
            )
        )

    def distancia_conductor_punto(
        distancia_horizontal_fase,
        distancia_vertical_fase,
        punto_horizontal,
        punto_vertical,
    ):
        return np.sqrt(
            (distancia_horizontal_fase - punto_horizontal) ** 2
            + (distancia_vertical_fase - punto_vertical) ** 2
        )

    def vector_unitario_horizontal_phi(
        distancia_vertical_fase,
        punto_vertical,
        distancia_fase,
    ):
        return -(distancia_vertical_fase - punto_vertical) / distancia_fase

    def vector_unitario_vertical_phi(
        distancia_horizontal_fase,
        punto_horizontal,
        distancia_fase,
    ):
        return (distancia_horizontal_fase - punto_horizontal) / distancia_fase

    def campo_magnetico_horizontal_fase(
        corriente_nominal,
        punto_horizontal,
        punto_vertical,
        distancia_horizontal_fase,
        distancia_vertical_fase,
    ):
        return -(corriente_nominal * (punto_vertical - distancia_vertical_fase)) / (
            (2 * np.pi)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase - punto_vertical) ** 2
            )
        ) + (corriente_nominal * (punto_vertical + distancia_vertical_fase)) / (
            (2 * np.pi)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase + punto_vertical) ** 2
            )
        )

    def campo_magnetico_vertical_fase(
        corriente_nominal,
        punto_horizontal,
        punto_vertical,
        distancia_horizontal_fase,
        distancia_vertical_fase,
    ):
        return (corriente_nominal * (punto_horizontal - distancia_horizontal_fase)) / (
            (2 * np.pi)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase - punto_vertical) ** 2
            )
        ) - (corriente_nominal * (punto_horizontal - distancia_horizontal_fase)) / (
            (2 * np.pi)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase + punto_vertical) ** 2
            )
        )

    def campo_magnetico(corriente_fase, distancia_fase_punto, vector_unitario):
        return corriente_fase / (2 * np.pi * distancia_fase_punto) * vector_unitario

    def campo_magnetico_horizontal_fase_imagenes(
        corriente_nominal,
        punto_horizontal,
        punto_vertical,
        distancia_horizontal_fase,
        distancia_vertical_fase,
    ):
        return -(corriente_nominal * (punto_vertical - distancia_vertical_fase)) / (
            (2 * np.pi)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase - punto_vertical) ** 2
            )
        ) + (
            corriente_nominal
            * (
                punto_vertical
                + np.sqrt(resistividad_suelo / (np.pi * f * 4 * np.pi * 1e-7))
            )
        ) / (
            (2 * np.pi)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (
                    np.sqrt(resistividad_suelo / (np.pi * f * 4 * np.pi * 1e-7))
                    + punto_vertical
                )
                ** 2
            )
        )

    def campo_magnetico_vertical_fase_imagenes(
        corriente_nominal,
        punto_horizontal,
        punto_vertical,
        distancia_horizontal_fase,
        distancia_vertical_fase,
    ):
        return (corriente_nominal * (punto_horizontal - distancia_horizontal_fase)) / (
            (2 * np.pi)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (distancia_vertical_fase - punto_vertical) ** 2
            )
        ) - (corriente_nominal * (punto_horizontal - distancia_horizontal_fase)) / (
            (2 * np.pi)
            * (
                (punto_horizontal - distancia_horizontal_fase) ** 2
                + (
                    np.sqrt(resistividad_suelo / (np.pi * f * 4 * np.pi * 1e-7))
                    + punto_vertical
                )
                ** 2
            )
        )

    def comprobacion_retie(intensidad_campo_electrico, densidad_flujo_magnetico):
        if np.all(np.array(intensidad_campo_electrico) / 1000 < 4.16):
            st.write(
                "RETIE limits for electric field intensity for general public exposure up to eight continuous hours: :white_check_mark:"
            )
        else:
            st.write(
                "RETIE limits for electric field intensity for general public exposure for up to eight continuous hours: :x:"
            )

        if np.all(np.array(intensidad_campo_electrico) / 1000 < 8.3):
            st.write(
                "RETIE limits for electric field intensity for occupational exposure during an eight-hour workday: :white_check_mark:"
            )
        else:
            st.write(
                "RETIE limits for electric field intensity for occupational exposure during an eight-hour workday: :x:"
            )

        if (
            np.all(np.array(densidad_flujo_magnetico) * 1e6 < 200)
            and np.all(np.array(densidad_flujo_magnetico_ideal) * 1e6 < 200)
            and np.all(np.array(densidad_flujo_magnetico_real_imagenes) * 1e6 < 200)
        ):
            st.write(
                "RETIE limits for magnetic flux density for general public exposure for up to eight continuous hours: :white_check_mark:"
            )
        else:
            st.write(
                "RETIE limits for magnetic flux density for general public exposure for up to eight continuous hours: :x:"
            )

        if (
            np.all(np.array(densidad_flujo_magnetico) * 1e6 < 1000)
            and np.all(np.array(densidad_flujo_magnetico_ideal) * 1e6 < 1000)
            and np.all(np.array(densidad_flujo_magnetico_real_imagenes) * 1e6 < 1000)
        ):
            st.write(
                "RETIE limits for magnetic flux density for occupational exposure for an eight-hour workday: :white_check_mark:"
            )
        else:
            st.write(
                "RETIE limits for magnetic flux density for occupational exposure for an eight-hour workday: :x:"
            )

    st.title("Electric and Magnetic Field Analyzer")
    st.subheader(":blue[Juan David Sánchez Gámez]", divider=True)

    # Inputs
    st.header("Input Parameters")

    ruta_archivo = st.file_uploader(
        "Enter file path for the transmission line physical layout", type="xlsx"
    )
    voltaje_linea_linea = (
        st.number_input("Enter line-to-line voltage (kV)", min_value=0.0, step=0.1)
        * 1e3
    )
    corriente_nominal = st.number_input(
        "Enter nominal current of conductors (A)", min_value=0.0, step=0.1
    )
    longitud_zona_incertidumbre = st.number_input(
        "Enter length of uncertainty zone (m)", min_value=0.0, step=0.1
    )
    resistividad_suelo = st.number_input(
        "Enter soil resistivity (Ω·m)", min_value=0.0, step=0.1
    )

    if st.button("Analyze"):

        if ruta_archivo is not None:
            # Leer el archivo Excel
            df = pd.read_excel(ruta_archivo)

            # Procesar el archivo
            line = [np.array(row[1:]) for row in df.itertuples()]

        numero_fase_a = 0
        numero_fase_b = 0
        numero_fase_c = 0
        numero_guarda = 0

        for _ in range(len(line)):
            if line[_][3] == "A":
                numero_fase_a += 1
            elif line[_][3] == "B":
                numero_fase_b += 1
            elif line[_][3] == "C":
                numero_fase_c += 1
            else:
                numero_guarda += 1

        f = 60
        punto_vertical = 1

        coeficientes_potencial_matriz = np.zeros((len(line), len(line)), dtype=complex)

        for i in range(len(line)):
            for j in range(len(line)):
                if i == j:
                    coeficientes_potencial_matriz[i, j] = (
                        coeficientes_potencial_propios(i)
                    )
                else:
                    coeficientes_potencial_matriz[i, j] = coeficientes_potencial_mutuos(
                        i, j
                    )

        coeficientes_fase_fase = coeficientes_potencial_matriz[
            0 : (numero_fase_a + numero_fase_b + numero_fase_c),
            0 : (numero_fase_a + numero_fase_b + numero_fase_c),
        ]

        coeficientes_fase_guarda = coeficientes_potencial_matriz[
            0 : (numero_fase_a + numero_fase_b + numero_fase_c),
            (numero_fase_a + numero_fase_b + numero_fase_c) : (
                numero_fase_a + numero_fase_b + numero_fase_c
            )
            + numero_guarda,
        ]

        coeficientes_guarda_fase = coeficientes_potencial_matriz[
            (numero_fase_a + numero_fase_b + numero_fase_c) : (
                numero_fase_a + numero_fase_b + numero_fase_c
            )
            + numero_guarda,
            0 : (numero_fase_a + numero_fase_b + numero_fase_c),
        ]

        coeficientes_guarda_guarda = coeficientes_potencial_matriz[
            (numero_fase_a + numero_fase_b + numero_fase_c) : (
                numero_fase_a + numero_fase_b + numero_fase_c
            )
            + numero_guarda,
            (numero_fase_a + numero_fase_b + numero_fase_c) : (
                numero_fase_a + numero_fase_b + numero_fase_c
            )
            + numero_guarda,
        ]

        coeficientes_potencial_equivalente = coeficientes_fase_fase - (
            coeficientes_fase_guarda
            @ np.linalg.inv(coeficientes_guarda_guarda)
            @ coeficientes_guarda_fase
        )

        capacitancia = np.linalg.inv(coeficientes_potencial_equivalente)

        punto_horizontal = np.linspace(
            -longitud_zona_incertidumbre, longitud_zona_incertidumbre, 100
        )

        distancia_horizontal_conductor = np.zeros((len(line) - numero_guarda))
        distancia_vertical_conductor = np.zeros((len(line) - numero_guarda))

        for _ in range(len(line) - numero_guarda):
            distancia_horizontal_conductor[_] = float(line[_][1])
            distancia_vertical_conductor[_] = float(line[_][0])

        corriente_fase = corriente_nominal * np.array(
            [[1], [np.exp(2 / 3 * np.pi * 1j) ** 2], [np.exp(2 / 3 * np.pi * 1j)]]
        )

        corriente = np.vstack(
            [
                np.tile(corriente_fase[0], (numero_fase_a, 1)),
                np.tile(corriente_fase[1], (numero_fase_b, 1)),
                np.tile(corriente_fase[2], (numero_fase_a, 1)),
            ]
        )

        voltaje_linea_neutro = voltaje_linea_linea / np.sqrt(3)

        voltaje = voltaje_linea_neutro * np.array(
            [[1], [np.exp(2 / 3 * np.pi * 1j) ** 2], [np.exp(2 / 3 * np.pi * 1j)]]
        )

        voltaje_matriz = np.vstack(
            [
                np.tile(voltaje[0], (numero_fase_a, 1)),
                np.tile(voltaje[1], (numero_fase_b, 1)),
                np.tile(voltaje[2], (numero_fase_a, 1)),
            ]
        )

        carga_conductores = capacitancia @ voltaje_matriz

        # Campo eléctrico

        campo_electrico_horizontal_a = []
        campo_electrico_horizontal_b = []
        campo_electrico_horizontal_c = []

        campo_electrico_vertical_a = []
        campo_electrico_vertical_b = []
        campo_electrico_vertical_c = []

        for _ in range(numero_fase_a):
            campo_electrico_horizontal_a.append(
                campo_electrico_horizontal_fase(
                    carga_conductores[_],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_],
                    distancia_vertical_conductor[_],
                )
            )
            campo_electrico_vertical_a.append(
                campo_electrico_vertical_fase(
                    carga_conductores[_],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_],
                    distancia_vertical_conductor[_],
                )
            )

        for _ in range(numero_fase_b):
            campo_electrico_horizontal_b.append(
                campo_electrico_horizontal_fase(
                    carga_conductores[_ + numero_fase_a],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a],
                    distancia_vertical_conductor[_ + numero_fase_a],
                )
            )

            campo_electrico_vertical_b.append(
                campo_electrico_vertical_fase(
                    carga_conductores[_ + numero_fase_a],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a],
                    distancia_vertical_conductor[_ + numero_fase_a],
                )
            )

        for _ in range(numero_fase_c):
            campo_electrico_horizontal_c.append(
                campo_electrico_horizontal_fase(
                    carga_conductores[_ + numero_fase_a + numero_fase_b],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a + numero_fase_b],
                    distancia_vertical_conductor[_ + numero_fase_a + numero_fase_b],
                )
            )

            campo_electrico_vertical_c.append(
                campo_electrico_vertical_fase(
                    carga_conductores[_ + numero_fase_a + numero_fase_b],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a + numero_fase_b],
                    distancia_vertical_conductor[_ + numero_fase_a + numero_fase_b],
                )
            )

        campo_electrico_horizontal_a = np.array(campo_electrico_horizontal_a)
        campo_electrico_horizontal_b = np.array(campo_electrico_horizontal_b)
        campo_electrico_horizontal_c = np.array(campo_electrico_horizontal_c)

        campo_electrico_vertical_a = np.array(campo_electrico_vertical_a)
        campo_electrico_vertical_b = np.array(campo_electrico_vertical_b)
        campo_electrico_vertical_c = np.array(campo_electrico_vertical_c)

        campo_electrico_horizontal_resultante = (
            campo_electrico_horizontal_a
            + campo_electrico_horizontal_b
            + campo_electrico_horizontal_c
        )

        campo_electrico_vertical_resultante = (
            campo_electrico_vertical_a
            + campo_electrico_vertical_b
            + campo_electrico_vertical_c
        )

        campo_electrico_horizontal_resultante = np.sum(
            campo_electrico_horizontal_resultante, axis=0
        )
        campo_electrico_vertical_resultante = np.sum(
            campo_electrico_vertical_resultante, axis=0
        )

        campo_electrico_total = np.sqrt(
            np.real(campo_electrico_horizontal_resultante) ** 2
            + np.real(campo_electrico_vertical_resultante) ** 2
            + np.imag(campo_electrico_horizontal_resultante) ** 2
            + np.imag(campo_electrico_vertical_resultante) ** 2
        )

        # Flujo magnético con suelo real despreciando imágenes

        distancia_conductores_punto = []
        vector_phi_horizontal = []
        vector_phi_vertical = []

        campo_magnetico_horizontal_a = []
        campo_magnetico_horizontal_b = []
        campo_magnetico_horizontal_c = []

        campo_magnetico_vertical_a = []
        campo_magnetico_vertical_b = []
        campo_magnetico_vertical_c = []

        for _ in range(numero_fase_a + numero_fase_b + numero_fase_c):
            distancia_conductores_punto.append(
                distancia_conductor_punto(
                    distancia_horizontal_conductor[_],
                    distancia_vertical_conductor[_],
                    punto_horizontal,
                    punto_vertical,
                )
            )
            vector_phi_horizontal.append(
                vector_unitario_horizontal_phi(
                    distancia_vertical_conductor[_],
                    punto_vertical,
                    distancia_conductores_punto[_],
                )
            )

            vector_phi_vertical.append(
                vector_unitario_vertical_phi(
                    distancia_horizontal_conductor[_],
                    punto_horizontal,
                    distancia_conductores_punto[_],
                )
            )

        for _ in range(numero_fase_a):
            campo_magnetico_horizontal_a.append(
                campo_magnetico(
                    corriente[_],
                    distancia_conductores_punto[_],
                    vector_phi_horizontal[_],
                )
            )
            campo_magnetico_vertical_a.append(
                campo_magnetico(
                    corriente[_],
                    distancia_conductores_punto[_],
                    vector_phi_vertical[_],
                )
            )

        for _ in range(numero_fase_b):
            campo_magnetico_horizontal_b.append(
                campo_magnetico(
                    corriente[_ + numero_fase_a],
                    distancia_conductores_punto[_ + numero_fase_a],
                    vector_phi_horizontal[_ + numero_fase_a],
                )
            )
            campo_magnetico_vertical_b.append(
                campo_magnetico(
                    corriente[_ + numero_fase_a],
                    distancia_conductores_punto[_ + numero_fase_a],
                    vector_phi_vertical[_ + numero_fase_a],
                )
            )

        for _ in range(numero_fase_c):
            campo_magnetico_horizontal_c.append(
                campo_magnetico(
                    corriente[_ + numero_fase_a + numero_fase_b],
                    distancia_conductores_punto[_ + numero_fase_a + numero_fase_b],
                    vector_phi_horizontal[_ + numero_fase_a + numero_fase_b],
                )
            )
            campo_magnetico_vertical_c.append(
                campo_magnetico(
                    corriente[_ + numero_fase_a + numero_fase_b],
                    distancia_conductores_punto[_ + numero_fase_a + numero_fase_b],
                    vector_phi_vertical[_ + numero_fase_a + numero_fase_b],
                )
            )

        campo_magnetico_horizontal_a = np.array(campo_magnetico_horizontal_a)
        campo_magnetico_horizontal_b = np.array(campo_magnetico_horizontal_b)
        campo_magnetico_horizontal_c = np.array(campo_magnetico_horizontal_c)

        campo_magnetico_vertical_a = np.array(campo_magnetico_vertical_a)
        campo_magnetico_vertical_b = np.array(campo_magnetico_vertical_b)
        campo_magnetico_vertical_c = np.array(campo_magnetico_vertical_c)

        campo_magnetico_horizontal_resultante = (
            campo_magnetico_horizontal_a
            + campo_magnetico_horizontal_b
            + campo_magnetico_horizontal_c
        )

        campo_magnetico_vertical_resultante = (
            campo_magnetico_vertical_a
            + campo_magnetico_vertical_b
            + campo_magnetico_vertical_c
        )

        campo_magnetico_horizontal_resultante = np.sum(
            campo_magnetico_horizontal_resultante, axis=0
        )

        campo_magnetico_vertical_resultante = np.sum(
            campo_magnetico_vertical_resultante, axis=0
        )

        campo_magnetico_total = np.sqrt(
            (
                np.real(campo_magnetico_horizontal_resultante) ** 2
                + np.real(campo_magnetico_vertical_resultante) ** 2
            )
            + (
                np.imag(campo_magnetico_horizontal_resultante) ** 2
                + np.imag(campo_magnetico_vertical_resultante) ** 2
            )
        )

        densidad_flujo_magnetico = campo_magnetico_total * 4 * np.pi * 1e-7

        # Flujo magnético ideal

        campo_magnetico_ideal_horizontal_a = []
        campo_magnetico_ideal_horizontal_b = []
        campo_magnetico_ideal_horizontal_c = []

        campo_magnetico_ideal_vertical_a = []
        campo_magnetico_ideal_vertical_b = []
        campo_magnetico_ideal_vertical_c = []

        for _ in range(numero_fase_a):
            campo_magnetico_ideal_horizontal_a.append(
                campo_magnetico_horizontal_fase(
                    corriente[_],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_],
                    distancia_vertical_conductor[_],
                )
            )
            campo_magnetico_ideal_vertical_a.append(
                campo_magnetico_vertical_fase(
                    corriente[_],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_],
                    distancia_vertical_conductor[_],
                )
            )

        for _ in range(numero_fase_b):
            campo_magnetico_ideal_horizontal_b.append(
                campo_magnetico_horizontal_fase(
                    corriente[_ + numero_fase_a],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a],
                    distancia_vertical_conductor[_ + numero_fase_a],
                )
            )
            campo_magnetico_ideal_vertical_b.append(
                campo_magnetico_vertical_fase(
                    corriente[_ + numero_fase_a],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a],
                    distancia_vertical_conductor[_ + numero_fase_a],
                )
            )

        for _ in range(numero_fase_c):
            campo_magnetico_ideal_horizontal_c.append(
                campo_magnetico_horizontal_fase(
                    corriente[_ + numero_fase_a + numero_fase_b],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a + numero_fase_b],
                    distancia_vertical_conductor[_ + numero_fase_a + numero_fase_b],
                )
            )
            campo_magnetico_ideal_vertical_c.append(
                campo_magnetico_vertical_fase(
                    corriente[_ + numero_fase_a + numero_fase_b],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a + numero_fase_b],
                    distancia_vertical_conductor[_ + numero_fase_a + numero_fase_b],
                )
            )

        campo_magnetico_ideal_horizontal_a = np.array(
            campo_magnetico_ideal_horizontal_a
        )
        campo_magnetico_ideal_horizontal_b = np.array(
            campo_magnetico_ideal_horizontal_b
        )
        campo_magnetico_horizontal_c = np.array(campo_magnetico_ideal_horizontal_c)

        campo_magnetico_ideal_vertical_a = np.array(campo_magnetico_ideal_vertical_a)
        campo_magnetico_ideal_vertical_b = np.array(campo_magnetico_ideal_vertical_b)
        campo_magnetico_ideal_vertical_c = np.array(campo_magnetico_ideal_vertical_c)

        campo_magnetico_ideal_horizontal_resultante = (
            campo_magnetico_ideal_horizontal_a
            + campo_magnetico_ideal_horizontal_b
            + campo_magnetico_ideal_horizontal_c
        )

        campo_magnetico_ideal_vertical_resultante = (
            campo_magnetico_ideal_vertical_a
            + campo_magnetico_ideal_vertical_b
            + campo_magnetico_ideal_vertical_c
        )

        campo_magnetico_ideal_horizontal_resultante = np.sum(
            campo_magnetico_ideal_horizontal_resultante, axis=0
        )
        campo_magnetico_ideal_vertical_resultante = np.sum(
            campo_magnetico_ideal_vertical_resultante, axis=0
        )

        campo_magnetico_ideal_total = np.sqrt(
            (
                np.real(campo_magnetico_ideal_horizontal_resultante) ** 2
                + np.real(campo_magnetico_ideal_vertical_resultante) ** 2
            )
            + (
                np.imag(campo_magnetico_ideal_horizontal_resultante) ** 2
                + np.imag(campo_magnetico_ideal_vertical_resultante) ** 2
            )
        )

        densidad_flujo_magnetico_ideal = campo_magnetico_ideal_total * 4 * np.pi * 1e-7

        # Flujo magnético con resistividad del suelo real y teniendo en cuenta el efecto de imágenes

        campo_magnetico_real_imagenes_horizontal_a = []
        campo_magnetico_real_imagenes_horizontal_b = []
        campo_magnetico_real_imagenes_horizontal_c = []

        campo_magnetico_real_imagenes_vertical_a = []
        campo_magnetico_real_imagenes_vertical_b = []
        campo_magnetico_real_imagenes_vertical_c = []

        for _ in range(numero_fase_a):
            campo_magnetico_real_imagenes_horizontal_a.append(
                campo_magnetico_horizontal_fase_imagenes(
                    corriente[_],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_],
                    distancia_vertical_conductor[_],
                )
            )
            campo_magnetico_real_imagenes_vertical_a.append(
                campo_magnetico_vertical_fase_imagenes(
                    corriente[_],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_],
                    distancia_vertical_conductor[_],
                )
            )

        for _ in range(numero_fase_b):
            campo_magnetico_real_imagenes_horizontal_b.append(
                campo_magnetico_horizontal_fase_imagenes(
                    corriente[_ + numero_fase_a],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a],
                    distancia_vertical_conductor[_ + numero_fase_a],
                )
            )
            campo_magnetico_real_imagenes_vertical_b.append(
                campo_magnetico_vertical_fase_imagenes(
                    corriente[_ + numero_fase_a],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a],
                    distancia_vertical_conductor[_ + numero_fase_a],
                )
            )

        for _ in range(numero_fase_c):
            campo_magnetico_real_imagenes_horizontal_c.append(
                campo_magnetico_horizontal_fase_imagenes(
                    corriente[_ + numero_fase_a + numero_fase_b],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a + numero_fase_b],
                    distancia_vertical_conductor[_ + numero_fase_a + numero_fase_b],
                )
            )
            campo_magnetico_real_imagenes_vertical_c.append(
                campo_magnetico_vertical_fase_imagenes(
                    corriente[_ + numero_fase_a + numero_fase_b],
                    punto_horizontal,
                    punto_vertical,
                    distancia_horizontal_conductor[_ + numero_fase_a + numero_fase_b],
                    distancia_vertical_conductor[_ + numero_fase_a + numero_fase_b],
                )
            )

        campo_magnetico_real_imagenes_horizontal_a = np.array(
            campo_magnetico_real_imagenes_horizontal_a
        )
        campo_magnetico_real_imagenes_horizontal_b = np.array(
            campo_magnetico_real_imagenes_horizontal_b
        )
        campo_magnetico_real_imagenes_horizontal_c = np.array(
            campo_magnetico_real_imagenes_horizontal_c
        )

        campo_magnetico_real_imagenes_vertical_a = np.array(
            campo_magnetico_real_imagenes_vertical_a
        )
        campo_magnetico_real_imagenes_vertical_b = np.array(
            campo_magnetico_real_imagenes_vertical_b
        )
        campo_magnetico_real_imagenes_vertical_c = np.array(
            campo_magnetico_real_imagenes_vertical_c
        )

        campo_magnetico_real_imagenes_horizontal_resultante = (
            campo_magnetico_real_imagenes_horizontal_a
            + campo_magnetico_real_imagenes_horizontal_b
            + campo_magnetico_real_imagenes_horizontal_c
        )

        campo_magnetico_real_imagenes_vertical_resultante = (
            campo_magnetico_real_imagenes_vertical_a
            + campo_magnetico_real_imagenes_vertical_b
            + campo_magnetico_real_imagenes_vertical_c
        )

        campo_magnetico_real_imagenes_horizontal_resultante = np.sum(
            campo_magnetico_real_imagenes_horizontal_resultante, axis=0
        )

        campo_magnetico_real_imagenes_vertical_resultante = np.sum(
            campo_magnetico_real_imagenes_vertical_resultante, axis=0
        )

        campo_magnetico_real_imagenes_total = np.sqrt(
            (
                np.real(campo_magnetico_real_imagenes_horizontal_resultante) ** 2
                + np.real(campo_magnetico_real_imagenes_vertical_resultante) ** 2
            )
            + (
                np.imag(campo_magnetico_real_imagenes_horizontal_resultante) ** 2
                + np.imag(campo_magnetico_real_imagenes_vertical_resultante) ** 2
            )
        )

        densidad_flujo_magnetico_real_imagenes = (
            campo_magnetico_real_imagenes_total * 4 * np.pi * 1e-7
        )

        # RETIE Compliance Checks
        st.header("RETIE Compliance Checks")
        comprobacion_retie(campo_electrico_total, densidad_flujo_magnetico)

        # Plots
        st.header("Visualization")

        st.subheader("Electric Field")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(punto_horizontal, campo_electrico_total / 1000)
        ax.set_ylabel("Electric Field Intensity (kV/m)")
        ax.set_xlabel("Easement (m)")
        st.pyplot(fig)

        st.subheader("Magnetic Flux Density (Real Soil, No Images)")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(punto_horizontal, densidad_flujo_magnetico * 1e6)
        ax.set_title("Real Soil, Neglecting Image Effect")
        ax.set_ylabel("Magnetic Flux Density (μT)")
        ax.set_xlabel("Easement (m)")
        st.pyplot(fig)

        st.subheader("Magnetic Flux Density (Ideal Soil)")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(punto_horizontal, densidad_flujo_magnetico_ideal * 1e6)
        ax.set_title("Ideal Soil")
        ax.set_ylabel("Magnetic Flux Density (μT)")
        ax.set_xlabel("Easement (m)")
        st.pyplot(fig)

        st.subheader("Magnetic Flux Density (Real Soil with Image Effect)")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(punto_horizontal, densidad_flujo_magnetico_real_imagenes * 1e6)
        ax.set_title("Real Soil with Image Effect")
        ax.set_ylabel("Magnetic Flux Density (μT)")
        ax.set_xlabel("Easement (m)")
        st.pyplot(fig)

        # Outputs

        st.header("Data")

        st.subheader("Electric Field")
        st.write(f"Total electric field ($E_T$): {campo_electrico_total / 1000} kV/m")

        st.subheader("Magnetic Field")
        st.write(f"Total magnetic field ($H_T$): {campo_magnetico_total} A/m")

        st.subheader("Magnetic Flux Density")
        st.write(f"Magnetic flux density ($B$): {densidad_flujo_magnetico * 1e6} μT")


if __name__ == "__main__":
    main()
