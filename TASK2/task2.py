# Module 14 - Introduction to Deep Learning
# Task 2 - Neural Network Basics

import numpy as np
import matplotlib.pyplot as plt


# Create figure

plt.figure(figsize=(12, 7))


# Define neuron positions

input_layer = np.array([
    [1, 3],
    [1, 2],
    [1, 1]
])

hidden_layer = np.array([
    [4, 3],
    [4, 2],
    [4, 1]
])

output_layer = np.array([
    [7, 2]
])


# Draw connections between input and hidden layer

for i in range(len(input_layer)):
    for j in range(len(hidden_layer)):

        x_values = [
            input_layer[i][0],
            hidden_layer[j][0]
        ]

        y_values = [
            input_layer[i][1],
            hidden_layer[j][1]
        ]

        plt.plot(x_values, y_values, 'gray', linewidth=1)

        # Calculate middle point for weight label

        x_mid = (x_values[0] + x_values[1]) / 2
        y_mid = (y_values[0] + y_values[1]) / 2

        plt.text(
            x_mid,
            y_mid,
            f"W{i+1}{j+1}",
            fontsize=8,
            ha='center',
            va='bottom'
        )


# Draw connections between hidden and output layer

for i in range(len(hidden_layer)):

    x_values = [
        hidden_layer[i][0],
        output_layer[0][0]
    ]

    y_values = [
        hidden_layer[i][1],
        output_layer[0][1]
    ]

    plt.plot(x_values, y_values, 'gray', linewidth=1)

    # Calculate middle point for weight label

    x_mid = (x_values[0] + x_values[1]) / 2
    y_mid = (y_values[0] + y_values[1]) / 2

    plt.text(
        x_mid,
        y_mid,
        f"W{i+1}O",
        fontsize=8,
        ha='center',
        va='bottom'
    )


# Draw input neurons

plt.scatter(
    input_layer[:, 0],
    input_layer[:, 1],
    s=1000,
    zorder=5
)


# Draw hidden neurons

plt.scatter(
    hidden_layer[:, 0],
    hidden_layer[:, 1],
    s=1000,
    zorder=5
)


# Draw output neuron

plt.scatter(
    output_layer[:, 0],
    output_layer[:, 1],
    s=1000,
    zorder=5
)


# Label input neurons

plt.text(1, 3, "X1", ha='center', va='center', fontsize=10)
plt.text(1, 2, "X2", ha='center', va='center', fontsize=10)
plt.text(1, 1, "X3", ha='center', va='center', fontsize=10)


# Label hidden neurons

plt.text(4, 3, "H1", ha='center', va='center', fontsize=10)
plt.text(4, 2, "H2", ha='center', va='center', fontsize=10)
plt.text(4, 1, "H3", ha='center', va='center', fontsize=10)


# Label output neuron

plt.text(7, 2, "Y", ha='center', va='center', fontsize=10)


# Layer labels

plt.text(
    1,
    3.8,
    "Input Layer",
    ha='center',
    fontsize=13,
    fontweight='bold'
)

plt.text(
    4,
    3.8,
    "Hidden Layer",
    ha='center',
    fontsize=13,
    fontweight='bold'
)

plt.text(
    7,
    3.8,
    "Output Layer",
    ha='center',
    fontsize=13,
    fontweight='bold'
)


# Bias labels

plt.text(
    4,
    0.35,
    "Bias: b1, b2, b3",
    ha='center',
    fontsize=10
)

plt.text(
    7,
    1.2,
    "Bias: b",
    ha='center',
    fontsize=10
)


# Data flow arrow

plt.annotate(
    "Data Flow",
    xy=(6.5, 2),
    xytext=(1.5, 4.5),
    arrowprops=dict(arrowstyle="->", linewidth=2),
    fontsize=12
)


# Remove axes

plt.xlim(0, 8)
plt.ylim(0, 5)
plt.axis("off")


# Title

plt.title(
    "Simple Neural Network",
    fontsize=16,
    fontweight='bold'
)


plt.show()