
---

# 📊 Optimal Control of COVID-19 (SEIQ Model)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Scientific Computing](https://img.shields.io/badge/SciPy-Integrated-green.svg)
![Mathematics](https://img.shields.io/badge/Control%20Theory-Pontryagin-red.svg)

This project explores the mathematical modeling of COVID-19 transmission using an augmented **SEIQ (Susceptible-Exposed-Infectious-Quarantined-Recovered)** model. It focuses on using **Optimal Control Theory** to minimize the spread of the virus while optimizing the cost of interventions like vaccination and treatment.

## 📖 Background
The model divides the population into five compartments:
- **S (Susceptible):** Individuals at risk.
- **E (Exposed):** Infected individuals in the latent period.
- **I (Infectious):** Symptomatic individuals spreading the virus.
- **Q (Quarantined):** Individuals removed from the infectious pool.
- **R (Recovered):** Individuals with immunity.



The project uses **Pontryagin’s Maximum Principle** to derive first-order necessary conditions for optimal control strategies.

---

## ➗ Mathematical Foundation (Optimal Control)

The primary objective is to minimize the number of infected individuals while keeping the cost of control measures (vaccination and treatment) as low as possible.

### 1. The Objective Function
We define the cost functional $J$ as:
$$J(u_1, u_2) = \int_{0}^{T} (A_1 E + A_2 I + \frac{1}{2} B_1 u_1^2 + \frac{1}{2} B_2 u_2^2) dt$$

Where:
- $u_1$: Effort related to preventive measures (e.g., social distancing/masks).
- $u_2$: Effort related to medical treatment/quarantine efficiency.
- $A_i, B_i$: Weight constants representing the relative importance of each term.

### 2. The Hamiltonian
To solve this, we construct the Hamiltonian ($H$):
$$H = (A_1 E + A_2 I + \frac{1}{2} B_1 u_1^2 + \frac{1}{2} B_2 u_2^2) + \lambda_S \frac{dS}{dt} + \lambda_E \frac{dE}{dt} + \lambda_I \frac{dI}{dt} + \lambda_Q \frac{dQ}{dt} + \lambda_R \frac{dR}{dt}$$

Where $\lambda_i$ are the **adjoint variables** (shadow prices) associated with each state.

### 3. Optimality Conditions
Using Pontryagin’s Maximum Principle, the optimal controls $u_1^*$ and $u_2^*$ are derived by taking the partial derivatives of $H$ with respect to the controls:
$$\frac{\partial H}{\partial u_1} = 0 \implies u_1^* = \min(1, \max(0, \text{expression in } \lambda))$$

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the required scientific libraries installed:
```bash
pip install numpy matplotlib scipy pandas
```

## 🛠️ Usage
1.  **Simulation:** Run the ODE solver to visualize the natural progression of the disease.
2.  **Sensitivity Analysis:** Evaluate how parameters like $\beta$ (transmission) affect the peak.
3.  **Optimal Control:** Execute the Hamiltonian optimization to see the "Flattening of the Curve."



[Image of flattening the curve graph COVID-19]


## 📈 Key Results
- **Stability Analysis:** The model performs Local and Global stability checks based on Routh criterion and Lyapunov theorem.
- **Efficiency:** Demonstrates that early intervention significantly reduces the total infected population at a lower long-term economic cost.

## 🎓 Credits
Developed based on the research provided in the `optimal-control-of-covial-19.ipynb` notebook.