# 🚀 Double Inverted Pendulum Control using Reinforcement Learning

## 📌 Project Overview

This project implements a **custom 2D physics-based environment** using `pymunk` and trains a Reinforcement Learning (RL) agent using **Proximal Policy Optimization (PPO)** to balance a **double inverted pendulum**.

The system consists of a cart moving horizontally with two connected poles. The objective is to apply forces to the cart such that both poles remain upright — a highly unstable and complex control problem.

This project demonstrates:

* Custom RL environment design
* Physics simulation
* Reward engineering
* End-to-end RL workflow (training → evaluation → analysis)
* Full reproducibility using Docker

---

## 🧠 Environment Design

The environment is implemented in `environment.py` following the Gymnasium API.

### 🔧 Components

* **Cart**: Moves along a horizontal track
* **Pole 1**: Attached to cart
* **Pole 2**: Attached to pole 1
* **Physics Engine**: `pymunk`
* **Rendering**: `pygame`

### 📊 Observation Space (6D)

The agent observes:

* Cart Position (x)
* Cart Velocity (vx)
* Pole 1 Angle (θ₁)
* Pole 1 Angular Velocity (ω₁)
* Pole 2 Angle (θ₂)
* Pole 2 Angular Velocity (ω₂)

```python
shape = (6,)
```

---

### 🎮 Action Space (1D)

* Continuous force applied to cart

```python
range = [-1.0, 1.0]
```

---

### ⏱ Simulation

* Physics timestep: **1/60 seconds**
* Uses realistic constraints and joints

---

## 🎯 Reward Function Design

Two reward strategies are implemented:

---

### 🔹 Baseline Reward

Encourages poles to remain upright:

```
reward = cos(θ₁) + cos(θ₂)
```

* Maximum reward = **2** (perfect upright)
* Simple and sparse signal

---

### 🔹 Shaped Reward (Improved Learning)

```
reward = 
    cos(θ₁) + cos(θ₂)
    - 0.1 * |cart_position|
    - 0.01 * (|ω₁| + |ω₂|)
    - 0.001 * (action²)
```

### 📌 Rationale

| Component        | Purpose                 |
| ---------------- | ----------------------- |
| Upright bonus    | Core objective          |
| Cart penalty     | Keeps cart centered     |
| Velocity penalty | Encourages stability    |
| Action penalty   | Reduces excessive force |

✔ Result: Faster and more stable learning

---

## 🤖 Training

Training is handled in `train.py` using PPO from Stable-Baselines3.

### Features:

* Configurable via CLI
* Logs training metrics
* Saves trained model

### Example:

```bash
python train.py --timesteps 200000 --reward_type shaped
```

---

## 🎮 Evaluation

The trained agent is evaluated in `evaluate.py`.

### Features:

* Loads trained model
* Runs simulation with rendering
* Generates GIFs

### Example:

```bash
python evaluate.py --model_path models/ppo_model.zip
```

---

## 🎥 Outputs

### 📊 Learning Curve

* `reward_comparison.png`
* Compares:

  * Baseline reward
  * Shaped reward

---

### 🎞 GIF Demonstrations

Located in `media/`:

* `agent_initial.gif` → Early training behavior
* `agent_final.gif` → Fully trained agent

---

## 🐳 Docker Setup (Reproducibility)

This project is fully containerized.

---

### 🔧 Build

```bash
docker-compose build
```

---

### ▶️ Train

```bash
docker-compose run train
```

---

### 🎮 Evaluate

```bash
docker-compose run evaluate
```

---

## ⚙️ Environment Variables

Defined in `.env.example`

Example:

```
TIMESTEPS=200000
REWARD_TYPE=shaped
MODEL_PATH=models/ppo_model.zip
```

---

## 📁 Project Structure

```
.
├── environment.py
├── train.py
├── evaluate.py
├── plot.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
├── logs/
├── models/
├── media/
└── reward_comparison.png
```

---

## 📊 Logging & Analysis

* Logs saved in `logs/monitor.csv`
* Plot generated using:

```bash
python plot.py
```

---

## ✅ Evaluation Checklist

✔ Docker builds successfully
✔ Training runs without errors
✔ Evaluation renders simulation
✔ Logs generated
✔ Model saved
✔ Plot generated
✔ GIFs present
✔ README complete

---

## 🚨 Notes

* Do **not commit**:

  * Models (`.zip`)
  * Logs (`.csv`)
* All outputs should be reproducible

---

## 🏁 Conclusion

This project demonstrates a complete Reinforcement Learning pipeline for solving a complex control problem. It highlights the importance of:

* Proper environment design
* Effective reward shaping
* Stable RL algorithms (PPO)
* Reproducible workflows using Docker

---

## 📌 Future Improvements

* Hyperparameter tuning
* Curriculum learning
* More advanced reward shaping
* Real-world robotics applications

---

## 🙌 Acknowledgements

* Stable-Baselines3
* Pymunk Physics Engine
* OpenAI Gymnasium API

---
