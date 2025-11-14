"""
XPLIA TIER 1 Advanced Features - Comprehensive Demo

Demonstrates all 6 cutting-edge TIER 1 modules:
1. Multimodal AI Explainability (Vision-Language, Diffusion)
2. Graph Neural Networks (GNN, Molecular, Knowledge Graphs)
3. Reinforcement Learning (Policy, Q-values, Trajectories)
4. Advanced Counterfactuals (Minimal, Feasible, Diverse, Actionable)
5. Time Series (Temporal, Forecasting, Anomaly)
6. Generative Models (VAE, GAN, StyleGAN)

Author: XPLIA Team
License: MIT
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print(" " * 35 + "XPLIA TIER 1 FEATURES DEMO")
print(" " * 20 + "6 Cutting-Edge Modules - State-of-the-Art 2024-2025")
print("=" * 100)

# =============================================================================
# MODULE 1: MULTIMODAL AI EXPLAINABILITY
# =============================================================================

print("\n" + "=" * 100)
print("MODULE 1: MULTIMODAL AI EXPLAINABILITY")
print("=" * 100)

from xplia.explainers.multimodal import (
    CLIPExplainer,
    StableDiffusionExplainer,
    MultimodalCounterfactualExplainer
)

print("\n1.1 CLIP Vision-Language Explanation")
print("-" * 100)

clip = CLIPExplainer(None)
image = np.random.rand(224, 224, 3)
text = "A golden retriever playing fetch in a sunny park"

clip_exp = clip.explain(image, text, method='attention')
print(f"Text: {text}")
print(f"Image-text similarity: {clip_exp.similarity_score:.4f}")
print(f"Cross-modal attention shape: {clip_exp.cross_modal_attention.shape}")

tokens = text.split()
print(f"\nWhich words attend to which image regions:")
for i, token in enumerate(tokens[:4]):
    top_patch = np.argmax(clip_exp.cross_modal_attention[:, i])
    print(f"  '{token}' → image patch {top_patch}")

print("\n1.2 Stable Diffusion Generation Explanation")
print("-" * 100)

sd = StableDiffusionExplainer(None)
prompt = "A cyberpunk city at night with neon lights"
gen_image = np.random.rand(512, 512, 3)

sd_exp = sd.explain(prompt, gen_image)
print(f"Prompt: {prompt}")
print(f"\nPrompt token attribution:")
for token, attr in zip(prompt.split()[:5], sd_exp.prompt_attribution[:5]):
    print(f"  '{token}': {attr:.4f}")

print(f"\nImportant concepts detected:")
for concept, score in list(sd_exp.concept_attribution.items())[:5]:
    if score > 0.3:
        print(f"  {concept}: {score:.4f}")

# =============================================================================
# MODULE 2: GRAPH NEURAL NETWORKS EXPLAINABILITY
# =============================================================================

print("\n" + "=" * 100)
print("MODULE 2: GRAPH NEURAL NETWORKS EXPLAINABILITY")
print("=" * 100)

from xplia.explainers.graph import (
    GNNExplainer,
    MolecularGNNExplainer,
    SubgraphXExplainer
)

print("\n2.1 GNN Node Classification Explanation")
print("-" * 100)

# Create graph
n_nodes = 20
edges = [(i, (i+1) % n_nodes) for i in range(n_nodes)]  # Ring
edges += [(i, (i+5) % n_nodes) for i in range(0, n_nodes, 5)]  # Extra connections

graph = {
    'nodes': list(range(n_nodes)),
    'edges': edges,
    'features': np.random.randn(n_nodes, 16),
    'adj_matrix': np.zeros((n_nodes, n_nodes))
}

for u, v in edges:
    graph['adj_matrix'][u, v] = 1
    graph['adj_matrix'][v, u] = 1

gnn_exp = GNNExplainer(None)
node_exp = gnn_exp.explain_node(graph, node_idx=5, top_k_edges=6)

print(f"Explaining node 5 in social network graph")
print(f"Important subgraph: {len(node_exp.subgraph_nodes)} nodes")
print(f"Important nodes: {sorted(list(node_exp.subgraph_nodes))[:6]}")

print("\n2.2 Molecular Property Prediction (Drug Discovery)")
print("-" * 100)

molecule = {
    'atoms': ['C'] * 9 + ['O'] * 4 + ['N'] * 2,
    'bonds': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)] + [(i, i+1) for i in range(6, 13)],
    'coords': None
}

mol_exp = MolecularGNNExplainer(None)
activity_exp = mol_exp.explain_molecule(molecule, property_name='bioactivity')

print(f"Molecule: {len(molecule['atoms'])} atoms")
print(f"Predicted property: {activity_exp.metadata['property']}")
print(f"\nImportant functional groups:")
for group in activity_exp.functional_groups:
    print(f"  - {group}")

print(f"\nPharmacophore features:")
for feature, count in activity_exp.pharmacophore.items():
    print(f"  {feature}: {count}")

# =============================================================================
# MODULE 3: REINFORCEMENT LEARNING EXPLAINABILITY
# =============================================================================

print("\n" + "=" * 100)
print("MODULE 3: REINFORCEMENT LEARNING EXPLAINABILITY")
print("=" * 100)

from xplia.explainers.reinforcement import (
    PolicyExplainer,
    QValueExplainer,
    TrajectoryExplainer
)

print("\n3.1 Policy Gradient Explanation")
print("-" * 100)

state = np.array([0.1, 0.5, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4])
action = 2

policy_exp = PolicyExplainer(None)
rl_exp = policy_exp.explain(state, action)

print(f"State: {state[:4]}...")
print(f"Action taken: {action}")
print(f"Action probabilities: {rl_exp.action_importance}")
print(f"Q-value decomposition:")
for component, value in rl_exp.q_value_decomposition.items():
    print(f"  {component}: {value:.4f}")

print("\n3.2 Trajectory Explanation (Episode Analysis)")
print("-" * 100)

states = [np.random.randn(8) for _ in range(5)]
actions = [0, 1, 1, 2, 3]
rewards = [0.1, 0.2, 0.5, 1.0, 2.0]

traj_exp = TrajectoryExplainer(None)
trajectory = traj_exp.explain_trajectory(states, actions, rewards)

print(f"Episode: {len(trajectory)} steps, total reward: {sum(rewards):.2f}")
for step in trajectory[:3]:
    print(f"\nStep {step['timestep']}:")
    print(f"  Action: {step['action']}, Reward: {step['reward']:.2f}")
    print(f"  Why: {step['why_action']}")

# =============================================================================
# MODULE 4: ADVANCED COUNTERFACTUAL GENERATION
# =============================================================================

print("\n" + "=" * 100)
print("MODULE 4: ADVANCED COUNTERFACTUAL GENERATION")
print("=" * 100)

from xplia.explainers.counterfactuals import (
    MinimalCounterfactualGenerator,
    FeasibleCounterfactualGenerator,
    ActionableRecourseGenerator
)

print("\n4.1 Minimal Counterfactual (Smallest Change)")
print("-" * 100)

x_original = np.array([0.3, 0.5, 0.8, 0.2, 0.9])
print(f"Original instance: {x_original}")

min_cf_gen = MinimalCounterfactualGenerator(None)
min_cf = min_cf_gen.generate(x_original, target_class=1)

print(f"Counterfactual: {min_cf.instance}")
print(f"L2 distance: {min_cf.distance:.4f}")
print(f"Changes ({len(min_cf.changes)} features):")
for idx, (old, new) in list(min_cf.changes.items())[:3]:
    print(f"  Feature {idx}: {old:.3f} → {new:.3f}")

print("\n4.2 Feasible Counterfactual (Respects Constraints)")
print("-" * 80)

constraints = {0: (0, 1), 1: (0, 1), 2: (0.5, 1.0)}
immutable = [4]  # Cannot change feature 4

feas_gen = FeasibleCounterfactualGenerator(None, constraints, immutable)
feas_cf = feas_gen.generate(x_original, target_class=1)

print(f"Constraints: {constraints}")
print(f"Immutable features: {immutable}")
print(f"Feasibility score: {feas_cf.feasibility_score}")
print(f"Feature 4 unchanged: {x_original[4] == feas_cf.instance[4]}")

print("\n4.3 Actionable Recourse (What User CAN Change)")
print("-" * 100)

actionable_features = [1, 2, 3]
costs = {1: 1.0, 2: 2.0, 3: 5.0}

action_gen = ActionableRecourseGenerator(None, actionable_features, costs)
recourse = action_gen.generate(x_original, target_class=1)

print(f"Actionable recommendations:")
for action in recourse['actions'][:3]:
    print(f"  - {action['action']}")
    print(f"    Cost: ${action['cost']:.0f}, Difficulty: {action['difficulty']}")
print(f"\nTotal cost: ${recourse['total_cost']:.2f}")

# =============================================================================
# MODULE 5: TIME SERIES EXPLAINABILITY
# =============================================================================

print("\n" + "=" * 100)
print("MODULE 5: TIME SERIES EXPLAINABILITY")
print("=" * 100)

from xplia.explainers.timeseries import (
    TemporalImportanceExplainer,
    ForecastExplainer,
    AnomalyExplainer
)

print("\n5.1 Temporal Importance (Which Timesteps Matter)")
print("-" * 100)

T, n_features = 100, 3
time_series = np.random.randn(T, n_features)
time_series[:, 0] += np.linspace(0, 10, T)  # Trend

temp_exp = TemporalImportanceExplainer(None)
ts_exp = temp_exp.explain(time_series, horizon=5)

print(f"Time series: {T} timesteps, {n_features} features")
print(f"Forecast horizon: {ts_exp.metadata['horizon']}")
print(f"\nContributions:")
print(f"  Trend: {ts_exp.trend_contribution:.2%}")
print(f"  Seasonality: {ts_exp.seasonality_contribution:.2%}")

print(f"\nTop 3 important timesteps:")
top_times = np.argsort(ts_exp.temporal_importance)[-3:][::-1]
for t in top_times:
    print(f"  t={t}: {ts_exp.temporal_importance[t]:.4f}")

print("\n5.2 Forecast Explanation")
print("-" * 100)

forecast = np.random.randn(10)
forecast_exp = ForecastExplainer(None)
fcast_exp = forecast_exp.explain_forecast(time_series, forecast, horizon=10)

print(f"Forecast: {fcast_exp['forecast'][:3]} ... (10 steps)")
print(f"\nForecast components:")
for comp, val in fcast_exp['components'].items():
    print(f"  {comp}: {val:.4f}")

print("\n5.3 Anomaly Detection Explanation")
print("-" * 100)

anomaly_idx = 42
time_series[anomaly_idx, 0] = 100  # Insert spike

anom_exp = AnomalyExplainer(None)
anom_result = anom_exp.explain_anomaly(time_series, anomaly_idx)

print(f"Anomaly at timestep {anomaly_idx}")
print(f"Anomaly score: {anom_result['anomaly_score']:.4f}")
print(f"Expected: {anom_result['expected_value']:.2f}, Actual: {anom_result['actual_value']:.2f}")
print(f"\nReasons:")
for reason in anom_result['reasons']:
    print(f"  - {reason['type']}: {reason['description']}")

# =============================================================================
# MODULE 6: GENERATIVE MODELS EXPLAINABILITY
# =============================================================================

print("\n" + "=" * 100)
print("MODULE 6: GENERATIVE MODELS EXPLAINABILITY")
print("=" * 100)

from xplia.explainers.generative import (
    VAEExplainer,
    GANExplainer,
    StyleGANExplainer
)

print("\n6.1 VAE Latent Space Explanation")
print("-" * 100)

z_vae = np.random.randn(64)
vae_exp = VAEExplainer(None)
latent_exp = vae_exp.explain_latent(z_vae)

print(f"Latent dimensions: {len(latent_exp.latent_dims_importance)}")
print(f"Disentanglement score: {latent_exp.disentanglement_score:.4f}")

print(f"\nTop 3 important dimensions:")
top_dims = np.argsort(latent_exp.latent_dims_importance)[-3:][::-1]
for dim in top_dims:
    print(f"  Dim {dim}: {latent_exp.latent_dims_importance[dim]:.4f}")

print(f"\nInterpretable directions:")
for direction in latent_exp.interpretable_directions.keys():
    print(f"  - {direction}")

print("\n6.2 GAN Generation Explanation")
print("-" * 100)

z_gan = np.random.randn(512)
gen_img = np.random.rand(256, 256, 3)

gan_exp = GANExplainer(None)
gan_result = gan_exp.explain_generation(z_gan, gen_img)

print(f"Latent code size: {len(gan_result['latent_code'])}")
print(f"\nLatent controls:")
for attr, dims in gan_result['latent_controls'].items():
    sens = gan_result['sensitivity'][attr]
    print(f"  {attr} (dims {dims}): sensitivity {sens:.4f}")

print("\n6.3 StyleGAN Disentanglement")
print("-" * 100)

w = np.random.randn(18, 512)
style_exp = StyleGANExplainer(None)
style_result = style_exp.explain_style_control(w)

print(f"StyleGAN W space: {style_result['n_layers']} layers")
print(f"\nLayer controls:")
for layers, control in style_result['layer_controls'].items():
    print(f"  {layers}: {control}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("TIER 1 ADVANCED FEATURES DEMO COMPLETE!")
print("=" * 100)

print("\nXPLIA now includes:")
print("  ✅ Multimodal AI (CLIP, Stable Diffusion, DALL-E)")
print("  ✅ Graph Neural Networks (GNN, Molecular graphs, Drug discovery)")
print("  ✅ Reinforcement Learning (Policy, Q-values, Trajectories)")
print("  ✅ Advanced Counterfactuals (Minimal, Feasible, Diverse, Actionable)")
print("  ✅ Time Series (Temporal importance, Forecasting, Anomaly detection)")
print("  ✅ Generative Models (VAE, GAN, StyleGAN latent spaces)")

print("\n🎯 XPLIA is now THE MOST COMPREHENSIVE XAI library in the world!")
print("=" * 100)
