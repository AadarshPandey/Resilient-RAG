# Taints and Tolerations
Taints allow a node to repel a set of pods unless the pod has a matching toleration.
Taints consist of a key, value, and effect such as NoSchedule, PreferNoSchedule, or NoExecute.
Tolerations are specified in PodSpecs and allow pods to schedule onto nodes with matching taints, enabling dedicated node isolation.
