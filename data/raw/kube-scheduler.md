# Kube-Scheduler
The kube-scheduler is the control plane component responsible for assigning newly created pods to optimal nodes within the cluster.
It monitors unscheduled pods and filters nodes according to resource requests, node selectors, taints, tolerations, and affinity rules.
During the scoring phase, the scheduler ranks feasible nodes using weighted criteria such as resource balance and pod topology spread to select the best node for placement.
