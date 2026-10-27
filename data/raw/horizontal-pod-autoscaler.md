# Horizontal Pod Autoscaler (HPA)
The Horizontal Pod Autoscaler automatically updates a workload resource, such as a Deployment or StatefulSet, with the aim of automatically scaling the workload to match demand.
HPA adjusts the number of replicas based on observed CPU utilization, memory metrics, or custom metrics provided by the metrics-server or Prometheus adapter.
