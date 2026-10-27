# PriorityClasses
A PriorityClass is an object that maps a priority name to an integer priority value.
Pods specify a PriorityClassName in their PodSpec; when scheduling queue conflicts arise or nodes run out of capacity, higher-priority pods can preempt lower-priority pods to claim resources.
