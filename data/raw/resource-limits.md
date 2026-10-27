# Resource Requests and Limits
Resource requests and limits specify CPU and memory allocations for containers.
Requests represent the minimum resources guaranteed to a container, used by the scheduler for node placement.
Limits represent the hard ceiling: if a container exceeds its memory limit, it is terminated with an OOMKilled error; CPU exceeding limits is throttled.
