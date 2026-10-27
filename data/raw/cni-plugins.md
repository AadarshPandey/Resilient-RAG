# Container Network Interface (CNI)
The Container Network Interface (CNI) defines a standardized specification and set of libraries for configuring network interfaces in Linux containers.
Kubernetes uses CNI plugins, such as Calico, Cilium, or Flannel, to allocate IP addresses to pods and manage network connectivity between pods across different worker nodes.
CNI ensures that every pod receives a unique, routable IP address without requiring network address translation within the cluster network.
