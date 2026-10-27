# Persistent Volume Claims (PVC)
A PersistentVolumeClaim (PVC) is a request for storage by a user.
It specifies the requested storage size, access modes (such as ReadWriteOnce or ReadWriteMany), and StorageClass.
The control plane binds matching PersistentVolumes to PVCs, which are then mounted into Pod containers.
