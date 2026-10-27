# StorageClasses
A StorageClass provides a way for administrators to describe the classes of storage they offer.
Different classes might map to quality-of-service levels, backup policies, or storage backend types like AWS EBS, Azure Disk, or local NVMe.
StorageClasses enable dynamic volume provisioning when PersistentVolumeClaims request storage without pre-allocated PersistentVolumes.
