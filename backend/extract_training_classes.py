import json
from torchvision import datasets
from torchvision import transforms

# CHANGE THIS to your actual training dataset root
TRAIN_DATASET_PATH = r"C:\plant_dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"

dataset = datasets.ImageFolder(
    root=TRAIN_DATASET_PATH,
    transform=transforms.ToTensor()
)

print("Number of classes:", len(dataset.classes))
print("\nCLASS ORDER USED DURING TRAINING:\n")

for idx, cls in enumerate(dataset.classes):
    print(f"{idx:02d}: {cls}")

# Optional: save for diffing
with open("training_classes.json", "w") as f:
    json.dump(dataset.classes, f, indent=2)

print("\nSaved to training_classes.json")
