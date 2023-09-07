import matplotlib.pyplot as plt

# List of model names
model_names = ["YOLOv5", "YOLOv8", "SSDLite32", "ResNet-50"]

# Accuracy scores for each model (example values, replace with your actual scores)
accuracy_scores = [0.82, 0.93, 0.95, 0.98]

# Efficiency scores for each model (example values, replace with your actual scores)
efficiency_scores = [0.78, 0.87, 0.84, 0.88]

# Create subplots for accuracy and efficiency
plt.figure(figsize=(12, 5))

# Subplot for accuracy
plt.subplot(1, 2, 1)
plt.bar(model_names, accuracy_scores, color='skyblue')
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')

# Subplot for efficiency
plt.subplot(1, 2, 2)
plt.bar(model_names, efficiency_scores, color='lightcoral')
plt.title('Efficiency Comparison')
plt.ylabel('Efficiency')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
