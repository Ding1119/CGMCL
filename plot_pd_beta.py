import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the text file
data = pd.read_csv('beta_performance.txt')

# Extract beta and AUC values
# Extract beta and AUC values
beta = data['beta']
auc = data['auc']

# Create the plot
plt.figure(figsize=(15, 10))

plt.plot(beta, auc, marker='o', linestyle='-', color='blue', label='AUC Performance')

# Add labels and title
plt.xlabel(r'$\beta$', fontsize=25)
plt.ylabel('AUC', fontsize=25)
plt.title('Beta vs AUC Performance', fontsize=25)
plt.ylim(0.6, 0.8)
plt.legend(fontsize=18)
plt.grid(True)

# Set y-axis ticks to two decimal places
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# Enlarge tick labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Save the plot
plt.savefig('beta_auc_performance.png')

# Show the plot
plt.show()
####################### Example plot ############
# import matplotlib.pyplot as plt

# # Data for the plot
# beta = [0.01, 0.05, 0.1]
# alpha_0_1 = [89.26, 88.97, 88.78]
# alpha_0_5 = [89.73, 89.33, 88.47]
# alpha_1 = [88.61, 90.23, 89.85]

# # Create the plot
# plt.figure(figsize=(6, 4))

# plt.plot(beta, alpha_0_1, marker='D', linestyle='-', color='green', label=r'$\alpha=0.1$')
# plt.plot(beta, alpha_0_5, marker='s', linestyle='-', color='blue', label=r'$\alpha=0.5$')
# plt.plot(beta, alpha_1, marker='^', linestyle='-', color='orange', label=r'$\alpha=1$')

# # Add annotations
# plt.text(0.01, 89.26, '89.26')
# plt.text(0.05, 88.97, '88.97')
# plt.text(0.1, 88.78, '88.78')
# plt.text(0.01, 89.73, '89.73')
# plt.text(0.05, 89.33, '89.33')
# plt.text(0.1, 88.47, '88.47')
# plt.text(0.01, 88.61, '88.61')
# plt.text(0.05, 90.23, '90.23')
# plt.text(0.1, 89.85, '89.85')

# # Add labels and title
# plt.xlabel(r'$\beta$')
# plt.ylabel('AP (%)')
# plt.ylim(88, 91)
# plt.legend()
# plt.grid(True)

# Save the plot
plt.savefig('replicated_plot.png')

# Show the plot
plt.show()
