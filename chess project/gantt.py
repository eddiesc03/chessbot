import matplotlib.pyplot as plt

# Second set of tasks (to appear above)
tasks2 = [
    
    ("WP 1.1: Create project plan", 1, 3),
    ("WP 2.1: Order components", 3, 1),
    ("WP 1.2: Learn software skills", 3, 2),
    ("WP 2.2: Assemble components", 4, 1),
    ("WP 3: Computer vision software", 5, 3),
    ("WP 1.3: Crowdfunding pitch", 8, 3),
    ("WP 4: AI piece detection software", 8, 4),
    ("WP 5: Chess engine software", 12, 1)
]

# First set of tasks
tasks1 = [
    
    ("WP 6.1: Combine sofware", 1, 1),
    ("WP 6.2: Enable AI to play as white", 2, 1),
    ("WP 6.3: Improve accuracy/speed of AI", 3, 2),
    ("WP 1.4: Write project abstract", 5, 2),
    ("WP 1.5: Write project report", 6, 3.5),
    ("WP 1.6: Bench inspection", 8, 1)
]

# Create figure and subplots
fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(12, 9))

# Plot second chart (appears on top)
for i, (task, start, duration) in enumerate(tasks2):
    ax2.barh(i, duration, left=start, height=0.6, color='lightgreen', edgecolor='black')
ax2.set_yticks(range(len(tasks2)))
ax2.set_yticklabels([t[0] for t in tasks2])
ax2.invert_yaxis()
ax2.set_title("Semester 1 Gantt chart")
ax2.set_xlabel("Time (s1 weeks)\n\n\n\n")

# Plot first chart (appears below)
for i, (task, start, duration) in enumerate(tasks1):
    ax1.barh(i, duration, left=start, height=0.6, color='skyblue', edgecolor='black')
ax1.set_yticks(range(len(tasks1)))
ax1.set_yticklabels([t[0] for t in tasks1])
ax1.invert_yaxis()
ax1.set_title("Semester 2 Gantt chart")
ax1.set_xlabel("Time (s2 weeks)")

ax1.axvspan(8.93, 9.07, color='lightgray', alpha=0.5, label='Easter Break')
ax1.legend(loc='upper right')


ax1.set_xlim(left=0)
ax2.set_xlim(left=0)

# Grid and layout
ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()