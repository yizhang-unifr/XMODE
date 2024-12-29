import matplotlib.pyplot as plt
# Data
centuries = [16, 18]
paintings_count = [1, 2]
#  Plot
plt.figure(figsize=(8, 6))
plt.bar(centuries, paintings_count, color='skyblue')
plt.xlabel('Century')
plt.ylabel('Number of War Paintings')
plt.title('Number of War Paintings by Century')
plt.xticks(centuries)
plt.savefig('plot-out.png')
plt.close()