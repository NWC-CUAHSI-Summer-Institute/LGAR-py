# # Plotting Code:
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# import pickle
#
# # Load layer depths and water content data
# with open('/Users/taddbindas/projects/dpLGAR/layer_data.pkl', 'rb') as f:
#     layer_data = pickle.load(f)
#
# with open('/Users/taddbindas/projects/dpLGAR/water_content.pkl', 'rb') as f:
#     water_content = pickle.load(f)
#
# fig, axs = plt.subplots(figsize=(6,4), facecolor='w', edgecolor='k', gridspec_kw={'width_ratios':[1]})
# fig.subplots_adjust(hspace =.02, wspace=1.02)
#
# axs.axhline(0, color='k', lw=1)  # Add a horizontal line at depth = 0
#
# time_txt = axs.text(0.2, 0.07, "", horizontalalignment='center', verticalalignment='center', transform = axs.transAxes, fontsize=13)
#
# axs.set_ylim(0, max(layer_data[0]))
# axs.set_xlim(0, 0.5)
# axs.invert_yaxis()
# axs.tick_params(axis='both', labelsize=12, rotation=0)
#
# axs.set_ylabel('Depth (cm)',fontsize=12)
# axs.set_xlabel(r'Water content, $\theta$ (-)',fontsize=12)
#
# def data_gen(t=0):
#     while t < len(layer_data):
#         yield t, layer_data[t], water_content[t]
#         t += 1
#
# def run(data):
#     t, d, d2 = data
#
#     # Clear the previous rectangles
#     for rect in axs.patches:
#         rect.remove()
#
#     # Draw the rectangles
#     for i in range(len(d)):
#         if i == 0:
#             rect = plt.Rectangle((0, 0), d2[i], d[i], fc='None', ec='black', lw=1)
#         else:
#             rect = plt.Rectangle((0, d[i-1]), d2[i], (d[i]-d[i-1]), fc='None', ec='black', lw=1)
#             # Remove the bottom edge of the rectangle
#             rect.set_clip_path(plt.Rectangle((0, 0), d2[i], d[i-1]))
#         axs.add_patch(rect)
#
#     time_txt.set_text(str(t) + ' (h)')
#
#     return [time_txt]
#
# # Smaller interval values will result in a faster animation
# ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=1000, repeat=False, save_count=len(layer_data))
#
# plt.tight_layout(pad=1)
#
# # Lower dpi values will result in a lower resolution
# ani.save('/Users/taddbindas/projects/dpLGAR/animation2.gif', writer='pillow', fps=30, dpi=100)
#
# plt.show()