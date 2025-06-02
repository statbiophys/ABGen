import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (8,6),dpi = 160)

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = [['Method','pearsonR'],
        ['HW with SASA','0.402'],
        ['HW without SASA','0.156'],
        ['Camsol','0.37'],
        ['Biopython','0.05'],
]

plt.table(table,colWidths = [0.3,0.3],cellLoc = 'center',loc = 'center')
plt.tight_layout()
plt.savefig('./solubilitytable.png')