import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (8,6),dpi = 320)

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = [['Embeddings','pearsonR'],
        ['OHV','0.56'],
        ['ESM_T6','0.58'],
        ['ESM_T30','0.58'],
        ['Antiberty','0.57'],
]

plt.table(table,colWidths = [0.15] * 2,cellLoc = 'center',loc = 'center')
plt.tight_layout()
plt.savefig('./figures/exactgptable.png')
