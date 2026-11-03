import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'axes.edgecolor':'#9a9994','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
BLUE,ORANGE,GREY='#2a78d6','#eb6834','#8a8984'
fig,(a,b)=plt.subplots(1,2,figsize=(7.0,2.8),dpi=300)
v=['B0','B1','B2','B3','B4','B5*']; tok=[6072,6388,6666,18374,16064,5900]
acc=[35.8,42.1,40.0,32.9,53.3,39.2]; lo=[30.0,36.0,34.0,27.3,47.0,30.9]; hi=[42.1,48.4,46.3,39.1,59.5,48.1]
col={'B0':BLUE,'B4':ORANGE}
for x,y,l,h,n in zip(tok,acc,lo,hi,v):
    c=col.get(n,GREY)
    a.errorbar(x/1000,y,yerr=[[y-l],[h-y]],fmt='o',color=c,ms=5,elinewidth=1,capsize=2,mec='white',mew=0.8,zorder=3)
off={'B0':(-1.6,-3.0),'B1':(0.5,1.5),'B2':(0.5,-3.5),'B3':(0.6,0),'B4':(0.6,0),'B5*':(-1.8,2.5)}
for x,y,n in zip(tok,acc,v):
    dx,dy=off[n]; a.text(x/1000+dx,y+dy,n,color='#0b0b0b',fontsize=8,va='center')
a.set_xlabel('Tokens per instance (thousands)'); a.set_ylabel('Exact-instance accuracy (%)')
a.set_title('(a) Accuracy against cost',fontsize=9,loc='left'); a.set_xlim(3,21); a.set_ylim(20,65)
bands=['12–14','16–18','20–22','24–26']
b0=[45.0,43.3,26.7,28.3]; b4=[68.3,61.7,43.3,40.0]
b.plot(bands,b4,'-o',color=ORANGE,lw=2,ms=5,mec='white',mew=0.8,label='B4 critique and revision')
b.plot(bands,b0,'-s',color=BLUE,lw=2,ms=5,mec='white',mew=0.8,label='B0 baseline')
b.text(3.08,40.0,'B4',va='center',fontsize=8); b.text(3.08,28.3,'B0',va='center',fontsize=8)
b.set_xlabel('Rooms'); b.set_ylabel('Exact-instance accuracy (%)'); b.set_ylim(20,75); b.set_xlim(-0.2,3.4)
b.set_title('(b) Accuracy by building size',fontsize=9,loc='left'); b.legend(frameon=False,fontsize=7.5,loc='upper right')
for ax in (a,b):
    ax.grid(axis='y',color='#e6e5e1',lw=0.6); ax.set_axisbelow(True)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
fig.tight_layout(w_pad=2.5)
fig.savefig(str(__import__('pathlib').Path(__file__).resolve().parent/'figure1_tradeoff.png'),dpi=300)
