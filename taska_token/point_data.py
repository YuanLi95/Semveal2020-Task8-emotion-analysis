import  numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

epochs =np.array([2,4,6,8,10,12,14,16,18,20]).astype(int)
epochs_f1 = np.array([0.2347,0.2125,0.2203,0.2606,0.2741,0.3359,0.3557,0.3011,0.2722,0.321])

batch_size = np.array([40,60,80,100,120,140])
batch_size_f1 = np.array([0.275,0.3557,0.3264,0.3359,0.2896,0.3147])

pdf = PdfPages('subpicture1.pdf')
plt.ylabel('f1-score on dev set ')
plt.xlabel('Iteration')
my_x_ticks = np.arange(0,22, 2)
plt.xticks(my_x_ticks)
max_indx=np.argmax(epochs_f1)
plt.plot(epochs[max_indx],epochs_f1[max_indx],'ks')
show_max='['+str(epochs[max_indx])+' '+str(epochs_f1[max_indx])+']'
plt.annotate(show_max,xytext=(epochs[max_indx],epochs_f1[max_indx]),xy=(epochs[max_indx],epochs_f1[max_indx]))

plt.plot(epochs,epochs_f1, color="r", linestyle="-.", marker="*", linewidth=1) # 画图
pdf.savefig()
plt.close()
pdf.close()
# foo_fig = plt.gcf() # 'get current figure'
# foo_fig.savefig('subpicture1.eps', format='eps', dpi=1000)
plt.show()



pdf = PdfPages('subpicture3-2.pdf')
plt.ylabel('f1-score on dev set ')
plt.xlabel('Batch_size')
max_indx=np.argmax(batch_size_f1)
plt.plot(batch_size[max_indx],batch_size_f1[max_indx],'ks')
show_max='['+str(batch_size[max_indx])+' '+str(batch_size_f1[max_indx])+']'
plt.annotate(show_max,xytext=(batch_size[max_indx],batch_size_f1[max_indx]),xy=(batch_size[max_indx],batch_size_f1[max_indx]))

plt.plot(batch_size,batch_size_f1, color="b", linestyle="-.", marker="*", linewidth=1) # 画图
# foo_fig = plt.gcf() # 'get current figure'
# foo_fig.savefig('subpicture2.eps', format='eps', dpi=1000)
pdf.savefig()
plt.close()
pdf.close()
plt.show()